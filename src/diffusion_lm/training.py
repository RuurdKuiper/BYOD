"""Accelerate training/evaluation with reproducible denoising validation."""
from __future__ import annotations
import json
import atexit
import hashlib
import importlib.util
import math
import os
from pathlib import Path
from statistics import median
from typing import Any

import torch
from tqdm.auto import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_polynomial_decay_schedule_with_warmup, get_scheduler

from .data import DenoisingCollator, llama_stored_ids_compatible, prepare_mask_only_cache_record, stored_example_usable
from .loss import masked_denoising_loss, selected_denoising_loss
from .modeling import forward_bidirectional, forward_bidirectional_selected, load_denoising_model, parameter_audit
from .inference import InferenceSession, _native_eot_token_id, llada_generate
from .generation_prompts import DEFAULT_GENERATION_PROMPTS, _load_generation_prompts
from .metrics import distinct_n


def _write_json(path: Path, value: Any) -> None:
    """Write one structured artifact with deterministic formatting."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _append_jsonl(path: Path, value: Any) -> None:
    """Append one metrics record to a JSON Lines file."""
    with path.open("a") as f:
        f.write(json.dumps(value, default=float) + "\n")


def _save_adapter(model, tokenizer, path: Path, initial_norms: dict[str, torch.Tensor] | None = None) -> None:
    """Save LoRA plus independently-unfrozen norm parameters for inference."""
    model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
    norm_state = {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if "norm" in name.lower()}
    torch.save(norm_state, path / "normalization_state.pt")
    if initial_norms is not None:
        torch.save(initial_norms, path / "normalization_initial_state.pt")
    tokenizer.save_pretrained(path)


def _loader(dataset, collator, batch_size, shuffle, seed, workers, prefetch_factor=4):
    """Build a reproducibly shuffled DataLoader using the supplied collator."""
    generator = torch.Generator().manual_seed(seed)
    kwargs = {}
    if workers:
        if int(prefetch_factor) < 1:
            raise ValueError("prefetch_factor must be positive")
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator, num_workers=workers, generator=generator, pin_memory=torch.cuda.is_available(), **kwargs)


def _normalization_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone all normalization parameters so base-model evaluation can restore them."""
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if "norm" in name.lower()}


def _load_normalization_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Copy a saved normalization state into a model without changing adapters."""
    current = dict(model.named_parameters())
    for name, value in state.items():
        if name in current:
            current[name].data.copy_(value.to(current[name].device, dtype=current[name].dtype))


@torch.no_grad()
def _base_perplexity(model: torch.nn.Module, tokenizer: Any, texts: list[str], initial_norms: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
    """Score generated texts with the original base model, excluding LoRA and trained norms.

    The aggregate metrics are token-weighted across all texts.  Individual
    perplexities are also retained so callers can associate them with the
    corresponding generated text.
    """
    trained_norms = _normalization_state(model)
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    per_text_perplexities: list[float | None] = []
    try:
        _load_normalization_state(model, initial_norms)
        with model.disable_adapter():
            for text in texts:
                encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
                input_ids = encoded["input_ids"].to(device)
                if input_ids.shape[1] < 2:
                    per_text_perplexities.append(None)
                    continue
                outputs = model(input_ids=input_ids, use_cache=False)
                logits = outputs.logits[:, :-1].float()
                labels = input_ids[:, 1:]
                nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction="sum")
                text_nll = float(nll.cpu())
                text_tokens = int(labels.numel())
                total_nll += text_nll
                total_tokens += text_tokens
                per_text_perplexities.append(float(torch.exp(torch.tensor(text_nll / text_tokens))))
    finally:
        _load_normalization_state(model, trained_norms)
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "generation_perplexity": float(torch.exp(torch.tensor(mean_nll))),
        "generation_mean_nll": mean_nll,
        "generation_tokens": total_tokens,
        "_per_text_perplexities": per_text_perplexities,
    }


def _generation_inference_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Use the shared open-ended protocol for intermediate generation validation."""
    settings = {
        "sampler": "llada_official",
        "num_prompts": len(DEFAULT_GENERATION_PROMPTS),
        "max_new_tokens": 128,
        "num_steps": 64,
        "block_length": 128,
        "temperature": 0.7,
        "confidence_eos_eot_inf": True,
        "system_prompt": "",
        "seed": 1234,
    }
    settings.update(config.get("generation_perplexity", {}))
    if settings["sampler"] != "llada_official":
        raise ValueError("Training generation validation requires sampler=llada_official")
    # These behaviors are intrinsic to the official low-confidence sampler.
    settings.update(permanent_unmask=True, confidence_guided=True,
                    proportional_unmask=False, remasking="low_confidence")
    return settings


def _generation_perplexity_interval(config: dict[str, Any]) -> int:
    """Resolve an exact generation interval aligned with loss validation."""
    validation_steps = int(config.get("validation_steps", 100))
    interval = int(config.get("generation_perplexity", {}).get("interval_steps", validation_steps))
    if validation_steps < 1:
        raise ValueError("validation_steps must be positive")
    if interval < 1:
        raise ValueError("generation_perplexity.interval_steps must be positive")
    if interval % validation_steps:
        raise ValueError("generation_perplexity.interval_steps must be a multiple of validation_steps")
    return interval


def _resolve_learning_rate(config: dict[str, Any], num_processes: int = 1) -> tuple[float, int, float]:
    """Resolve optional square-root or linear scaling from a reference batch size."""
    base_learning_rate = float(config["learning_rate"])
    if base_learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    batch_size = int(config["batch_size"])
    gradient_accumulation = int(config.get("gradient_accumulation_steps", 1))
    if batch_size < 1 or gradient_accumulation < 1 or num_processes < 1:
        raise ValueError("batch_size, gradient_accumulation_steps, and num_processes must be positive")
    effective_batch_size = batch_size * gradient_accumulation * num_processes

    settings = config.get("learning_rate_scaling", {}) or {}
    if not isinstance(settings, dict):
        raise ValueError("learning_rate_scaling must be a mapping")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("learning_rate_scaling.enabled must be true or false")
    mode = str(settings.get("mode", "sqrt")).lower()
    if mode not in {"sqrt", "linear"}:
        raise ValueError("learning_rate_scaling.mode must be 'sqrt' or 'linear'")
    reference_batch_size = int(settings.get("reference_batch_size", 8))
    if reference_batch_size < 1:
        raise ValueError("learning_rate_scaling.reference_batch_size must be positive")

    batch_ratio = effective_batch_size / reference_batch_size
    scale = (math.sqrt(batch_ratio) if mode == "sqrt" else batch_ratio) if enabled else 1.0
    return base_learning_rate * scale, effective_batch_size, scale


def _build_learning_rate_scheduler(
    config: dict[str, Any], optimizer: torch.optim.Optimizer, max_updates: int, initial_learning_rate: float
):
    """Build the configured scheduler, optionally with a nonzero linear endpoint."""
    scheduler_name = str(config.get("scheduler", "linear")).lower()
    warmup_steps = int(config.get("warmup_steps", 0))
    end_learning_rate = config.get("end_learning_rate")
    if end_learning_rate is None:
        return get_scheduler(scheduler_name, optimizer, warmup_steps, max_updates)
    if scheduler_name != "linear":
        raise ValueError("end_learning_rate is supported only with scheduler=linear")
    end_learning_rate = float(end_learning_rate)
    if end_learning_rate < 0 or end_learning_rate > initial_learning_rate:
        raise ValueError("end_learning_rate must be between zero and the resolved learning_rate")
    return get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_updates,
        lr_end=end_learning_rate,
        power=1.0,
    )


def _native_fp8_capability(capability: tuple[int, int]) -> bool:
    """Return whether NVIDIA Transformer Engine supports native FP8 on this GPU."""
    major, minor = capability
    return (major, minor) == (8, 9) or major >= 9


def _resolve_fp8(
    config: dict[str, Any],
    *,
    cuda_available: bool | None = None,
    capability: tuple[int, int] | None = None,
    device_name: str | None = None,
    transformer_engine_available: bool | None = None,
) -> dict[str, Any]:
    """Resolve hardware-gated Transformer Engine FP8 or a BF16 fallback."""
    settings = config.get("fp8", {}) or {}
    if not isinstance(settings, dict):
        raise ValueError("fp8 must be a mapping")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("fp8.enabled must be true or false")
    base_precision = str(config.get("precision", "bf16")).lower()
    if not enabled:
        return {
            "requested": False,
            "active": False,
            "model_precision": base_precision,
            "mixed_precision": None if base_precision == "fp32" else base_precision,
            "device_name": None,
            "capability": None,
            "notice": None,
        }

    backend = str(settings.get("backend", "transformer_engine")).lower()
    if backend not in {"transformer_engine", "te"}:
        raise ValueError("fp8.backend must be 'transformer_engine'")
    cuda_available = torch.cuda.is_available() if cuda_available is None else cuda_available
    if cuda_available:
        capability = torch.cuda.get_device_capability() if capability is None else capability
        device_name = torch.cuda.get_device_name() if device_name is None else device_name
    supported = bool(cuda_available and capability is not None and _native_fp8_capability(capability))
    if not supported:
        description = "no CUDA GPU" if not cuda_available else f"{device_name or 'CUDA GPU'} (compute capability {capability[0]}.{capability[1]})"
        return {
            "requested": True,
            "active": False,
            "model_precision": "bf16",
            "mixed_precision": "bf16",
            "device_name": device_name,
            "capability": capability,
            "notice": f"FP8 requested, but {description} does not provide supported native FP8 training; falling back to BF16.",
        }

    if base_precision not in {"fp16", "bf16"}:
        raise ValueError("FP8 training requires precision to be fp16 or bf16 for master weights")
    if transformer_engine_available is None:
        transformer_engine_available = importlib.util.find_spec("transformer_engine") is not None
    if not transformer_engine_available:
        raise ImportError(
            "FP8-capable GPU detected, but NVIDIA Transformer Engine is not installed. "
            "Install it with `pip install -e '.[fp8]'`."
        )
    return {
        "requested": True,
        "active": True,
        "model_precision": base_precision,
        "mixed_precision": "fp8",
        "device_name": device_name,
        "capability": capability,
        "notice": None,
    }


def _available_output_dir(path: Path) -> Path:
    """Return path or the next unused suffixed sibling without modifying it."""
    if not path.exists():
        return path
    suffix = 1
    while True:
        candidate = path.parent / f"{path.name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def generation_validation(model: torch.nn.Module, tokenizer: Any, mask_token_id: int, config: dict[str, Any], initial_norms: dict[str, torch.Tensor], device: torch.device, output: Path, step: int) -> dict[str, float]:
    """Generate fixed prompts, save final answers, and calculate base perplexity."""
    settings = _generation_inference_settings(config)
    prompts = settings.get("prompts", DEFAULT_GENERATION_PROMPTS)
    session = InferenceSession(model, tokenizer, device, output, config, mask_token_id, str(config.get("quantization", "none")))
    records = []
    finals = []
    model.eval()
    for prompt_index, prompt in enumerate(prompts[: int(settings["num_prompts"])]):
        final_text = llada_generate(
            session, prompt,
            gen_length=int(settings["max_new_tokens"]),
            steps=int(settings["num_steps"]),
            block_length=int(settings["block_length"]),
            temperature=float(settings["temperature"]),
            remasking=settings["remasking"],
            confidence_eos_eot_inf=bool(settings["confidence_eos_eot_inf"]),
            eot_token_id=_native_eot_token_id(tokenizer),
            system_prompt=str(settings["system_prompt"]),
            seed=int(settings["seed"]) + prompt_index,
        )
        finals.append(final_text)
        records.append({"step": step, "prompt_index": prompt_index, "distinct_1": distinct_n(final_text, tokenizer, 1), "distinct_2": distinct_n(final_text, tokenizer, 2), "distinct_3": distinct_n(final_text, tokenizer, 3), "prompt": prompt, "final": final_text})
    generation_metrics = _base_perplexity(model, tokenizer, finals, initial_norms, device)
    per_text_perplexities = generation_metrics.pop("_per_text_perplexities")
    valid_perplexities = [value for value in per_text_perplexities if value is not None]
    generation_metrics["generation_mean_perplexity"] = (
        float(sum(valid_perplexities) / len(valid_perplexities)) if valid_perplexities else None
    )
    generation_metrics["generation_median_perplexity"] = (
        float(median(valid_perplexities)) if valid_perplexities else None
    )
    for n in (1, 2, 3):
        generation_metrics[f"generation_mean_distinct_{n}"] = float(
            sum(record[f"distinct_{n}"] for record in records) / len(records)
        ) if records else None
    for record in records:
        # Rebuild the mapping to keep the JSONL field order stable/readable.
        record["generation_perplexity"] = per_text_perplexities[record["prompt_index"]]
        ordered = {"step": record["step"], "prompt_index": record["prompt_index"], "distinct_1": record["distinct_1"], "distinct_2": record["distinct_2"], "distinct_3": record["distinct_3"], "generation_perplexity": record["generation_perplexity"], "prompt": record["prompt"], "final": record["final"]}
        record.clear(); record.update(ordered)
    generation_path = output / "generation_metrics.jsonl"
    with generation_path.open("a") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return generation_metrics

def _resolve_answer_padding_weights(config: dict[str, Any]) -> tuple[float, float] | None:
    """Validate the optional separately normalized masked-response objective."""
    settings = config.get("answer_padding_loss", {})
    if not isinstance(settings, dict):
        raise ValueError("answer_padding_loss must be a mapping")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("answer_padding_loss.enabled must be true or false")
    if not enabled:
        return None
    if config.get("corruption_mode") != "mask_only" or config.get("structured_loss_behavior", "all_answer_tokens") not in {
        "all_answer_tokens", "corrupted_answer_tokens",
    }:
        raise ValueError("answer_padding_loss requires mask_only with a corrupted-answer objective, not all_tokens")
    if config.get("eos_padding_loss") is not True:
        raise ValueError("answer_padding_loss requires eos_padding_loss=true")
    weights = tuple(float(settings.get(key, default)) for key, default in (
        ("answer_weight", 0.9), ("padding_weight", 0.1),
    ))
    if not all(math.isfinite(weight) and weight >= 0 for weight in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("answer_padding_loss weights must be finite, nonnegative, and sum to 1")
    return weights


@torch.no_grad()
def evaluate(model, loader, accelerator, mode: str, all_tokens: bool = False, eos_padding_loss: bool = False,
             answer_padding_weights: tuple[float, float] | None = None) -> dict[str, float]:
    """Evaluate deterministic denoising loss and aggregate metrics across ranks."""
    model.eval()
    totals = {"weighted_loss_sum": 0.0, "unweighted_ce_sum": 0.0, "valid_examples": 0, "supervised_tokens": 0, "eligible_answer_tokens": 0, "masked_tokens": 0, "t_sum": 0.0, "t_count": 0}
    if answer_padding_weights is not None:
        totals.update(answer_loss_sum=0.0, padding_loss_sum=0.0)
    for batch in loader:
        logits = forward_bidirectional(model, batch["input_ids"], batch["padding_mask"])
        t = batch["sampled_t"] if mode == "mask_only" and not all_tokens else None
        normalization_mask = batch["answer_mask"] | batch["padding_mask"] if eos_padding_loss else batch["answer_mask"]
        loss, m = masked_denoising_loss(
            logits,
            batch["labels"],
            batch["loss_mask"],
            t,
            normalization_mask,
            sparse_positions=not all_tokens,
            answer_padding_weights=answer_padding_weights,
            answer_mask=batch["answer_mask"], padding_mask=batch["padding_mask"],
            token_weights=batch.get("token_loss_weights") if t is not None else None,
        )
        valid = int(m["valid_examples"])
        tokens = int(m["supervised_tokens"])
        totals["weighted_loss_sum"] += float(loss) * valid
        totals["unweighted_ce_sum"] += float(m["unweighted_masked_token_ce"]) * tokens
        totals["valid_examples"] += valid
        totals["supervised_tokens"] += tokens
        if answer_padding_weights is not None:
            totals["answer_loss_sum"] += float(m["answer_loss"]) * valid
            totals["padding_loss_sum"] += float(m["padding_loss"]) * valid
        totals["eligible_answer_tokens"] += int((batch["answer_mask"] & ~batch["padding_mask"]).sum())
        totals["masked_tokens"] += int(batch["loss_mask"].sum())
        if mode == "mask_only":
            totals["t_sum"] += float(torch.nansum(batch["sampled_t"]))
            totals["t_count"] += len(batch["sampled_t"])
    keys = list(totals)
    totals = accelerator.reduce(torch.tensor([totals[k] for k in keys], device=accelerator.device), reduction="sum").tolist()
    d = dict(zip(keys, totals))
    d["weighted_loss"] = d["weighted_loss_sum"] / max(d["valid_examples"], 1)
    if answer_padding_weights is not None:
        for component in ("answer", "padding"):
            d[f"{component}_loss"] = d[f"{component}_loss_sum"] / max(d["valid_examples"], 1)
    d["unweighted_masked_token_ce"] = d["unweighted_ce_sum"] / max(d["supervised_tokens"], 1)
    d["realized_masked_fraction"] = d["masked_tokens"] / max(d["eligible_answer_tokens"], 1)
    if mode == "mask_only": d["mean_sampled_t"] = d["t_sum"] / max(d["t_count"], 1)
    return d


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """Execute model setup, training, validation, selection, and final testing."""
    answer_padding_weights = _resolve_answer_padding_weights(config)
    if answer_padding_weights is not None:
        config["answer_padding_loss"] = dict(enabled=True, answer_weight=answer_padding_weights[0],
                                           padding_weight=answer_padding_weights[1])
    storage_root = os.getenv("LAD_STORAGE")
    if storage_root:
        # Relative configured paths become node-local; absolute paths preserve
        # their existing local-execution meaning.
        for key, default in {
            "output_dir": "outputs", "cache_dir": "data/huggingface",
            "base_model_cache_dir": "base_models",
            "prepared_data_cache_dir": "data/prepared",
            "resume_from_checkpoint": None, "resume_from_adapter": None,
        }.items():
            value = config.get(key, default)
            if value and not Path(value).is_absolute():
                config[key] = str(Path(storage_root) / value)
    output_root = os.getenv("LAD_OUTPUT_ROOT")
    if output_root and config.get("output_dir"):
        output_path = Path(config["output_dir"])
        if not output_path.is_absolute():
            config["output_dir"] = str(Path(output_root) / output_path)
        for key in ("resume_from_checkpoint", "resume_from_adapter"):
            value = config.get(key)
            if value and not Path(value).is_absolute():
                config[key] = str(Path(output_root) / value)
    output = Path(config["output_dir"])
    if not config.get("resume_from_checkpoint") and not config.get("resume_from_adapter"):
        output = _available_output_dir(output)
        config["output_dir"] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    configured_updates_hint = config.get("max_updates")
    if configured_updates_hint is None:
        configured_updates_hint = config.get("max_steps")
    train_sample_limit = None
    resume_data_updates = int(config.get("resume_data_updates", 0) or 0)
    if resume_data_updates < 0:
        raise ValueError("resume_data_updates must be non-negative")
    if resume_data_updates and not config.get("resume_from_adapter"):
        raise ValueError("resume_data_updates is only supported with resume_from_adapter")
    if configured_updates_hint is not None:
        configured_updates_hint = int(configured_updates_hint)
        if configured_updates_hint < 1:
            raise ValueError("max_updates must be a positive number of gradient updates")
        train_sample_limit = (configured_updates_hint + resume_data_updates) * int(config.get("gradient_accumulation_steps", 1)) * int(config.get("batch_size", 1))
    fp8_resolution = _resolve_fp8(config)
    config["precision"] = fp8_resolution["model_precision"]
    if fp8_resolution["notice"] and int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(f"WARNING: {fp8_resolution['notice']}", flush=True)
    accelerator_handlers = []
    if fp8_resolution["active"]:
        from accelerate.utils import TERecipeKwargs

        fp8_settings = config.get("fp8", {})
        override_linear_precision = tuple(fp8_settings.get("override_linear_precision", (False, False, False)))
        if len(override_linear_precision) != 3 or not all(isinstance(value, bool) for value in override_linear_precision):
            raise ValueError("fp8.override_linear_precision must contain three booleans")
        accelerator_handlers.append(TERecipeKwargs(
            use_autocast_during_eval=False,
            margin=int(fp8_settings.get("margin", 0)),
            interval=int(fp8_settings.get("interval", 1)),
            fp8_format=str(fp8_settings.get("format", "HYBRID")).upper(),
            amax_history_len=int(fp8_settings.get("amax_history_len", 1024)),
            amax_compute_algo=str(fp8_settings.get("amax_compute_algo", "max")).lower(),
            override_linear_precision=override_linear_precision,
        ))
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            capability = fp8_resolution["capability"]
            print(
                f"FP8 enabled with NVIDIA Transformer Engine on {fp8_resolution['device_name']} "
                f"(compute capability {capability[0]}.{capability[1]}); "
                f"validation remains {fp8_resolution['model_precision'].upper()}.",
                flush=True,
            )
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        mixed_precision=fp8_resolution["mixed_precision"],
        dataloader_config=DataLoaderConfiguration(non_blocking=torch.cuda.is_available()),
        kwargs_handlers=accelerator_handlers,
    )
    # Ensure NCCL process groups are released when a worker is interrupted
    # (for example with Ctrl-C or a scheduler pre-emption signal).
    def _cleanup_process_group() -> None:
        """Destroy the distributed process group during interpreter shutdown."""
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    atexit.register(_cleanup_process_group)
    checkpoint_mode = config.get("checkpoint_mode", "only_best_model")
    if checkpoint_mode not in {"only_best_model", "every_checkpoint", "every_model"}:
        raise ValueError("checkpoint_mode must be 'only_best_model', 'every_model', or 'every_checkpoint'")
    generation_settings = config.get("generation_perplexity", {})
    generation_interval = _generation_perplexity_interval(config) if generation_settings.get("enabled", False) else None
    from datasets import load_dataset
    cache_dir = Path(config.get("cache_dir", "data/huggingface")); cache_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")
    # Serialize the initial dataset download/cache population so distributed
    # workers do not all perform the expensive preparation concurrently.
    with accelerator.main_process_first():
        raw = load_dataset(
            config["dataset_name"],
            config.get("dataset_config"),
            revision=config.get("dataset_revision"),
            cache_dir=str(cache_dir),
            token=hf_token,
        )
    split_names = config.get("splits", {"train": "train", "validation": "validation", "test": "test"})
    token_name = config.get("tokenizer_name_or_path", config["model_name_or_path"])
    model_cache = Path(config.get("base_model_cache_dir", "base_models")); model_cache.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        token_name,
        revision=config.get("model_revision"),
        use_fast=True,
        token=os.getenv("HF_TOKEN"),
        cache_dir=str(model_cache),
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    seed = int(config.get("seed", 42)); torch.manual_seed(seed)
    def indexed(ds):
        """Attach stable original indices for deterministic evaluation corruption."""
        return ds.map(lambda _, index: {"_index": index}, with_indices=True)
    def get_split(spec):
        """Resolve either a named split or a Hugging Face split expression."""
        if spec in raw:
            return raw[spec]
        # Hugging Face split expressions (e.g. train[:8]) are valid smoke-test inputs.
        return load_dataset(
            config["dataset_name"],
            config.get("dataset_config"),
            revision=config.get("dataset_revision"),
            split=spec,
            cache_dir=str(cache_dir),
            token=hf_token,
        )
    def bounded_filter(dataset, predicate, limit: int | None):
        """Filter only as many source rows as needed for a capped run."""
        if limit is None or len(dataset) <= limit:
            return dataset.filter(predicate)
        from datasets import concatenate_datasets
        chunks = []
        kept = 0
        chunk_size = 2048
        for start in range(0, len(dataset), chunk_size):
            chunk = dataset.select(range(start, min(start + chunk_size, len(dataset))))
            filtered = chunk.filter(predicate)
            if len(filtered):
                chunks.append(filtered)
                kept += len(filtered)
            if kept >= limit:
                break
        if not chunks:
            return dataset.select([])
        result = concatenate_datasets(chunks)
        return result.select(range(min(limit, len(result))))

    prep_key = hashlib.sha256(json.dumps({"format": 2, "dataset": config["dataset_name"], "dataset_revision": config.get("dataset_revision"), "config": config.get("dataset_config"), "splits": split_names, "tokenizer": token_name, "model_revision": config.get("model_revision"), "chat_template": getattr(tokenizer, "chat_template", None), "mode": config["corruption_mode"], "max_length": int(config["max_sequence_length"]), "include_answer_eos": bool(config.get("include_answer_eos", True)), "train_sample_limit": train_sample_limit, "seed": seed}, sort_keys=True).encode()).hexdigest()[:16]
    prep_root = Path(config.get("prepared_data_cache_dir", "data/prepared")) / prep_key
    prepared_cache_loaded = False
    with accelerator.main_process_first():
        if all((prep_root / split).is_dir() for split in ("train", "validation", "test")):
            from datasets import load_from_disk
            train_data, val_data, test_data = (load_from_disk(str(prep_root / split)) for split in ("train", "validation", "test"))
            prepared_cache_loaded = True
        else:
            train_data, val_data, test_data = (indexed(get_split(split_names[k])) for k in ("train", "validation", "test"))
    if config["corruption_mode"] == "structured" and prepared_cache_loaded and train_sample_limit is not None and len(train_data) > train_sample_limit:
        train_data = train_data.shuffle(seed=int(config.get("seed", 42))).select(range(train_sample_limit))
    if config["corruption_mode"] != "structured" and not prepared_cache_loaded:
        # The published dataset contains a small number of rows with missing
        # or empty outputs. Remove them before collation, otherwise a worker
        # would fail mid-epoch instead of skipping malformed examples.
        has_output = lambda row: bool((row.get("output") or "").strip())
        train_data = bounded_filter(train_data.shuffle(seed=int(config.get("seed", 42))), has_output, train_sample_limit)
        val_data = val_data.filter(has_output)
        test_data = test_data.filter(has_output)
        preprocessing_workers = int(config.get("preprocessing_num_workers", config.get("num_workers", 1)))
        if preprocessing_workers < 1:
            raise ValueError("preprocessing_num_workers must be positive")
        with accelerator.main_process_first():
            for key, dataset in (("train", train_data), ("validation", val_data), ("test", test_data)):
                original_columns = dataset.column_names
                dataset = dataset.map(
                    lambda row: prepare_mask_only_cache_record(
                        row,
                        tokenizer,
                        int(config["max_sequence_length"]),
                        bool(config.get("include_answer_eos", True)),
                    ),
                    remove_columns=original_columns,
                    num_proc=preprocessing_workers if preprocessing_workers > 1 else None,
                    desc=f"Tokenizing {key} split",
                )
                if key == "train": train_data = dataset
                elif key == "validation": val_data = dataset
                else: test_data = dataset
            prep_root.mkdir(parents=True, exist_ok=True)
            train_data.save_to_disk(str(prep_root / "train"))
            val_data.save_to_disk(str(prep_root / "validation"))
            test_data.save_to_disk(str(prep_root / "test"))
    marker_dropped = {}
    if config["corruption_mode"] == "structured" and not prepared_cache_loaded:
        model_name = token_name.lower()
        if not any(name in model_name for name in ("llama", "meta-llama")):
            raise ValueError("structured mode is supported only for Llama-tokenized data; use mask_only for Qwen/Gemma")
        marker_dropped = {}
        with accelerator.main_process_first():
            if train_sample_limit is not None and len(train_data) > train_sample_limit:
                train_data = train_data.shuffle(seed=int(config.get("seed", 42)))
            for key, dataset in (("train", train_data), ("validation", val_data), ("test", test_data)):
                before = len(dataset)
                limit = train_sample_limit if key == "train" else None
                dataset = bounded_filter(dataset, lambda row: llama_stored_ids_compatible(row, tokenizer) and stored_example_usable(row, tokenizer, int(config["max_sequence_length"]), bool(config.get("include_answer_eos", True))), limit)
                marker_dropped[key] = before - len(dataset)
                if key == "train": train_data = dataset
                elif key == "validation": val_data = dataset
                else: test_data = dataset
            prep_root.mkdir(parents=True, exist_ok=True)
            train_data.save_to_disk(str(prep_root / "train"))
            val_data.save_to_disk(str(prep_root / "validation"))
            test_data.save_to_disk(str(prep_root / "test"))
    validation_limit = config.get("validation_samples", 200)
    if validation_limit is not None:
        validation_limit = min(int(validation_limit), len(val_data))
        val_data = val_data.select(range(validation_limit))
    common = dict(tokenizer=tokenizer, corruption_mode=config["corruption_mode"], max_sequence_length=int(config["max_sequence_length"]), include_answer_eos=bool(config.get("include_answer_eos", True)), pad_to_multiple_of=config.get("pad_to_multiple_of"), structured_loss_behavior=config.get("structured_loss_behavior", "all_answer_tokens"), eos_padding_loss=config.get("eos_padding_loss"), seed=seed, t_min=float(config.get("t_min", .1)), multi_turn_prob=float(config.get("multi_turn_prob", 0.0)), max_history_turns=int(config.get("max_history_turns", 2)), mask_token=str(config.get("mask_token", "MASK")), frontier_padding_mode=str(config.get("frontier_padding_mode", "iid")))
    train_collator = DenoisingCollator(
        **common, deterministic=False,
        frontier_masking_probability=float(config.get("frontier_masking_probability", 0.0)),
        frontier_masking_epsilon=float(config.get("frontier_masking_epsilon", 0.03)),
        frontier_masking_tau=float(config.get("frontier_masking_tau", 3.0)),
    )
    # Keep validation single-turn by default; multi-turn can be enabled
    # explicitly when comparing models on conversational context.
    eval_collator = DenoisingCollator(**common, deterministic=True)
    if resume_data_updates:
        already_seen_examples = resume_data_updates * int(config.get("gradient_accumulation_steps", 1)) * int(config.get("batch_size", 1))
        if len(train_data) <= already_seen_examples:
            raise ValueError(
                "resume_data_updates removes the entire prepared training set; "
                "increase the training sample limit or reduce resume_data_updates"
            )
        # The preparation pipeline uses a stable seed-based shuffle before
        # applying train_sample_limit. Remove the prefix consumed by the
        # original run before constructing the new dataloader; the dataloader
        # may reshuffle the remaining examples freely without reusing them.
        train_data = train_data.select(range(already_seen_examples, len(train_data)))
    prefetch_factor = int(config.get("prefetch_factor", 4))
    train_loader = _loader(train_data.shuffle(seed=seed), train_collator, int(config["batch_size"]), True, seed, int(config.get("num_workers", 0)), prefetch_factor)
    val_loader = _loader(val_data, eval_collator, int(config.get("eval_batch_size", config["batch_size"])), False, seed, int(config.get("num_workers", 0)), prefetch_factor)
    test_loader = _loader(test_data, eval_collator, int(config.get("eval_batch_size", config["batch_size"])), False, seed, int(config.get("num_workers", 0)), prefetch_factor)
    model, audit = load_denoising_model(config)
    initial_norms = _normalization_state(model)
    resolved_learning_rate, effective_batch_size, learning_rate_scale = _resolve_learning_rate(config, accelerator.num_processes)
    resolved = dict(config); resolved["eos_padding_loss"] = train_collator.eos_padding_loss; resolved["frontier_padding_mode"] = train_collator.frontier_padding_mode; resolved["training_samples_used"] = len(train_data); resolved["training_sample_limit"] = train_sample_limit; resolved["validation_samples_used"] = len(val_data); resolved["structured_marker_dropped"] = marker_dropped if config["corruption_mode"] == "structured" else {}; resolved["effective_batch_size"] = effective_batch_size; resolved["learning_rate_scale"] = learning_rate_scale; resolved["resolved_learning_rate"] = resolved_learning_rate; resolved["fp8_requested"] = fp8_resolution["requested"]; resolved["fp8_active"] = fp8_resolution["active"]; resolved["fp8_device_name"] = fp8_resolution["device_name"]; resolved["fp8_compute_capability"] = fp8_resolution["capability"]; resolved["resolved_training_precision"] = fp8_resolution["mixed_precision"] or "fp32"
    _write_json(output / "resolved_config.json", resolved); _write_json(output / "parameter_audit.json", audit); _write_json(output / "mask_token.json", train_collator.mask_info)
    trainable_parameter_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    if optimizer_name in {"adamw8bit", "8bit_adamw", "paged_adamw8bit"}:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("optimizer=adamw8bit requires bitsandbytes; install it on CUDA Linux with `pip install bitsandbytes`") from exc
        optimizer = bnb.optim.AdamW8bit(trainable_parameters, lr=resolved_learning_rate, weight_decay=float(config.get("weight_decay", 0.0)))
    elif optimizer_name == "adamw":
        optimizer_kwargs = {
            "lr": resolved_learning_rate,
            "weight_decay": float(config.get("weight_decay", 0.0)),
        }
        # PyTorch's fused implementation performs the same AdamW update with
        # substantially fewer CUDA kernel launches. Optimizer state is created
        # lazily after Accelerate moves the parameters to the CUDA device.
        if torch.cuda.is_available():
            optimizer_kwargs["fused"] = True
        optimizer = AdamW(trainable_parameters, **optimizer_kwargs)
    else:
        raise ValueError(f"Unknown optimizer={optimizer_name}; expected adamw or adamw8bit")
    grad_accumulation = int(config.get("gradient_accumulation_steps", 1))
    max_grad_norm = config.get("max_grad_norm")
    if max_grad_norm is not None and float(max_grad_norm) <= 0:
        raise ValueError("max_grad_norm must be positive when set")
    # `max_updates` is deliberately expressed in optimizer/gradient updates,
    # rather than dataloader batches. Keep max_steps as a backwards-compatible
    # alias for existing configurations.
    configured_updates = config.get("max_updates")
    if configured_updates is None:
        configured_updates = config.get("max_steps")
    if configured_updates is not None and int(configured_updates) < 1:
        raise ValueError("max_updates must be a positive number of gradient updates")
    max_updates = int(configured_updates) if configured_updates is not None else (len(train_loader) * int(config.get("epochs", 1)) + grad_accumulation - 1) // grad_accumulation
    max_steps = max_updates * grad_accumulation
    scheduler = _build_learning_rate_scheduler(config, optimizer, max_updates, resolved_learning_rate)
    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader, scheduler)
    if fp8_resolution["active"]:
        # Accelerate replaces nn.Linear modules with Transformer Engine modules.
        # Restore the pre-conversion trainable set so frozen base weights do not
        # unexpectedly receive gradients after replacement.
        unwrapped = accelerator.unwrap_model(model)
        for name, parameter in unwrapped.named_parameters():
            parameter.requires_grad_(name in trainable_parameter_names)
        converted_trainable_names = {name for name, parameter in unwrapped.named_parameters() if parameter.requires_grad}
        if converted_trainable_names != trainable_parameter_names:
            raise RuntimeError(
                "Transformer Engine conversion changed parameter names; refusing to train with an incorrect trainable set."
            )
        post_fp8_audit = parameter_audit(unwrapped)
        audit.update({key: value for key, value in post_fp8_audit.items() if key != "trainable_names"})
        audit["fp8_transformer_engine"] = True
        _write_json(output / "parameter_audit.json", audit)
    start_step = 0
    if resume := config.get("resume_from_checkpoint"):
        accelerator.load_state(resume)
        state = json.loads((Path(resume) / "state.json").read_text()); start_step = int(state["step"])
        train_loader = accelerator.skip_first_batches(train_loader, start_step * int(config.get("gradient_accumulation_steps", 1)))
    best = float("inf"); metrics_path = output / "metrics.jsonl"
    model.train()
    progress = tqdm(total=max_updates, initial=start_step, desc="training", unit="update", disable=not accelerator.is_local_main_process)
    # Keep training aggregates on-device. Calling float()/int() on CUDA tensors
    # in every iteration serializes the CPU and GPU; scalars are copied only
    # when a log or validation record is actually emitted.
    interval_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    interval_examples = torch.zeros((), device=accelerator.device, dtype=torch.int64)
    interval_component_sums = {
        name: torch.zeros((), device=accelerator.device, dtype=torch.float64)
        for name in (("answer_loss", "padding_loss") if answer_padding_weights is not None else ())
    }
    update_step = start_step
    for microstep, batch in enumerate(train_loader, start=start_step * grad_accumulation + 1):
        step = microstep
        if step > max_steps: break
        with accelerator.accumulate(model):
            use_t_weighting = config["corruption_mode"] == "mask_only" and config.get("structured_loss_behavior", "all_answer_tokens") != "all_tokens"
            normalization_mask = batch["answer_mask"] | batch["padding_mask"] if bool(config.get("eos_padding_loss", False)) else batch["answer_mask"]
            sparse_positions = config.get("structured_loss_behavior", "all_answer_tokens") != "all_tokens"
            use_selected_logits = (
                sparse_positions
                and accelerator.num_processes == 1
                and bool(config.get("selected_logit_optimization", False))
                and not fp8_resolution["active"]
            )
            if use_selected_logits:
                # Calling the transformer backbone directly bypasses
                # Accelerate's model.forward wrapper, so reproduce its autocast
                # context and FP32 output conversion explicitly.
                with accelerator.autocast():
                    selected_logits, example_ids, token_ids = forward_bidirectional_selected(
                        model, batch["input_ids"], batch["padding_mask"], batch["loss_mask"]
                    )
                selected_logits = selected_logits.float()
                loss, info = selected_denoising_loss(
                    selected_logits,
                    batch["labels"][example_ids, token_ids],
                    example_ids,
                    batch["loss_mask"].sum(dim=1),
                    batch["sampled_t"] if use_t_weighting else None,
                    normalization_mask,
                    compute_unweighted_metric=False,
                    token_weights=(batch["token_loss_weights"][example_ids, token_ids]
                                   if use_t_weighting and "token_loss_weights" in batch else None),
                    answer_padding_weights=answer_padding_weights,
                    selected_answer_mask=(batch["answer_mask"] & ~batch["padding_mask"])[example_ids, token_ids]
                    if answer_padding_weights is not None else None,
                    selected_padding_mask=batch["padding_mask"][example_ids, token_ids]
                    if answer_padding_weights is not None else None,
                    answer_lengths=(batch["answer_mask"] & ~batch["padding_mask"]).sum(dim=1)
                    if answer_padding_weights is not None else None,
                    padding_lengths=batch["padding_mask"].sum(dim=1)
                    if answer_padding_weights is not None else None,
                )
            else:
                logits = forward_bidirectional(model, batch["input_ids"], batch["padding_mask"])
                loss, info = masked_denoising_loss(
                    logits,
                    batch["labels"],
                    batch["loss_mask"],
                    batch["sampled_t"] if use_t_weighting else None,
                    normalization_mask,
                    compute_unweighted_metric=False,
                    sparse_positions=sparse_positions,
                    token_weights=batch.get("token_loss_weights") if use_t_weighting else None,
                    answer_padding_weights=answer_padding_weights,
                    answer_mask=batch["answer_mask"], padding_mask=batch["padding_mask"],
                )
            accelerator.backward(loss)
            # Clip only after all gradient-accumulation microbatches have
            # contributed, matching Trainer's max_grad_norm behavior.
            if accelerator.sync_gradients and max_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        interval_loss_sum += loss.detach().to(torch.float64) * info["valid_examples"]
        interval_examples += info["valid_examples"]
        for name, total in interval_component_sums.items():
            total.add_(info[name].to(torch.float64) * info["valid_examples"])
        if accelerator.sync_gradients:
            update_step += 1
            progress.update(1)
        if not accelerator.sync_gradients:
            continue
        if accelerator.is_main_process and step % int(config.get("logging_steps", 10)) == 0:
            loss_value = loss.detach().item()
            supervised_tokens = info["supervised_tokens"].item()
            train_avg = (interval_loss_sum / interval_examples.clamp_min(1)).item()
            progress.set_postfix(train_loss=f"{loss_value:.4f}", train_avg=f"{train_avg:.4f}")
            _append_jsonl(metrics_path, {"split": "train", "step": step, "weighted_loss": loss_value,
                                        "supervised_tokens": supervised_tokens,
                                        **{name: info[name].item() for name in interval_component_sums}})
        if update_step % int(config.get("validation_steps", 100)) == 0 or update_step == max_updates:
            metrics = evaluate(model, val_loader, accelerator, config["corruption_mode"], config.get("structured_loss_behavior") == "all_tokens", bool(config.get("eos_padding_loss", False)), answer_padding_weights)
            if accelerator.is_main_process:
                generation_due = generation_interval is not None and (update_step % generation_interval == 0 or update_step == max_updates)
                if generation_due:
                    unwrapped = accelerator.unwrap_model(model)
                    metrics.update(generation_validation(unwrapped, tokenizer, train_collator.mask_info["mask_token_id"], config, initial_norms, accelerator.device, output, update_step))
                metrics.update({"split": "validation", "step": update_step}); _append_jsonl(metrics_path, metrics)
                generation_note = "".join(
                    f" | {label}={metrics[key]:.4f}"
                    for key, label in (
                        ("generation_median_perplexity", "generation_median_ppl"),
                        ("generation_perplexity", "generation_pooled_ppl"),
                        ("generation_mean_distinct_1", "generation_distinct_1"),
                    )
                    if metrics.get(key) is not None
                )
                train_avg = (interval_loss_sum / interval_examples.clamp_min(1)).item()
                interval_example_count = interval_examples.item()
                progress.write(f"step {update_step}/{max_updates} | train_loss_avg={train_avg:.4f} | validation_loss={metrics['weighted_loss']:.4f}{generation_note}")
                if accelerator.is_main_process:
                    _append_jsonl(metrics_path, {"split": "train_interval", "step": update_step,
                                                "weighted_loss": train_avg, "examples": interval_example_count,
                                                **{name: (total / interval_examples.clamp_min(1)).item()
                                                   for name, total in interval_component_sums.items()}})
                interval_loss_sum.zero_()
                interval_examples.zero_()
                for total in interval_component_sums.values():
                    total.zero_()
                if metrics["weighted_loss"] < best:
                    best = metrics["weighted_loss"]; unwrapped = accelerator.unwrap_model(model); _save_adapter(unwrapped, tokenizer, output / "best", initial_norms)
            accelerator.wait_for_everyone()
            model.train()
        if checkpoint_mode in {"every_checkpoint", "every_model"} and (update_step % int(config.get("checkpoint_steps", 500)) == 0 or update_step == max_updates):
            checkpoint = output / f"checkpoint-{update_step}"
            if checkpoint_mode == "every_checkpoint":
                accelerator.save_state(checkpoint, safe_serialization=True, save_embedding_layers=False)
                if accelerator.is_main_process:
                    _write_json(checkpoint / "state.json", {"step": update_step, "best_validation_loss": best})
            elif accelerator.is_main_process:
                # Inference-ready snapshot without optimizer/scheduler/RNG
                # state; it can also warm-start through resume_from_adapter.
                _save_adapter(accelerator.unwrap_model(model), tokenizer, checkpoint, initial_norms)
    progress.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and checkpoint_mode == "every_checkpoint":
        unwrapped = accelerator.unwrap_model(model); _save_adapter(unwrapped, tokenizer, output / "final", initial_norms)
    elif accelerator.is_main_process and checkpoint_mode == "every_model":
        unwrapped = accelerator.unwrap_model(model); _save_adapter(unwrapped, tokenizer, output / "final", initial_norms)
    # Test is deliberately after best-model selection/finalization.
    test_metrics = evaluate(model, test_loader, accelerator, config["corruption_mode"], config.get("structured_loss_behavior") == "all_tokens", bool(config.get("eos_padding_loss", False)), answer_padding_weights)
    if accelerator.is_main_process: _write_json(output / "test_metrics.json", test_metrics)
    accelerator.end_training()
    return test_metrics
