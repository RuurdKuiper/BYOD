"""Accelerate training/evaluation with reproducible denoising validation."""
from __future__ import annotations
import json
import atexit
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

from .data import DenoisingCollator
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
    if path.name == "metrics.jsonl" and os.getenv("BYOD_NOTEBOOK") == "1":
        print("__BYOD_METRIC__" + json.dumps(value, default=float), flush=True)


def _save_adapter(model, tokenizer, path: Path, initial_norms: dict[str, torch.Tensor] | None = None) -> None:
    """Save LoRA plus independently-unfrozen norm parameters for inference."""
    model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
    norm_state = {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if "norm" in name.lower()}
    torch.save(norm_state, path / "normalization_state.pt")
    if initial_norms is not None:
        torch.save(initial_norms, path / "normalization_initial_state.pt")
    tokenizer.save_pretrained(path)


def _worker_count(requested: int) -> int:
    """Respect the CPUs assigned to this runtime, including Colab affinity limits."""
    if requested < 0:
        raise ValueError("num_workers must be non-negative")
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError, NotImplementedError):
        available = os.cpu_count() or 1
    return min(requested, max(1, available))


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
        "temperature": 0.7,
        "confidence_eos_eot_inf": True,
        "system_prompt": "",
        "seed": 1234,
    }
    settings.update(config.get("generation_perplexity", {}))
    if settings.get("block_length") is None:
        settings["block_length"] = settings["max_new_tokens"]
    length, block, steps = (int(settings[key]) for key in ("max_new_tokens", "block_length", "num_steps"))
    if min(length, block, steps) < 1 or length % block or steps % (length // block):
        raise ValueError("Generation requires positive budgets, max_new_tokens divisible by block_length, "
                         "and num_steps divisible by the number of blocks")
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


def load_training_data(config: dict[str, Any], train_sample_limit: int | None, accelerator=None):
    """Download matching clean tokens once and reuse the local/Drive cache."""
    from contextlib import nullcontext
    from datasets import load_from_disk
    from .prepared_cache import (
        preparation_metadata, preparation_key, load_published_prepared_data,
        restore_prepared_cache, persist_prepared_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer_name_or_path", config["model_name_or_path"]),
        revision=config.get("model_revision"), use_fast=True,
        token=os.getenv("HF_TOKEN"), cache_dir=config.get("base_model_cache_dir", "base_models"),
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(int(config.get("seed", 42)))
    key = preparation_key(preparation_metadata(config, tokenizer, train_sample_limit))
    local = Path(config.get("prepared_data_cache_dir", "data/prepared")) / key
    persistent_dir = config.get("prepared_data_persistent_cache_dir")
    persistent = Path(persistent_dir) / key if persistent_dir else None
    main_process_first = accelerator.main_process_first if accelerator is not None else nullcontext
    with main_process_first():
        if restore_prepared_cache(local, persistent):
            print(f"Reusing tokenized data: {local}", flush=True)
            data = {split: load_from_disk(str(local / split)) for split in ("train", "validation", "test")}
        else:
            data = load_published_prepared_data(config, tokenizer, train_sample_limit)
            if data is None:
                raise ValueError(
                    "No compatible published token profile for this recipe. "
                    "Use a shipped preset within its 400,000-example training budget, "
                    "or configure a matching published dataset and revision."
                )
            for split, dataset in data.items():
                dataset.save_to_disk(str(local / split))
        persist_prepared_cache(local, persistent)
    return tokenizer, data["train"], data["validation"], data["test"]


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
    precision = str(config.get("precision", "bf16")).lower()
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError("precision must be fp16, bf16, or fp32")
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        mixed_precision="no" if precision == "fp32" else precision,
        dataloader_config=DataLoaderConfiguration(non_blocking=torch.cuda.is_available()),
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
    if generation_interval is not None:
        _generation_inference_settings(config)  # Fail before model loading, rather than at the first diagnostic.
    tokenizer, train_data, val_data, test_data = load_training_data(
        config, train_sample_limit, accelerator,
    )
    seed = int(config.get("seed", 42))
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
    config["num_workers"] = _worker_count(int(config.get("num_workers", 0)))
    prefetch_factor = int(config.get("prefetch_factor", 4))
    train_loader = _loader(train_data.shuffle(seed=seed), train_collator, int(config["batch_size"]), True, seed, int(config.get("num_workers", 0)), prefetch_factor)
    val_loader = _loader(val_data, eval_collator, int(config.get("eval_batch_size", config["batch_size"])), False, seed, int(config.get("num_workers", 0)), prefetch_factor)
    test_loader = _loader(test_data, eval_collator, int(config.get("eval_batch_size", config["batch_size"])), False, seed, int(config.get("num_workers", 0)), prefetch_factor)
    model, audit = load_denoising_model(config)
    initial_norms = _normalization_state(model)
    resolved_learning_rate, effective_batch_size, learning_rate_scale = _resolve_learning_rate(config, accelerator.num_processes)
    resolved = dict(config); resolved["eos_padding_loss"] = train_collator.eos_padding_loss; resolved["frontier_padding_mode"] = train_collator.frontier_padding_mode; resolved["training_samples_used"] = len(train_data); resolved["training_sample_limit"] = train_sample_limit; resolved["validation_samples_used"] = len(val_data); resolved["effective_batch_size"] = effective_batch_size; resolved["learning_rate_scale"] = learning_rate_scale; resolved["resolved_learning_rate"] = resolved_learning_rate; resolved["resolved_training_precision"] = precision
    _write_json(output / "resolved_config.json", resolved); _write_json(output / "parameter_audit.json", audit); _write_json(output / "mask_token.json", train_collator.mask_info)
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
    try:
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
            if accelerator.is_main_process and update_step % int(config.get("logging_steps", 10)) == 0:
                loss_value = loss.detach().item()
                supervised_tokens = info["supervised_tokens"].item()
                train_avg = (interval_loss_sum / interval_examples.clamp_min(1)).item()
                progress.set_postfix(train_loss=f"{loss_value:.4f}", train_avg=f"{train_avg:.4f}")
                _append_jsonl(metrics_path, {"split": "train", "step": update_step, "weighted_loss": loss_value,
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
    except KeyboardInterrupt:
        if accelerator.is_main_process and update_step > start_step:
            interrupted = output / "interrupted"
            _save_adapter(accelerator.unwrap_model(model), tokenizer, interrupted, initial_norms)
            _write_json(interrupted / "state.json", {"step": update_step})
            print(f"\nInterrupted at update {update_step}; adapter saved to {interrupted}", flush=True)
        raise
    finally:
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
