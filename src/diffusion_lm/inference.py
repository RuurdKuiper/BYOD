"""Interactive iterative denoising inference for saved LoRA adapters."""
from __future__ import annotations

import gc
from html import escape
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .data import validate_mask_token
from .legacy_compat import install_legacy_pickle_modules, patch_legacy_lora_modules, restore_legacy_pickle_modules
from .modeling import forward_bidirectional


def find_adapters(outputs_dir: str | Path = "outputs") -> list[str]:
    """Return adapter directories relative to outputs_dir, newest first."""
    root = Path(outputs_dir).resolve()
    if not root.exists():
        return []
    paths = [path for path in root.rglob("adapter_config.json") if path.parent.is_dir()]
    return [str(path.parent.relative_to(root)) for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)]


def _safe_adapter_path(outputs_dir: str | Path, selection: str) -> Path:
    """Resolve a selected adapter while preventing paths outside outputs_dir."""
    root = Path(outputs_dir).resolve()
    path = (root / selection).resolve()
    if root not in path.parents or not (path / "adapter_config.json").is_file():
        raise ValueError("Select a valid adapter directory below outputs/.")
    return path


def _precision_dtype(precision: str, device: torch.device) -> torch.dtype:
    """Map configured precision to a safe dtype for the selected device."""
    requested = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(precision, torch.float32)
    # CPU inference with low-precision weights is not generally supported; MPS
    # has better float32 compatibility for interactive single-request inference.
    if device.type == "cpu":
        return torch.float32
    # T4-class CUDA GPUs have no native BF16 Tensor Core support.  BF16
    # quantized compute there is substantially slower than FP16, so retain the
    # saved run's preference only where the hardware can execute it natively.
    if device.type == "cuda" and requested == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return torch.float16
    return requested


def select_device(requested: str = "auto") -> torch.device:
    """Choose an available CUDA, MPS, or CPU device from a UI selection."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    return torch.device(requested)


@dataclass
class InferenceSession:
    model: torch.nn.Module
    tokenizer: Any
    device: torch.device
    adapter_path: Path
    config: dict[str, Any]
    mask_token_id: int
    quantization: str = "none"
    compute_dtype: str = "unknown"
    legacy_wrapper: bool = False
    prompt_format: str = "chat_template"
    llada: bool = False


def _load_adapter_path(
    adapter_path: str | Path,
    device_name: str = "auto",
    quantization: str | None = None,
    preflight: bool = True,
) -> InferenceSession:
    """Load an adapter directory that has already been resolved and validated."""
    adapter_path = Path(adapter_path).expanduser().resolve()
    if not (adapter_path / "adapter_config.json").is_file():
        raise ValueError(f"Adapter directory has no adapter_config.json: {adapter_path}")
    config_candidates = (
        adapter_path / "resolved_config.json",
        adapter_path / "lad_run_config.json",
        adapter_path.parent / "resolved_config.json",
    )
    run_config_path = next((path for path in config_candidates if path.is_file()), None)
    run_config = json.loads(run_config_path.read_text()) if run_config_path else {}
    adapter_config = json.loads((adapter_path / "adapter_config.json").read_text())
    base_model = adapter_config["base_model_name_or_path"]
    device = select_device(device_name)
    dtype = _precision_dtype(run_config.get("precision", "fp32"), device)
    requested_quantization = str(quantization or "auto").lower()
    resolved_quantization = str(run_config.get("quantization", "none") if requested_quantization == "auto" else requested_quantization).lower()
    if resolved_quantization in {"4-bit", "qlora"}:
        resolved_quantization = "4bit"
    if resolved_quantization not in {"none", "off", "false", "4bit"}:
        raise ValueError("Inference quantization must be 'auto', 'none', or '4bit'.")
    use_4bit = resolved_quantization == "4bit"
    compute_dtype = dtype
    cache_dir = run_config.get("base_model_cache_dir", "base_models")
    token = os.getenv("HF_TOKEN")
    tokenizer_name = run_config.get("tokenizer_name_or_path", base_model)
    tokenizer_kwargs: dict[str, Any] = {}
    if "mistral" in str(tokenizer_name).lower():
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        token=token,
        cache_dir=cache_dir,
        clean_up_tokenization_spaces=False,
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs: dict[str, Any] = dict(torch_dtype=dtype, token=token, cache_dir=cache_dir, trust_remote_code=False)
    if use_4bit:
        if device.type != "cuda":
            raise RuntimeError("4-bit inference requires an NVIDIA CUDA device.")
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise ImportError("4-bit inference requires bitsandbytes; install with `pip install -e '.[cuda]'`.") from exc
        compute_dtype = _precision_dtype(run_config.get("compute_dtype", run_config.get("precision", "bf16")), device)
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(run_config.get("quantization_type", "nf4")),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(run_config.get("double_quant", True)),
        )
        # Quantized modules cannot subsequently be moved with model.to().
        load_kwargs["device_map"] = {"": device.index if device.index is not None else 0}
    base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    base.config.use_cache = False
    base.config.is_causal = False
    if hasattr(base.config, "use_bidirectional_attention"):
        base.config.use_bidirectional_attention = True
    adapter_load_kwargs: dict[str, Any] = {}
    if not use_4bit:
        # Stage adapter tensors on CPU before moving the assembled root module.
        # This is required by ZeroGPU, where CUDA is represented by a proxy at
        # module-import time but no physical GPU has been allocated yet.
        adapter_load_kwargs["device_map"] = {"": "cpu"}
        adapter_load_kwargs["torch_device"] = "cpu"
    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=False,
        **adapter_load_kwargs,
    )
    norm_path = adapter_path / "normalization_state.pt"
    if norm_path.is_file():
        model.load_state_dict(torch.load(norm_path, map_location="cpu", weights_only=True), strict=False)
    if not use_4bit:
        model.to(device)
    model.eval()
    mask_info = validate_mask_token(tokenizer, str(run_config.get("mask_token", "MASK")))
    session = InferenceSession(model, tokenizer, device, adapter_path, run_config, mask_info["mask_token_id"], "4bit" if use_4bit else "none", str(compute_dtype).removeprefix("torch."))
    if preflight:
        preflight_session(session)
    return session


def load_session(adapter_selection: str, outputs_dir: str | Path = "outputs", device_name: str = "auto", quantization: str | None = None) -> InferenceSession:
    """Load a base model, saved LoRA adapter, tokenizer, and norm state."""
    adapter_path = _safe_adapter_path(outputs_dir, adapter_selection)
    return _load_adapter_path(adapter_path, device_name, quantization)


def load_hub_adapter_session(
    repo_id: str,
    device_name: str = "auto",
    quantization: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    preflight: bool = True,
) -> InferenceSession:
    """Download and load a BYOD adapter from the Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Hub inference requires huggingface_hub; install it with `pip install huggingface-hub`."
        ) from exc
    adapter_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        token=os.getenv("HF_TOKEN"),
    )
    return _load_adapter_path(adapter_path, device_name, quantization, preflight=preflight)


def load_merged_session(
    model_path: str | Path,
    device_name: str = "auto",
    quantization: str | None = None,
    source_config: dict[str, Any] | None = None,
) -> InferenceSession:
    """Load a standalone model produced by ``merge_adapter.py``."""
    model_path = Path(model_path).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise ValueError(f"Merged model directory has no config.json: {model_path}")

    run_config = dict(source_config or {})
    saved_run_config = model_path / "lad_run_config.json"
    if not run_config and saved_run_config.is_file():
        run_config = json.loads(saved_run_config.read_text())

    device = select_device(device_name)
    dtype = _precision_dtype(run_config.get("precision", "bf16"), device)
    requested_quantization = str(quantization or "none").lower()
    if requested_quantization == "auto":
        requested_quantization = "none"
    if requested_quantization in {"4-bit", "qlora"}:
        requested_quantization = "4bit"
    if requested_quantization not in {"none", "off", "false", "4bit"}:
        raise ValueError("Merged-model quantization must be 'none' or '4bit'.")

    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        token=token,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = requested_quantization == "4bit"
    compute_dtype = dtype
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "token": token,
        "trust_remote_code": False,
    }
    if use_4bit:
        if device.type != "cuda":
            raise RuntimeError("4-bit merged-model inference requires an NVIDIA CUDA device.")
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise ImportError("4-bit inference requires bitsandbytes; install with `pip install -e '.[cuda]'`.") from exc
        compute_dtype = _precision_dtype(run_config.get("compute_dtype", run_config.get("precision", "bf16")), device)
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(run_config.get("quantization_type", "nf4")),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(run_config.get("double_quant", True)),
        )
        load_kwargs["device_map"] = {"": device.index if device.index is not None else 0}

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.config.use_cache = False
    model.config.is_causal = False
    if hasattr(model.config, "use_bidirectional_attention"):
        model.config.use_bidirectional_attention = True
    if not use_4bit:
        model.to(device)
    model.eval()

    mask_info = validate_mask_token(tokenizer, str(run_config.get("mask_token", "MASK")))
    session = InferenceSession(
        model=model,
        tokenizer=tokenizer,
        device=device,
        adapter_path=model_path,
        config=run_config,
        mask_token_id=mask_info["mask_token_id"],
        quantization="4bit" if use_4bit else "none",
        compute_dtype=str(compute_dtype).removeprefix("torch."),
    )
    preflight_session(session)
    return session


def _load_legacy_checkpoint_session(checkpoint: str | Path, tokenizer_name_or_path: str, device_name: str = "auto", source_config: dict[str, Any] | None = None) -> InferenceSession:
    """Load one trusted legacy full-object checkpoint from a local path."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Legacy checkpoint does not exist: {checkpoint}")
    if not tokenizer_name_or_path.strip():
        raise ValueError("Legacy loading requires a tokenizer name or local tokenizer path.")
    # A full-object checkpoint can execute pickle code.  This loader is for
    # checkpoints the user trusts, including their locally archived model.
    previous_modules = install_legacy_pickle_modules()
    try:
        model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    finally:
        restore_legacy_pickle_modules(previous_modules)
    if not isinstance(model, torch.nn.Module):
        raise ValueError(f"{checkpoint} is not a full torch.nn.Module checkpoint.")
    patch_legacy_lora_modules(model)

    token = os.getenv("HF_TOKEN")
    device = select_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path.strip(), use_fast=True, token=token, clean_up_tokenization_spaces=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device).eval()
    mask_info = validate_mask_token(tokenizer)
    session = InferenceSession(
        model=model,
        tokenizer=tokenizer,
        device=device,
        adapter_path=checkpoint,
        config=source_config or {"model_source": "local_legacy", "checkpoint": str(checkpoint), "tokenizer_name_or_path": tokenizer_name_or_path.strip()},
        mask_token_id=mask_info["mask_token_id"],
        quantization="none",
        compute_dtype=str(next((parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()), torch.float32)).removeprefix("torch."),
        legacy_wrapper=True,
        prompt_format="legacy_llama",
    )
    preflight_session(session)
    return session


def load_local_legacy_session(checkpoint_path: str | Path, tokenizer_name_or_path: str, device_name: str = "auto") -> InferenceSession:
    """Load and preflight a trusted local legacy full-model checkpoint."""
    return _load_legacy_checkpoint_session(checkpoint_path, tokenizer_name_or_path, device_name)


def load_hosted_legacy_session(repo_id: str, filename: str, tokenizer_name_or_path: str, device_name: str = "auto") -> InferenceSession:
    """Load the trusted legacy full-model checkpoint hosted on Hugging Face.

    This exists for controlled comparisons: the checkpoint uses its original
    wrapper to construct full bidirectional attention, while decoding uses the
    current project's prompt and denoising loop.  Pickled checkpoints are only
    safe to load from a repository you trust.
    """
    if not repo_id.strip() or not filename.strip() or not tokenizer_name_or_path.strip():
        raise ValueError("Hosted legacy loading requires a repository ID, checkpoint filename, and tokenizer name.")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Hosted model loading requires huggingface_hub, installed with transformers.") from exc

    token = os.getenv("HF_TOKEN")
    checkpoint = hf_hub_download(repo_id=repo_id.strip(), filename=filename.strip(), token=token)
    return _load_legacy_checkpoint_session(
        checkpoint,
        tokenizer_name_or_path,
        device_name,
        {"model_source": "huggingface_legacy", "repo_id": repo_id.strip(), "filename": filename.strip(), "tokenizer_name_or_path": tokenizer_name_or_path.strip()},
    )


def load_llada_session(repo_id: str = "GSAI-ML/LLaDA-8B-Instruct", device_name: str = "auto") -> InferenceSession:
    """Load LLaDA Instruct as a mask predictor for this app's denoising loop."""
    if not repo_id.strip():
        raise ValueError("LLaDA loading requires a Hugging Face repository ID.")
    device = select_device(device_name)
    if device.type not in {"cuda", "mps"}:
        raise ValueError("LLaDA-8B-Instruct requires CUDA or MPS inference; select a GPU-capable runtime.")
    # Apple MPS does not reliably support BF16 inference for this remote model.
    # FP16 is the practical MPS format; CUDA retains BF16 where available.
    dtype = torch.float16 if device.type == "mps" else _precision_dtype("bf16", device)
    token = os.getenv("HF_TOKEN")
    cache_dir = "base_models"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id.strip(), trust_remote_code=True, token=token, cache_dir=cache_dir,
        )
        model = AutoModel.from_pretrained(
            repo_id.strip(), trust_remote_code=True, torch_dtype=dtype, token=token, cache_dir=cache_dir,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load LLaDA. Its official implementation requires the remote model code "
            "and is tested with transformers==4.38.2."
        ) from exc
    if tokenizer.pad_token_id == 126336:
        raise ValueError("LLaDA's pad token must differ from its fixed mask token (126336).")
    tokenizer.padding_side = "left"
    model.to(device).eval()
    session = InferenceSession(
        model=model,
        tokenizer=tokenizer,
        device=device,
        adapter_path=Path(repo_id.strip()),
        config={"model_source": "huggingface_llada", "repo_id": repo_id.strip()},
        mask_token_id=126336,
        quantization="none",
        compute_dtype=str(dtype).removeprefix("torch."),
        prompt_format="llada",
        llada=True,
    )
    preflight_session(session)
    return session


def _sample(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k sample token IDs and return their normalized sampling confidence."""
    logits = logits / max(temperature, 1e-5)
    vocab_size = logits.shape[-1]
    k = min(max(int(top_k), 1), vocab_size)
    values, indices = torch.topk(logits, k, dim=-1)
    probabilities = F.softmax(values, dim=-1)
    picked_local = torch.multinomial(probabilities, 1, generator=generator)
    picked = indices.gather(-1, picked_local).squeeze(-1)
    confidence = probabilities.gather(-1, picked_local).squeeze(-1)
    return picked, confidence


def _llada_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply the float64 Gumbel-max transform used by official LLaDA decoding."""
    if float(temperature) == 0.0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / (-torch.log(noise)).pow(float(temperature))


def _apply_repetition_penalty(
    logits: torch.Tensor,
    answer_ids: torch.Tensor,
    penalty: float,
    mask_token_id: int,
    *,
    exclude_self: bool = False,
    excluded_token_ids: set[int] | None = None,
) -> torch.Tensor:
    """Reduce repeated-token probability weights by ``penalty ** count``.

    Only tokens already present in the generated answer are penalized; prompt
    tokens, MASK, and configured special tokens never contribute. Subtracting
    ``count * log(penalty)`` from a logit divides its unnormalized softmax
    probability by ``penalty ** count``. For the revisable denoise stream, a
    position's current token is excluded from its count, avoiding needless
    churn of unique predictions.
    """
    penalty = float(penalty)
    if penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.0")
    if penalty == 1.0:
        return logits
    if logits.ndim != 3 or answer_ids.ndim != 2 or logits.shape[:2] != answer_ids.shape:
        raise ValueError("repetition penalty expects logits [batch, length, vocab] matching answer IDs")

    adjusted = logits.clone()
    vocabulary_size = adjusted.shape[-1]
    for batch_index in range(answer_ids.shape[0]):
        valid = answer_ids[batch_index]
        valid_mask = (
            (valid != int(mask_token_id))
            & (valid >= 0)
            & (valid < vocabulary_size)
        )
        excluded = set(excluded_token_ids or ())
        excluded.add(int(mask_token_id))
        if excluded:
            excluded_tensor = torch.tensor(
                sorted(excluded), device=valid.device, dtype=valid.dtype
            )
            valid_mask &= ~torch.isin(valid, excluded_tensor)
        valid = valid[valid_mask]
        if not len(valid):
            continue
        token_ids, counts = torch.unique(valid, return_counts=True)
        current = answer_ids[batch_index]
        scores = adjusted[batch_index, :, token_ids]
        exponents = counts[None, :].expand(logits.shape[1], -1)
        if exclude_self:
            exponents = exponents - (current[:, None] == token_ids[None, :]).to(
                dtype=exponents.dtype
            )
        log_penalty = torch.log(
            torch.tensor(penalty, device=logits.device, dtype=torch.float32)
        )
        penalized = scores.float() - exponents.float() * log_penalty
        # Penalty arithmetic stays in FP32 for numerical stability, then returns
        # to the model's native BF16/FP16 dtype for indexed assignment.
        adjusted[batch_index, :, token_ids] = penalized.to(dtype=adjusted.dtype)
    return adjusted


def _apply_eos_eot_prediction_penalty(
    logits: torch.Tensor,
    penalty: float,
    eos_token_id: int,
    eot_token_id: int | None = None,
) -> torch.Tensor:
    """Reduce EOS/EoT sampling weights without changing retention confidence.

    Subtracting ``log(penalty)`` from the selected logits divides their
    unnormalized probability weight by ``penalty``.  This is deliberately
    independent of the LLaDA-style delayed-retention option, which changes
    which sampled positions are retained or re-masked rather than what token
    is sampled in the first place.
    """
    penalty = float(penalty)
    if penalty < 1.0:
        raise ValueError("EOS/EOT prediction penalty must be at least 1.0")
    if penalty == 1.0:
        return logits

    token_ids = {int(eos_token_id)}
    if eot_token_id is not None:
        token_ids.add(int(eot_token_id))
    token_ids = {token_id for token_id in token_ids if 0 <= token_id < logits.shape[-1]}
    if not token_ids:
        return logits

    adjusted = logits.clone()
    log_penalty = torch.log(
        torch.tensor(penalty, device=logits.device, dtype=torch.float32)
    )
    indices = torch.tensor(sorted(token_ids), device=logits.device, dtype=torch.long)
    adjusted[..., indices] = (
        adjusted[..., indices].float() - log_penalty
    ).to(dtype=adjusted.dtype)
    return adjusted


def _llada_transfer_schedule(mask_count: int, steps: int) -> list[int]:
    """Distribute a linear-noise transfer budget uniformly across steps."""
    if mask_count < 0 or steps < 1:
        raise ValueError("mask_count must be non-negative and steps must be positive")
    base, remainder = divmod(mask_count, steps)
    return [base + int(index < remainder) for index in range(steps)]


def _block_step_plan(
    generation_length: int,
    steps: int,
    block_length: int | None,
) -> list[tuple[int, int, int, int, int]]:
    """Allocate a fixed total step budget across sequential answer blocks."""
    generation_length, steps = int(generation_length), int(steps)
    requested_block_length = generation_length if block_length is None else int(block_length)
    if generation_length < 1 or steps < 1 or requested_block_length < 1:
        raise ValueError("generation length, steps, and block length must be positive")
    effective_block_length = min(requested_block_length, generation_length)
    num_blocks = (generation_length + effective_block_length - 1) // effective_block_length
    if steps < num_blocks:
        raise ValueError(
            f"Denoising steps ({steps}) must be at least the number of blocks ({num_blocks})"
        )
    base, remainder = divmod(steps, num_blocks)
    plan = []
    for block_index in range(num_blocks):
        block_start = block_index * effective_block_length
        block_end = min(generation_length, block_start + effective_block_length)
        block_steps = base + int(block_index < remainder)
        for block_step in range(block_steps):
            plan.append((block_index, block_start, block_end, block_step, block_steps))
    return plan


def _remask_offsets(confidence: torch.Tensor, mask_probability: float, confidence_guided: bool) -> torch.Tensor:
    """Choose answer offsets to re-mask, preferring uncertain tokens when guided."""
    probability = max(0.0, min(1.0, float(mask_probability)))
    if confidence_guided:
        count = round(probability * len(confidence))
        return torch.argsort(confidence)[:count]
    return torch.where(torch.rand(len(confidence), device=confidence.device) < probability)[0]


def _native_eot_token_id(tokenizer: Any) -> int | None:
    """Return a tokenizer's native end-of-turn ID when it has one."""
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    unknown = getattr(tokenizer, "unk_token_id", None)
    for token in ("<|eot_id|>", "<end_of_turn>", "<|end_of_turn|>"):
        token_id = convert(token)
        if token_id is not None and token_id != unknown and int(token_id) >= 0:
            return int(token_id)
    return None


def forward_denoising(session: InferenceSession, input_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Return denoising logits for either the current or legacy model wrapper."""
    if session.llada:
        # LLaDA caches rotary embeddings during preflight. Its remote model code
        # requires all later uses of those cached inference tensors to remain in
        # inference mode as well.
        with torch.inference_mode():
            outputs = session.model(input_ids, attention_mask=(~padding_mask).to(dtype=torch.long))
        return outputs.logits
    if session.legacy_wrapper:
        # The archived CustomTransformerModel builds its own full-attention
        # 4-D mask and passes use_cache=False to its inner Peft model. Passing
        # either argument here would duplicate the wrapper's keyword.
        outputs = session.model(input_ids=input_ids)
        return outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    return forward_bidirectional(session.model, input_ids, padding_mask)


@torch.inference_mode()
def preflight_session(session: InferenceSession) -> tuple[int, int]:
    """Run one real forward pass and fail early if a loaded model is unusable."""
    prefix = _prompt_ids(session.tokenizer, "Reply with OK.", "You are a helpful assistant.", session.prompt_format)
    input_ids = torch.tensor([prefix + [session.mask_token_id]], device=session.device, dtype=torch.long)
    padding = torch.zeros_like(input_ids, dtype=torch.bool)
    try:
        logits = forward_denoising(session, input_ids, padding)
    except Exception as exc:
        source = "LLaDA" if session.llada else "legacy hosted checkpoint" if session.legacy_wrapper else "saved adapter"
        raise RuntimeError(f"Inference preflight failed for {source}; the model was not loaded for generation: {exc}") from exc
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise RuntimeError(f"Inference preflight returned invalid logits shape {tuple(logits.shape)} for input shape {tuple(input_ids.shape)}")
    if not torch.isfinite(logits[:, -1]).all():
        raise RuntimeError("Inference preflight produced non-finite final-token logits.")
    return int(input_ids.shape[1]), int(logits.shape[-1])


def _prompt_ids(tokenizer: Any, question: str, system_prompt: str, prompt_format: str = "chat_template") -> list[int]:
    """Render system/user messages through a tokenizer’s native chat template."""
    from .data import apply_neutral_chat_template
    if not question.strip():
        raise ValueError("Enter a question or prompt.")
    if prompt_format == "legacy_llama":
        # The hosted historical checkpoint used a base Llama tokenizer with no
        # chat_template. Match the prompt layout from its original app while
        # still running the current project's denoising/sampling loop.
        prompt = (
            "<|begin_of_text|>\n"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{question.strip()}\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        return list(tokenizer.encode(prompt, add_special_tokens=False))
    if prompt_format == "llada":
        content = f"{system_prompt}\n\n{question.strip()}" if system_prompt.strip() else question.strip()
        rendered = apply_neutral_chat_template(
            tokenizer,
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(rendered, str):
            rendered = tokenizer.encode(rendered, add_special_tokens=False)
        elif hasattr(rendered, "input_ids"):
            rendered = rendered.input_ids
        if rendered and isinstance(rendered[0], list):
            rendered = rendered[0]
        return list(rendered)
    if prompt_format != "chat_template":
        raise ValueError(f"Unknown prompt format: {prompt_format}")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{tokenizer.name_or_path} has no chat template; inference needs one to identify the answer boundary.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    try:
        rendered = apply_neutral_chat_template(tokenizer, messages, tokenize=True, add_generation_prompt=True)
    except Exception as exc:
        # Gemma's official template rejects a separate system role; preserve
        # the prompt by folding it into the user message, as training does.
        if exc.__class__.__name__ != "TemplateError" or "System role not supported" not in str(exc):
            raise
        rendered = apply_neutral_chat_template(tokenizer, [
            {"role": "user", "content": f"{system_prompt}\n\n{question}"},
        ], tokenize=True, add_generation_prompt=True)
    if isinstance(rendered, str):
        rendered = tokenizer.encode(rendered, add_special_tokens=False)
    elif hasattr(rendered, "input_ids"):
        rendered = rendered.input_ids
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    if not all(isinstance(token, int) for token in rendered):
        raise ValueError(f"Tokenizer returned a non-integer chat-template encoding: {type(rendered).__name__}")
    return list(rendered)


@torch.inference_mode()
def llada_generate(
    session: InferenceSession,
    question: str,
    *,
    gen_length: int,
    steps: int,
    block_length: int | None = None,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    eot_token_id: int | None = None,
    system_prompt: str = "",
    seed: int = 1234,
    repetition_penalty: float = 1.0,
    eos_eot_prediction_penalty: float = 1.0,
) -> str:
    """Generate with the official LLaDA fixed-budget transfer algorithm.

    This intentionally bypasses ``denoise_stream``: official LLaDA predicts
    only still-masked positions, permanently transfers a fixed number per
    reverse step, and uses neither proportional unmasking nor a mask-ratio
    heuristic. The sampler is model-agnostic, so mask-only LAD adapters can use
    it with their native tokenizer and prompt format as well.
    """
    gen_length, steps = int(gen_length), int(steps)
    block_length = int(block_length or gen_length)
    if gen_length < 1 or steps < 1 or block_length < 1:
        raise ValueError("gen_length, steps, and block_length must be positive")
    if gen_length % block_length:
        raise ValueError("LLaDA gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks:
        raise ValueError("LLaDA steps must be divisible by the number of blocks")
    if remasking not in {"low_confidence", "random"}:
        raise ValueError("LLaDA remasking must be 'low_confidence' or 'random'")

    torch.manual_seed(int(seed))
    if session.device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    prefix = _prompt_ids(session.tokenizer, question, system_prompt, session.prompt_format)
    prompt_length = len(prefix)
    x = torch.full((1, prompt_length + gen_length), session.mask_token_id, dtype=torch.long, device=session.device)
    x[0, :prompt_length] = torch.tensor(prefix, dtype=torch.long, device=session.device)
    padding = torch.zeros_like(x, dtype=torch.bool)
    prompt_index = x != session.mask_token_id
    steps_per_block = steps // num_blocks
    eos_token_id = int(session.tokenizer.eos_token_id)

    for block in range(num_blocks):
        block_start = prompt_length + block * block_length
        block_end = block_start + block_length
        transfer_schedule = _llada_transfer_schedule(int((x[:, block_start:block_end] == session.mask_token_id).sum()), steps_per_block)
        for transfer_count in transfer_schedule:
            mask_index = x == session.mask_token_id
            if cfg_scale > 0.0:
                unconditional = x.clone()
                unconditional[prompt_index] = session.mask_token_id
                model_input = torch.cat([x, unconditional], dim=0)
                model_padding = torch.cat([padding, padding], dim=0)
                conditional_logits, unconditional_logits = forward_denoising(session, model_input, model_padding).chunk(2, dim=0)
                logits = unconditional_logits + (float(cfg_scale) + 1.0) * (conditional_logits - unconditional_logits)
            else:
                logits = forward_denoising(session, x, padding)
            logits[:, prompt_length:] = _apply_repetition_penalty(
                logits[:, prompt_length:],
                x[:, prompt_length:],
                repetition_penalty,
                session.mask_token_id,
                excluded_token_ids=set(getattr(session.tokenizer, "all_special_ids", [])),
            )
            logits[:, prompt_length:] = _apply_eos_eot_prediction_penalty(
                logits[:, prompt_length:],
                eos_eot_prediction_penalty,
                eos_token_id,
                eot_token_id,
            )
            if logits_eos_inf:
                logits = logits.clone()
                logits[..., eos_token_id] = -torch.inf
            predictions = torch.argmax(_llada_gumbel_noise(logits, temperature), dim=-1)
            if remasking == "low_confidence":
                probabilities = F.softmax(logits, dim=-1)
                confidence = probabilities.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
                if confidence_eos_eot_inf:
                    # Appendix B.4 delays EOS/EoT predictions by assigning
                    # them the lowest transfer confidence; they remain valid
                    # predictions and can still transfer in later steps.
                    special_prediction = predictions == eos_token_id
                    if eot_token_id is not None and 0 <= int(eot_token_id) < logits.shape[-1]:
                        special_prediction |= predictions == int(eot_token_id)
                    confidence = confidence.masked_fill(special_prediction, torch.finfo(confidence.dtype).min)
            else:
                confidence = torch.rand(predictions.shape, device=session.device)
            candidate = mask_index.clone()
            candidate[:, :block_start] = False
            candidate[:, block_end:] = False
            confidence = confidence.masked_fill(~candidate, -torch.inf)
            if transfer_count:
                transfer = torch.topk(confidence[0], k=int(transfer_count)).indices
                x[0, transfer] = predictions[0, transfer]

    answer = x[0, prompt_length:].tolist()
    return session.tokenizer.decode(answer, skip_special_tokens=True).strip()


def render_denoising_step(
    tokens: list[int],
    confidences: list[float],
    answer_start: int,
    tokenizer: Any,
    mask_token_id: int,
    step: int,
    total_steps: int,
    retained: set[int] | None = None,
    frozen: dict[int, int] | None = None,
    frozen_confidences: dict[int, float] | None = None,
    frozen_steps: dict[int, int] | None = None,
    color_mode: str = "Prediction probability",
) -> str:
    """Render one denoising state with optional token coloring."""
    eos_id = tokenizer.eos_token_id
    pieces = []
    answer = tokens[answer_start:]
    output_token_count = 0
    for offset, token in enumerate(answer):
        if token == eos_id:
            break
        output_token_count += 1
        token_text = escape(tokenizer.decode([token], skip_special_tokens=False)).replace("\n", "↵ ")
        if token == mask_token_id:
            style, token_text = (
                "display:inline-block;background:#d1d5db;color:#4b5563;"
                "border:1px solid #9ca3af;border-radius:4px;padding:0 4px;"
                "font-size:.78em;line-height:1.45;margin:0 1px;vertical-align:baseline",
                "mask",
            )
            title = f"token position {offset} · masked at iteration {step}"
        else:
            confidence = (
                frozen_confidences[offset]
                if frozen and offset in frozen and frozen_confidences and offset in frozen_confidences
                else float(confidences[offset]) if offset < len(confidences) else 0.0
            )
            confidence = max(0.0, min(1.0, confidence))
            predicted_step = (
                frozen_steps[offset]
                if frozen and offset in frozen and frozen_steps and offset in frozen_steps
                else step
            )
            if color_mode == "Prediction iteration":
                iteration_fraction = max(0.0, min(1.0, predicted_step / max(total_steps, 1)))
                lightness = 72 - round(42 * iteration_fraction)
                style = f"color:hsl(210,90%,{lightness}%);font-weight:500"
            elif color_mode == "Prediction probability":
                hue = int(confidence * 120)
                style = f"color:hsl({hue},90%,30%);font-weight:{'600' if confidence > .8 else '400'}"
            else:
                style = "color:inherit;font-weight:400"
            title = (
                f"token position {offset} · predicted at iteration {predicted_step} · "
                f"sampling probability {confidence:.1%}"
            )
        pieces.append(f"<span style='{style}' title='{title}'>{token_text}</span>")
    pct = int(100 * step / max(total_steps, 1))
    if color_mode == "Prediction iteration":
        legend = "Light-to-dark blue indicates earlier-to-later prediction iterations"
    elif color_mode == "Prediction probability":
        legend = "Red-to-green indicates lower-to-higher sampling probability"
    else:
        legend = "Token coloring is disabled"
    return (f"<div style='font-family:system-ui;padding:14px;border:1px solid #d1d5db;border-radius:9px;background:#fafafa'>"
            f"<div style='font-weight:700;color:#2563eb;margin-bottom:7px'>Denoising step {step}/{total_steps} · {output_token_count} output tokens</div>"
            f"<div style='background:#e5e7eb;border-radius:4px;height:7px;margin-bottom:10px'><div style='background:#2563eb;width:{pct}%;height:7px;border-radius:4px'></div></div>"
            f"<div style='line-height:2;font-size:15px;white-space:pre-wrap'>{''.join(pieces)}</div>"
            f"<div style='font-size:11px;color:#6b7280;margin-top:8px'>{legend}; gray chips are masks. Hover over a token for its position, prediction iteration, and probability.</div></div>")


def decode_denoising_state(
    tokens: list[int],
    tokenizer: Any,
    mask_token_id: int,
    *,
    show_eos_tokens: bool = False,
) -> str:
    """Decode one answer state with unresolved positions and optional EOS shown."""
    eos_id = tokenizer.eos_token_id
    eot_id = _native_eot_token_id(tokenizer) if show_eos_tokens else None
    visible_end_ids = {int(eos_id)}
    if eot_id is not None:
        visible_end_ids.add(int(eot_id))
    if not show_eos_tokens and eos_id in tokens:
        tokens = tokens[:tokens.index(eos_id)]

    # Decode contiguous resolved spans so subword spacing remains natural, but
    # make adjacent mask tokens unambiguous and independent of the configured
    # mask vocabulary item (some model configs use markers such as `<?>`).
    pieces: list[str] = []
    resolved: list[int] = []

    def flush_resolved() -> None:
        if resolved:
            text = tokenizer.decode(
                resolved,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if text:
                pieces.append(text)
            resolved.clear()

    for token in tokens:
        if token == mask_token_id:
            flush_resolved()
            pieces.append("MASK")
        elif show_eos_tokens and token in visible_end_ids:
            flush_resolved()
            marker = tokenizer.decode(
                [token],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).strip()
            fallback = "<EOS>" if token == eos_id else "<EOT>"
            pieces.append(marker or fallback)
        else:
            resolved.append(token)
    flush_resolved()
    return " ".join(pieces)


def denoise_stream(session: InferenceSession, question: str, system_prompt: str, max_new_tokens: int, num_steps: int, noise_level: float, temperature: float, top_k: int, seed: int, permanent_unmask: bool = False, confidence_guided: bool = False, proportional_unmask: bool = True, early_stopping: bool = False, confidence_eos_eot_inf: bool = False, freeze_retained_tokens: bool = True, repetition_penalty: float = 1.0, eos_eot_prediction_penalty: float = 1.0, include_pre_remask_prediction: bool = False, block_length: int | None = None, trajectory_color_mode: str = "Prediction probability"):
    """Yield denoising states with optionally retained positions and locked values."""
    prefix = _prompt_ids(session.tokenizer, question, system_prompt, session.prompt_format)
    max_new_tokens, num_steps = int(max_new_tokens), int(num_steps)
    if max_new_tokens < 1 or num_steps < 1:
        raise ValueError("max_new_tokens and num_steps must both be at least 1.")
    step_plan = _block_step_plan(max_new_tokens, num_steps, block_length)
    num_blocks = step_plan[-1][0] + 1
    ids = prefix + [session.mask_token_id] * max_new_tokens
    answer_start = len(prefix)
    # Use the device's default RNG so this works consistently on CUDA, MPS, and
    # CPU; seed it once per request for reproducible interactive runs.
    torch.manual_seed(int(seed))
    if session.device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    padding = torch.zeros((1, len(ids)), device=session.device, dtype=torch.bool)
    last_confidence = 0.0
    retained: set[int] = set()
    frozen: dict[int, int] = {}
    frozen_confidences: dict[int, float] = {}
    frozen_steps: dict[int, int] = {}
    last_predictions: list[tuple[int, ...]] = []
    eot_token_id = _native_eot_token_id(session.tokenizer) if confidence_eos_eot_inf or float(eos_eot_prediction_penalty) > 1.0 else None
    # EOS/EOT delaying and confidence-guided retention are independent. In
    # random mode, endings are still retained last, while ordinary tokens are
    # selected randomly.
    guided_retention = confidence_guided
    skip_block_index: int | None = None
    for step, (block_index, block_start, block_end, block_step, block_steps) in enumerate(step_plan):
        if block_index == skip_block_index:
            continue
        if block_step == 0:
            last_predictions.clear()
        tokens = torch.tensor([ids], device=session.device, dtype=torch.long)
        with torch.inference_mode():
            answer_ids = tokens[:, answer_start:]
            logits = forward_denoising(session, tokens, padding)[:, answer_start:]
            logits = _apply_repetition_penalty(
                logits,
                answer_ids,
                repetition_penalty,
                session.mask_token_id,
                exclude_self=True,
                excluded_token_ids=set(getattr(session.tokenizer, "all_special_ids", [])),
            )[0]
            logits = _apply_eos_eot_prediction_penalty(
                logits,
                eos_eot_prediction_penalty,
                session.tokenizer.eos_token_id,
                eot_token_id,
            )
            sampled, confidence = _sample(logits, float(temperature), int(top_k), None)
        retention_confidence = confidence
        special_prediction = torch.zeros_like(sampled, dtype=torch.bool)
        if confidence_eos_eot_inf:
            special_prediction = sampled == session.tokenizer.eos_token_id
            if eot_token_id is not None:
                special_prediction |= sampled == eot_token_id
            retention_confidence = confidence.masked_fill(
                special_prediction, torch.finfo(confidence.dtype).min
            )
        ids[answer_start + block_start : answer_start + block_end] = sampled[block_start:block_end].tolist()
        if freeze_retained_tokens:
            for offset, token in frozen.items():
                ids[answer_start + offset] = token
        predicted_text = decode_denoising_state(
            ids[answer_start:],
            session.tokenizer,
            session.mask_token_id,
            show_eos_tokens=include_pre_remask_prediction,
        )
        last_confidence = float(confidence[block_start:block_end].mean().cpu())
        # Compare the visible sampled answer before the next iteration's
        # re-masking. Tokens after the first EOS are not part of the answer and
        # must not prevent convergence. Excluding EOS itself still preserves
        # its position through the tuple length: moving EOS changes the prefix.
        prediction = ids[answer_start:]
        if session.tokenizer.eos_token_id in prediction:
            prediction = prediction[:prediction.index(session.tokenizer.eos_token_id)]
        last_predictions.append(tuple(prediction))
        if len(last_predictions) > 2:
            last_predictions.pop(0)
        stopped_early = early_stopping and len(last_predictions) == 2 and len(set(last_predictions)) == 1
        # Progressively reduce corruption. Re-mask independently, retaining the
        # legacy schedule's initial noise_level and ending with a clean sample.
        if block_step + 1 < block_steps and not stopped_early:
            block_size = block_end - block_start
            mask_probability = max(0.0, min(1.0, float(noise_level) * (1.0 - (block_step + 1) / block_steps)))
            if permanent_unmask:
                keep_count = min(block_size, max(0, round((1.0 - mask_probability) * block_size)))
                retained_in_block = sum(block_start <= i < block_end for i in retained)
                needed = keep_count - retained_in_block
                candidates = [i for i in range(block_start, block_end) if i not in retained]
                if needed > 0 and candidates:
                    def random_retention_order(pool: list[int]) -> list[int]:
                        order = torch.randperm(len(pool), device=session.device).tolist()
                        randomized = [pool[i] for i in order]
                        if not confidence_eos_eot_inf:
                            return randomized
                        ordinary = [i for i in randomized if not bool(special_prediction[i])]
                        endings = [i for i in randomized if bool(special_prediction[i])]
                        return ordinary + endings

                    if proportional_unmask:
                        eos_positions = [i for i in range(block_start, block_end) if ids[answer_start + i] == session.tokenizer.eos_token_id]
                        boundary = min(eos_positions) if eos_positions else block_end
                        pools = [[i for i in candidates if i < boundary], [i for i in candidates if i >= boundary]]
                        target_normal = round(keep_count * (boundary - block_start) / block_size)
                        target_counts = [
                            max(0, target_normal - sum(block_start <= i < boundary for i in retained)),
                            max(0, keep_count - target_normal - sum(boundary <= i < block_end for i in retained)),
                        ]
                        chosen = []
                        for pool, target in zip(pools, target_counts):
                            if not pool or target <= 0:
                                continue
                            if guided_retention:
                                order = torch.argsort(retention_confidence, descending=True).tolist()
                                chosen.extend([i for i in order if i in pool][:target])
                            else:
                                chosen.extend(random_retention_order(pool)[:target])
                        if len(chosen) < needed:
                            remainder = [i for i in candidates if i not in chosen]
                            chosen.extend(remainder[: needed - len(chosen)])
                    elif guided_retention:
                        confidence_order = torch.argsort(retention_confidence, descending=True).tolist()
                        chosen = [i for i in confidence_order if i in candidates][:needed]
                    else:
                        chosen = random_retention_order(candidates)[:needed]
                    for offset in chosen:
                        retained.add(offset)
                        if freeze_retained_tokens:
                            frozen[offset] = ids[answer_start + offset]
                            frozen_confidences[offset] = float(confidence[offset].cpu())
                            frozen_steps[offset] = step + 1
                for offset in range(block_start, block_end):
                    if offset not in retained:
                        ids[answer_start + offset] = session.mask_token_id
            else:
                # Confidence-guided refinement keeps every token revisable, but
                # preferentially re-masks the least certain predictions. The
                # unguided mode retains the original random re-masking policy.
                remask_offsets = _remask_offsets(
                    retention_confidence[block_start:block_end],
                    mask_probability,
                    guided_retention,
                )
                if confidence_eos_eot_inf and not guided_retention:
                    ending_offsets = torch.where(special_prediction[block_start:block_end])[0]
                    remask_offsets = torch.unique(torch.cat((remask_offsets, ending_offsets)))
                for offset in remask_offsets.tolist():
                    ids[answer_start + block_start + offset] = session.mask_token_id
        current_answer = ids[answer_start:]
        visible_answer = current_answer
        if session.tokenizer.eos_token_id in visible_answer:
            visible_answer = visible_answer[:visible_answer.index(session.tokenizer.eos_token_id)]
        remasked_text = decode_denoising_state(
            current_answer,
            session.tokenizer,
            session.mask_token_id,
            show_eos_tokens=include_pre_remask_prediction,
        )
        current_text = remasked_text
        if include_pre_remask_prediction:
            remask_label = (
                "State after re-mask"
                if block_step + 1 < block_steps and not stopped_early
                else "State after re-mask (unchanged; final state)"
            )
            current_text = (
                f"Predicted (before re-mask):\n{predicted_text}\n"
                f"{remask_label}:\n{remasked_text}"
            )
        status = f"Denoising step {step + 1}/{num_steps} · {len(visible_answer)} output tokens · mean confidence {last_confidence:.3f}"
        if num_blocks > 1:
            status += f" · block {block_index + 1}/{num_blocks}"
        if permanent_unmask:
            status += f" · retained {len(retained)} tokens"
        if stopped_early:
            status += " · block stopped early (same answer for 2 consecutive iterations)"
        html = render_denoising_step(
            ids,
            confidence.tolist(),
            answer_start,
            session.tokenizer,
            session.mask_token_id,
            step + 1,
            num_steps,
            retained if permanent_unmask else None,
            frozen if permanent_unmask and freeze_retained_tokens else None,
            frozen_confidences if permanent_unmask and freeze_retained_tokens else None,
            frozen_steps if permanent_unmask and freeze_retained_tokens else None,
            trajectory_color_mode,
        )
        yield current_text, status, html
        if stopped_early:
            if num_blocks == 1:
                break
            skip_block_index = block_index
    answer = ids[answer_start:]
    if session.tokenizer.eos_token_id in answer:
        answer = answer[:answer.index(session.tokenizer.eos_token_id)]
    text = session.tokenizer.decode(answer, skip_special_tokens=True).strip()
    return


def denoise(session: InferenceSession, question: str, system_prompt: str, max_new_tokens: int, num_steps: int, noise_level: float, temperature: float, top_k: int, seed: int, permanent_unmask: bool = False, confidence_guided: bool = False, proportional_unmask: bool = True, early_stopping: bool = False, progress: Callable[[float, str], None] | None = None, confidence_eos_eot_inf: bool = False, freeze_retained_tokens: bool = True, repetition_penalty: float = 1.0, eos_eot_prediction_penalty: float = 1.0, block_length: int | None = None) -> tuple[str, str]:
    """Run denoising to completion and return only the final text and status."""
    result = ("", "")
    for step, (text, status, _html) in enumerate(denoise_stream(session, question, system_prompt, max_new_tokens, num_steps, noise_level, temperature, top_k, seed, permanent_unmask, confidence_guided, proportional_unmask, early_stopping, confidence_eos_eot_inf, freeze_retained_tokens, repetition_penalty, eos_eot_prediction_penalty, False, block_length), start=1):
        result = (text, status)
        if progress:
            progress(step / int(num_steps), status)
    return result


def release_session(session: InferenceSession | None) -> None:
    """Free a loaded inference model and release backend allocator caches."""
    if session is None:
        return
    del session.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
