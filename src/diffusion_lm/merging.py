"""Merge a saved LAD LoRA adapter into its base CausalLM."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .modeling import forward_bidirectional


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class MergeReport:
    base_model: str
    adapter_path: str
    output_path: str
    dtype: str
    normalization_tensors: int
    verification_max_abs_error: float | None
    verification_mean_abs_error: float | None


def _load_normalization_state(model: torch.nn.Module, adapter_path: Path) -> int:
    """Restore independently trained norms and fail if the checkpoint is incompatible."""
    state_path = adapter_path / "normalization_state.pt"
    if not state_path.is_file():
        return 0

    state = torch.load(state_path, map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    missing = sorted(name for name in state if name not in parameters)
    mismatched = sorted(
        name
        for name, value in state.items()
        if name in parameters and parameters[name].shape != value.shape
    )
    if missing or mismatched:
        raise ValueError(
            "The saved normalization state does not match the adapter/base model: "
            f"missing={missing[:5]}, shape_mismatch={mismatched[:5]}"
        )
    for name, value in state.items():
        parameter = parameters[name]
        parameter.data.copy_(value.to(parameter.device, dtype=parameter.dtype))
    return len(state)


@torch.inference_mode()
def _reference_logits(model: torch.nn.Module, tokenizer: Any, prompt: str) -> torch.Tensor:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16)
    input_ids = encoded["input_ids"].to(device)
    padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    return forward_bidirectional(model, input_ids, padding_mask).float().cpu()


def merge_adapter(
    adapter_path: str | Path,
    output_path: str | Path,
    *,
    dtype: str = "bf16",
    device: str = "cpu",
    cache_dir: str | Path | None = None,
    max_shard_size: str = "5GB",
    verify: bool = True,
    verification_prompt: str = "The capital of the Netherlands is",
) -> MergeReport:
    """Load, merge, verify, and save one adapter as a standalone model.

    The base model must be loaded without bitsandbytes quantization. A merged
    checkpoint can be quantized afterward for serving.
    """
    adapter_path = Path(adapter_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"Not a PEFT adapter directory: {adapter_path}")
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    adapter_config = json.loads(config_path.read_text())
    base_model = adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError("adapter_config.json has no base_model_name_or_path")

    run_config_path = adapter_path.parent / "resolved_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.is_file() else {}
    resolved_cache = str(cache_dir or run_config.get("base_model_cache_dir", "base_models"))
    token = os.getenv("HF_TOKEN")
    torch_dtype = DTYPES[dtype]

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        token=token,
        cache_dir=resolved_cache,
        revision=run_config.get("model_revision") or adapter_config.get("revision"),
    )
    base.config.use_cache = False
    base.config.is_causal = False
    if hasattr(base.config, "use_bidirectional_attention"):
        base.config.use_bidirectional_attention = True
    base.to(torch.device(device))

    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=False,
    ).eval()
    normalization_tensors = _load_normalization_state(model, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        use_fast=True,
        token=token,
        cache_dir=resolved_cache,
        clean_up_tokenization_spaces=False,
    )
    reference = _reference_logits(model, tokenizer, verification_prompt) if verify else None

    # This replaces every W + scale*(B@A) LoRA path with one ordinary W and
    # removes the adapter modules. safe_merge rejects NaN/Inf updates.
    merged = model.merge_and_unload(safe_merge=True, progressbar=True).eval()
    remaining_lora = [name for name, _ in merged.named_parameters() if "lora_" in name]
    if remaining_lora:
        raise RuntimeError(f"Merge left LoRA parameters behind: {remaining_lora[:5]}")
    merged.config.use_cache = False
    merged.config.is_causal = False
    if hasattr(merged.config, "use_bidirectional_attention"):
        merged.config.use_bidirectional_attention = True

    max_error = mean_error = None
    if reference is not None:
        candidate = _reference_logits(merged, tokenizer, verification_prompt)
        difference = (reference - candidate).abs()
        max_error = float(difference.max())
        mean_error = float(difference.mean())

    merged.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(output_path)
    if run_config:
        (output_path / "lad_run_config.json").write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n"
        )

    report = MergeReport(
        base_model=str(base_model),
        adapter_path=str(adapter_path),
        output_path=str(output_path),
        dtype=dtype,
        normalization_tensors=normalization_tensors,
        verification_max_abs_error=max_error,
        verification_mean_abs_error=mean_error,
    )
    (output_path / "lad_merge_report.json").write_text(
        json.dumps(report.__dict__, indent=2, sort_keys=True) + "\n"
    )
    return report
