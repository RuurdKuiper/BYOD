"""Compatibility classes for trusted full-model checkpoints from the legacy app."""
from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


class LegacyCustomTransformerConfig(PretrainedConfig):
    """Pickle-compatible replacement for ``model_config.CustomTransformerConfig``."""

    def __init__(self, vocab_size=128256, hidden_size=4096, num_layers=32, num_heads=32,
                 prediction_chunk=256, dropout=0, max_position_embeddings=4096,
                 masking_type="bidirectional", **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.prediction_chunk = prediction_chunk
        self.max_position_embeddings = max_position_embeddings
        self.input_size = prediction_chunk
        self.masking_type = masking_type


class LegacyCustomTransformerModel(PreTrainedModel):
    """Pickle-compatible legacy wrapper that supplies full bidirectional attention."""

    config_class = LegacyCustomTransformerConfig

    def forward(self, input_ids, labels=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        masking_type = getattr(self.config, "masking_type", "bidirectional")
        if masking_type == "bidirectional":
            base_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device)
        elif masking_type == "bidirectional_masked":
            base_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device)
            base_mask.fill_diagonal_(False)
        elif masking_type == "unidirectional":
            base_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device))
        else:
            raise ValueError(f"Unknown masking type: {masking_type}")
        llama = getattr(self.llama, "base_model", self.llama)
        compute_dtype = next(
            (parameter.dtype for parameter in llama.parameters() if parameter.is_floating_point()),
            torch.float32,
        )
        # The legacy checkpoint is commonly loaded in FP16 on Colab. SDPA
        # requires an additive attention bias to have the same dtype as the
        # query, so avoid the old unconditional float32 mask here.
        attention_mask = base_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, seq_len, seq_len).to(dtype=compute_dtype)
        # The hosted full checkpoint was serialized with peft==0.15.1.  Newer
        # PEFT's outer PeftModel.forward expects attributes absent from that
        # old pickled object.  Its base_model is the already-injected LoraModel
        # (and therefore retains the trained LoRA layers), so call it directly
        # when present rather than relying on version-sensitive PEFT hooks.
        outputs = llama(input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False, **kwargs)
        logits = outputs.logits[:, :, :self.config.vocab_size].view(batch_size, seq_len, self.config.vocab_size)
        if labels is None:
            return {"logits": logits}
        loss = nn.CrossEntropyLoss()(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return {"loss": loss, "logits": logits}


_MISSING = object()


def install_legacy_pickle_modules() -> dict[str, object]:
    """Temporarily register the historical class locations expected by torch.load."""
    previous: dict[str, object] = {name: sys.modules.get(name) for name in ("model_config", "models")}
    config_module = types.ModuleType("model_config")
    config_module.CustomTransformerConfig = LegacyCustomTransformerConfig
    model_module = types.ModuleType("models")
    model_module.CustomTransformerModel = LegacyCustomTransformerModel
    sys.modules["model_config"] = config_module
    sys.modules["models"] = model_module
    # Some notebook-created full checkpoints pickle these classes under
    # ``__main__`` rather than their original source modules.
    main_module = sys.modules["__main__"]
    for name, value in {
        "CustomTransformerConfig": LegacyCustomTransformerConfig,
        "CustomTransformerModel": LegacyCustomTransformerModel,
    }.items():
        previous[f"__main__.{name}"] = getattr(main_module, name, _MISSING)
        setattr(main_module, name, value)
    return previous


def restore_legacy_pickle_modules(previous: dict[str, object]) -> None:
    """Restore module registrations changed for one trusted checkpoint load."""
    for name in ("model_config", "models"):
        module = previous[name]
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module  # type: ignore[assignment]
    main_module = sys.modules["__main__"]
    for name in ("CustomTransformerConfig", "CustomTransformerModel"):
        previous_value = previous[f"__main__.{name}"]
        if previous_value is _MISSING:
            delattr(main_module, name)
        else:
            setattr(main_module, name, previous_value)


def patch_legacy_lora_modules(model: nn.Module) -> int:
    """Add fields expected by newer PEFT LoRA forwards to an old pickle.

    The hosted checkpoint predates PEFT's adapter-variant mechanism.  Its
    injected LoRA linears remain ordinary LoRA modules; an empty mapping makes
    current PEFT take that unchanged vanilla-LoRA branch.
    """
    patched = 0
    for module in model.modules():
        if hasattr(module, "lora_A") and not hasattr(module, "lora_variant"):
            module.lora_variant = {}
            patched += 1
    return patched
