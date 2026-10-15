#!/usr/bin/env python3
"""Train one of the four BYOD masked-diffusion models."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv
import yaml

ROOT = Path(__file__).resolve().parent


def load_training_config(
    model: str,
    mode: str = "paper",
    *,
    output_dir: str | None = None,
    max_updates: int | None = None,
    prepared_data_cache_dir: str | None = None,
    prepared_data_persistent_cache_dir: str | None = None,
) -> dict:
    """Resolve the paper recipe or its smaller-model, rank-128 variant."""
    config = yaml.safe_load((ROOT / "configs" / "paper.yaml").read_text())
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    if model not in models:
        raise ValueError(f"Unknown model {model!r}; choose one of {', '.join(models)}")
    preset = deepcopy(models[model])
    quick_preset = preset.pop("quick", None)
    display_name = preset.pop("display_name")
    output_name = preset.pop("output_name")
    config.update(preset)
    config["selected_model"] = model
    config["display_name"] = display_name
    config["run_mode"] = mode

    if mode == "quick":
        if quick_preset is None:
            raise ValueError(
                f"{model!r} has no quick preset; choose llama, gemma, or qwen, "
                "or use --mode paper for Ministral 8B"
            )
        # Keep the complete 25k-update recipe and dataset budget. Quick mode
        # reduces memory through the smaller backbone and rank-128 adapters;
        # FP16 is used because common Colab T4 GPUs do not support BF16.
        quick_output_name = quick_preset.pop("output_name")
        config.update({
            "lora_r": 128,
            "lora_alpha": 128,
            "precision": "fp16",
        })
        config.update(quick_preset)
        output_name = quick_output_name
    elif mode != "paper":
        raise ValueError("mode must be 'quick' or 'paper'")

    if max_updates is not None:
        if max_updates < 1:
            raise ValueError("max_updates must be positive")
        config["max_updates"] = int(max_updates)
    config["output_dir"] = output_dir or f"outputs/{output_name}"
    if prepared_data_cache_dir is not None:
        config["prepared_data_cache_dir"] = prepared_data_cache_dir
    if prepared_data_persistent_cache_dir is not None:
        config["prepared_data_persistent_cache_dir"] = prepared_data_persistent_cache_dir
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["llama", "gemma", "qwen", "ministral"])
    parser.add_argument("--mode", choices=["quick", "paper"], default="paper")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-updates", type=int, help="Optional run-length override")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--logging-steps", type=int, help="Loss logging interval in optimizer updates")
    parser.add_argument("--prepared-data-cache-dir", help="Local tokenized-data cache")
    parser.add_argument("--prepared-data-persistent-cache-dir", help="Persistent backup restored into the local tokenized cache")
    return parser.parse_args()


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    config = load_training_config(
        args.model,
        args.mode,
        output_dir=args.output_dir,
        max_updates=args.max_updates,
        prepared_data_cache_dir=args.prepared_data_cache_dir,
        prepared_data_persistent_cache_dir=args.prepared_data_persistent_cache_dir,
    )
    if args.logging_steps is not None:
        if args.logging_steps < 1:
            raise ValueError("logging_steps must be positive")
        config["logging_steps"] = args.logging_steps
    print(
        f"Training {config['display_name']} in {args.mode!r} mode for "
        f"{config['max_updates']:,} updates -> {config['output_dir']}",
        flush=True,
    )
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False), flush=True)
        return
    from diffusion_lm.training import run_training

    run_training(config)


if __name__ == "__main__":
    main()
