#!/usr/bin/env python3
"""Merge a trained LoRA adapter into a standalone diffusion model."""
from __future__ import annotations

import argparse
import json

from diffusion_lm.merging import merge_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", help="Usually outputs/<run>/best")
    parser.add_argument("output", help="New or empty destination directory")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    report = merge_adapter(args.adapter, args.output, dtype=args.dtype, device=args.device)
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
