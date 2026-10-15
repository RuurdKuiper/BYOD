"""Persist clean-token datasets and stage them locally for fast training reads."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


SPLITS = ("train", "validation", "test")


def preparation_metadata(config, tokenizer, train_sample_limit):
    return {
        "format": 2, "dataset": config["dataset_name"],
        "dataset_revision": config.get("dataset_revision"), "config": config.get("dataset_config"),
        "splits": config.get("splits", {"train": "train", "validation": "validation", "test": "test"}),
        "tokenizer": config.get("tokenizer_name_or_path", config["model_name_or_path"]),
        "model_revision": config.get("model_revision"),
        "chat_template": getattr(tokenizer, "chat_template", None), "mode": config["corruption_mode"],
        "max_length": int(config["max_sequence_length"]),
        "include_answer_eos": bool(config.get("include_answer_eos", True)),
        "train_sample_limit": train_sample_limit, "seed": int(config.get("seed", 42)),
    }


def preparation_key(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]


def load_published_prepared_data(config, tokenizer, train_sample_limit):
    """Load a pinned Hub profile only when its clean tokens match this recipe."""
    repo = config.get("prepared_dataset_name")
    profile = config.get("prepared_dataset_config")
    revision = config.get("prepared_dataset_revision")
    if not repo or not profile or not revision:
        return None
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset
    from huggingface_hub.errors import EntryNotFoundError

    try:
        manifest = hf_hub_download(
            repo, f"prepared/{profile}.json", repo_type="dataset", revision=revision,
            cache_dir=str(Path(config.get("cache_dir", "data/huggingface")) / "prepared-hub"),
            token=os.getenv("HF_TOKEN"),
        )
    except EntryNotFoundError:
        print(f"No published token profile for {profile}.", flush=True)
        return None
    published = json.loads(Path(manifest).read_text())
    expected = preparation_metadata(config, tokenizer, train_sample_limit)
    actual = published["preparation"]
    differing = [key for key in expected if key != "train_sample_limit" and expected[key] != actual.get(key)]
    limit = actual.get("train_sample_limit")
    if limit is not None and (train_sample_limit is None or train_sample_limit > limit):
        differing.append("train_sample_limit")
    if differing:
        print("Published tokens do not match " + ", ".join(differing) + ".", flush=True)
        return None
    print(f"Downloading pretokenized data: {repo} / {profile} @ {revision[:8]}", flush=True)
    data = load_dataset(
        repo, profile, revision=revision,
        cache_dir=config.get("cache_dir", "data/huggingface"), token=os.getenv("HF_TOKEN"),
    )
    required = {"_lad_clean_ids", "_lad_answer_mask", "_lad_padding_mask", "_lad_usable", "_lad_empty_answer", "_lad_truncated", "_index"}
    for split in SPLITS:
        if not required.issubset(data[split].column_names) or len(data[split]) != published["rows"][split]:
            raise ValueError(f"Published token profile has invalid columns or row count: {profile}/{split}")
    if train_sample_limit is not None and len(data["train"]) > train_sample_limit:
        data["train"] = data["train"].select(range(train_sample_limit))
    return data


def cache_complete(root: Path) -> bool:
    """Reject missing splits or Arrow files left by interrupted preparation."""
    try:
        for split in SPLITS:
            folder = root / split
            if not (folder / "dataset_info.json").is_file():
                return False
            state = json.loads((folder / "state.json").read_text())
            for item in state["_data_files"]:
                if not (folder / item["filename"]).is_file():
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _copy_complete_cache(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "cache"
        shutil.copytree(source, staged)
        if not cache_complete(staged):
            raise ValueError(f"Incomplete tokenized cache: {source}")
        if destination.exists():
            # Only this derived cache key is replaced, never its parent folder.
            shutil.rmtree(destination)
        staged.rename(destination)


def restore_prepared_cache(local: Path, persistent: Path | None) -> bool:
    if cache_complete(local):
        return True
    if persistent is not None and local != persistent and cache_complete(persistent):
        print(f"Restoring tokenized data from {persistent} to local disk…", flush=True)
        _copy_complete_cache(persistent, local)
        return True
    return False


def persist_prepared_cache(local: Path, persistent: Path | None) -> None:
    if persistent is None or local == persistent or cache_complete(persistent):
        return
    if not cache_complete(local):
        raise ValueError(f"Cannot persist incomplete tokenized cache: {local}")
    print(f"Saving reusable tokenized data to {persistent}…", flush=True)
    _copy_complete_cache(local, persistent)
