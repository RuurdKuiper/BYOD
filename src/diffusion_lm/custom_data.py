"""Tokenize a user-supplied instruction dataset when no published profile matches."""
import os
import math

from .data import prepare_mask_only_cache_record


def tokenize_dataset(config, tokenizer, train_sample_limit):
    from datasets import DatasetDict, load_dataset

    if config["corruption_mode"] != "mask_only":
        raise ValueError("Custom raw datasets require corruption_mode: mask_only")
    kwargs = dict(revision=config.get("dataset_revision"), token=os.getenv("HF_TOKEN"),
                  cache_dir=config.get("cache_dir", "data/huggingface"))
    raw = load_dataset(config["dataset_name"], config.get("dataset_config"), **kwargs)
    specs = config.get("splits", {"train": "train", "validation": "validation", "test": "test"})
    columns = config.get("dataset_columns", {})
    data = DatasetDict()
    for name in ("train", "validation", "test"):
        spec = specs.get(name, name)
        if spec in raw:
            split = raw[spec]
        elif "[" in spec:
            split = load_dataset(config["dataset_name"], config.get("dataset_config"), split=spec, **kwargs)
        elif name == "train":
            raise ValueError(f"Training split {spec!r} was not found")
        else:
            continue
        split = split.map(lambda row, index: {"_index": index}, with_indices=True)
        mapping = {source: target for target, source in columns.items() if source != target}
        if mapping:
            split = split.rename_columns(mapping)
        if not {"instruction", "output"}.issubset(split.column_names):
            raise ValueError("Dataset needs instruction/output columns; use dataset_columns to map question/answer fields")
        data[name] = split.filter(lambda row: bool((row.get("output") or "").strip()))
    seed = int(config.get("seed", 42))
    missing = [name for name in ("validation", "test") if name not in data]
    if missing:
        fraction = float(config.get("heldout_fraction", .05))
        if not 0 < fraction * len(missing) < 1:
            raise ValueError("heldout_fraction must reserve a positive fraction and leave training examples")
        size = max(len(missing), math.ceil(len(data["train"]) * fraction * len(missing)))
        if size >= len(data["train"]):
            raise ValueError("Not enough nonempty training examples to create the missing held-out splits")
        reserved = data["train"].train_test_split(test_size=size, seed=seed)
        data["train"] = reserved["train"]
        if len(missing) == 2:
            halves = reserved["test"].train_test_split(test_size=.5, seed=seed)
            data[missing[0]], data[missing[1]] = halves["train"], halves["test"]
        else:
            data[missing[0]] = reserved["test"]
    data["train"] = data["train"].shuffle(seed=seed)
    if train_sample_limit is not None:
        data["train"] = data["train"].select(range(min(len(data["train"]), train_sample_limit)))
    for name, split in data.items():
        if not len(split):
            raise ValueError(f"Dataset split {name!r} has no nonempty answers")
        data[name] = split.map(
            lambda row: prepare_mask_only_cache_record(row, tokenizer, int(config["max_sequence_length"]),
                                                       bool(config.get("include_answer_eos", True))),
            remove_columns=split.column_names, writer_batch_size=128, desc=f"Tokenizing {name}",
        ).filter(lambda row: row["_lad_usable"])
        if not len(data[name]):
            raise ValueError(f"No answer tokens fit in {name!r}; increase max_sequence_length or shorten prompts")
    return data
