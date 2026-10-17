from datasets import Dataset, DatasetDict
import pytest

from diffusion_lm.custom_data import tokenize_dataset
from diffusion_lm.prepared_cache import preparation_key, preparation_metadata
from test_data_corruption_loss import ToyTokenizer


def test_custom_column_mapping_and_automatic_holdouts_are_disjoint_and_repeatable(monkeypatch):
    raw = DatasetDict(train=Dataset.from_list([
        {"question": f"Question {i}", "answer": f"Answer {i}"} for i in range(40)
    ]))
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: raw)
    config = {"dataset_name": "custom", "corruption_mode": "mask_only", "max_sequence_length": 64,
              "dataset_columns": {"instruction": "question", "output": "answer"}, "seed": 42}
    data = tokenize_dataset(config, ToyTokenizer(), 12)
    again = tokenize_dataset(config, ToyTokenizer(), 12)
    assert [len(data[k]) for k in ("train", "validation", "test")] == [12, 2, 2]
    ids = [set(data[k]["_index"]) for k in ("train", "validation", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert all(data[k].to_dict() == again[k].to_dict() for k in data)
    assert all("_lad_answer_mask" in d.column_names for d in data.values())


def test_existing_heldout_splits_are_preserved_and_empty_answers_filtered(monkeypatch):
    raw = DatasetDict({split: Dataset.from_list([
        {"instruction": "Question", "output": "Answer"}, {"instruction": "Empty", "output": ""}
    ]) for split in ("train", "valid", "test")})
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: raw)
    config = {"dataset_name": "custom", "corruption_mode": "mask_only", "max_sequence_length": 64,
              "splits": {"train": "train", "validation": "valid", "test": "test"}}
    data = tokenize_dataset(config, ToyTokenizer(), 20)
    assert all(len(d) == 1 for d in data.values())


def test_custom_data_explains_missing_columns(monkeypatch):
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: DatasetDict(
        train=Dataset.from_list([{"text": "example"}])))
    with pytest.raises(ValueError, match="dataset_columns"):
        tokenize_dataset({"dataset_name": "custom", "corruption_mode": "mask_only"}, ToyTokenizer(), None)


def test_custom_column_mapping_changes_the_token_cache_key():
    config = {"dataset_name": "custom", "model_name_or_path": "toy", "corruption_mode": "mask_only",
              "max_sequence_length": 64}
    original = preparation_key(preparation_metadata(config, ToyTokenizer(), 20))
    config["dataset_columns"] = {"instruction": "question", "output": "answer"}
    assert preparation_key(preparation_metadata(config, ToyTokenizer(), 20)) != original


def test_small_custom_dataset_can_reserve_one_example_per_heldout_split(monkeypatch):
    raw = DatasetDict(train=Dataset.from_list([
        {"instruction": "Q", "output": "A"} for _ in range(3)
    ]))
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: raw)
    data = tokenize_dataset({"dataset_name": "small", "corruption_mode": "mask_only",
                             "max_sequence_length": 64}, ToyTokenizer(), None)
    assert all(len(split) == 1 for split in data.values())
