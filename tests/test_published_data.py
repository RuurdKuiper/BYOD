import json
from types import SimpleNamespace

from datasets import Dataset, DatasetDict
import pytest

from diffusion_lm.prepared_cache import load_published_prepared_data, preparation_metadata


def hub_profile(tmp_path, monkeypatch):
    config = {"dataset_name": "source", "dataset_revision": "source-v1",
              "model_name_or_path": "tokenizer", "model_revision": "tokenizer-v1",
              "corruption_mode": "mask_only", "max_sequence_length": 256,
              "prepared_dataset_name": "owner/prepared", "prepared_dataset_config": "profile",
              "prepared_dataset_revision": "pinned-hub-commit", "cache_dir": str(tmp_path)}
    tokenizer = SimpleNamespace(chat_template="template")
    record = {"_lad_clean_ids": [1, 2], "_lad_answer_mask": [True, True],
              "_lad_padding_mask": [False, False], "_lad_usable": True,
              "_lad_empty_answer": False, "_lad_truncated": False, "_index": 0}
    data = DatasetDict({split: Dataset.from_list([{**record, "_index": i} for i in range(4)])
                        for split in ("train", "validation", "test")})
    manifest = {"preparation": preparation_metadata(config, tokenizer, 400000),
                "rows": {split: 4 for split in data}}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(manifest))
    downloads = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(path))
    def load(*args, **kwargs):
        downloads.append((args, kwargs))
        return data
    monkeypatch.setattr("datasets.load_dataset", load)
    return config, tokenizer, downloads


def test_published_profile_uses_pinned_revision_and_original_prefix(tmp_path, monkeypatch):
    config, tokenizer, downloads = hub_profile(tmp_path, monkeypatch)
    data = load_published_prepared_data(config, tokenizer, 2)
    assert list(data["train"]["_index"]) == [0, 1]
    assert len(data["validation"]) == len(data["test"]) == 4
    assert downloads[0][1]["revision"] == "pinned-hub-commit"


@pytest.mark.parametrize("changes", [{"model_revision": "wrong"}, {"seed": 99}, {"max_sequence_length": 512}])
def test_incompatible_published_profile_is_rejected_before_downloading_tokens(tmp_path, monkeypatch, changes):
    config, tokenizer, downloads = hub_profile(tmp_path, monkeypatch)
    assert load_published_prepared_data({**config, **changes}, tokenizer, 2) is None
    assert not downloads


def test_larger_training_budget_does_not_silently_reuse_shorter_profile(tmp_path, monkeypatch):
    config, tokenizer, downloads = hub_profile(tmp_path, monkeypatch)
    assert load_published_prepared_data(config, tokenizer, 800000) is None
    assert not downloads


def test_unpinned_profile_is_not_used(tmp_path, monkeypatch):
    config, tokenizer, downloads = hub_profile(tmp_path, monkeypatch)
    config.pop("prepared_dataset_revision")
    assert load_published_prepared_data(config, tokenizer, 2) is None
    assert not downloads
