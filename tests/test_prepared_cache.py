"""Published tokens are downloaded once and safely reused across runtimes."""
from pathlib import Path
import shutil
from types import SimpleNamespace

from datasets import Dataset, DatasetDict
import pytest

from diffusion_lm import training
from diffusion_lm.prepared_cache import cache_complete


def setup_data(tmp_path, monkeypatch):
    config = {
        "dataset_name": "toy", "dataset_revision": "source-v1",
        "model_name_or_path": "toy", "model_revision": "model-v1",
        "corruption_mode": "mask_only", "max_sequence_length": 32,
        "prepared_data_cache_dir": str(tmp_path / "local"),
        "prepared_data_persistent_cache_dir": str(tmp_path / "persistent"),
    }
    tokenizer = SimpleNamespace(pad_token_id=0, chat_template="template")
    monkeypatch.setattr(training.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    data = DatasetDict({split: Dataset.from_list([
        {"_lad_clean_ids": [7, 2], "_index": i} for i in range(3)
    ]) for split in ("train", "validation", "test")})
    downloads = []

    def download(*args):
        downloads.append(True)
        return data

    def forbidden(*args, **kwargs):
        pytest.fail("Training must not load or tokenize the raw source")

    monkeypatch.setattr("diffusion_lm.prepared_cache.load_published_prepared_data", download)
    monkeypatch.setattr("datasets.load_dataset", forbidden)
    return config, downloads


def test_download_is_cached_without_raw_preparation(tmp_path, monkeypatch):
    config, downloads = setup_data(tmp_path, monkeypatch)
    first = training.load_training_data(config, 3)
    second = training.load_training_data(config, 3)
    assert len(downloads) == 1
    assert [d.to_dict() for d in first[1:]] == [d.to_dict() for d in second[1:]]


def test_drive_cache_restores_locally_without_another_download(tmp_path, monkeypatch):
    config, downloads = setup_data(tmp_path, monkeypatch)
    training.load_training_data(config, 3)
    shutil.rmtree(config["prepared_data_cache_dir"])
    training.load_training_data(config, 3)
    assert len(downloads) == 1
    assert cache_complete(next(Path(config["prepared_data_cache_dir"]).iterdir()))


@pytest.mark.parametrize("changed", [
    {"model_revision": "model-v2"}, {"dataset_revision": "source-v2"},
    {"max_sequence_length": 64}, {"seed": 7},
])
def test_recipe_change_does_not_reuse_incompatible_cache(tmp_path, monkeypatch, changed):
    config, downloads = setup_data(tmp_path, monkeypatch)
    training.load_training_data(config, 3)
    training.load_training_data({**config, **changed}, 3)
    assert len(downloads) == 2


def test_incomplete_cache_is_downloaded_again(tmp_path, monkeypatch):
    config, downloads = setup_data(tmp_path, monkeypatch)
    training.load_training_data(config, 3)
    persistent = next(Path(config["prepared_data_persistent_cache_dir"]).iterdir())
    next((persistent / "train").glob("*.arrow")).unlink()
    assert not cache_complete(persistent)
    shutil.rmtree(config["prepared_data_cache_dir"])
    training.load_training_data(config, 3)
    assert len(downloads) == 2 and cache_complete(persistent)


def test_missing_profile_uses_custom_tokenization_once_then_caches(tmp_path, monkeypatch):
    config, _ = setup_data(tmp_path, monkeypatch)
    monkeypatch.setattr("diffusion_lm.prepared_cache.load_published_prepared_data", lambda *a: None)
    calls = []
    def tokenize(*args):
        calls.append(True)
        return DatasetDict({split: Dataset.from_list([{"_lad_clean_ids": [7, 2], "_index": 0}])
                            for split in ("train", "validation", "test")})
    monkeypatch.setattr("diffusion_lm.custom_data.tokenize_dataset", tokenize)
    training.load_training_data(config, 3)
    training.load_training_data(config, 3)
    assert calls == [True]
