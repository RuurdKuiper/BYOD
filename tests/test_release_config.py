import json
from pathlib import Path

import pytest
import yaml

from train import load_training_config


@pytest.mark.parametrize("model", ["llama", "gemma", "qwen", "ministral"])
def test_paper_presets_keep_the_original_recipe(model):
    config = load_training_config(model)
    assert config["dataset_name"] == "Ruurd/LAD-training-1m-256"
    assert config["max_updates"] == 25_000
    assert config["batch_size"] == 16
    assert config["lora_r"] == config["lora_alpha"] == 1024
    assert config["corruption_mode"] == "mask_only"
    assert config["structured_loss_behavior"] == "corrupted_answer_tokens"
    assert config["frontier_masking_probability"] == 0.0
    assert config["answer_padding_loss"] == {
        "enabled": True, "answer_weight": 0.9, "padding_weight": 0.1,
    }
    assert "fp8" not in config
    assert config["quantization"] == "none"


def test_quick_mode_is_explicitly_smaller():
    config = load_training_config("llama", "quick")
    assert config["run_mode"] == "quick"
    assert config["max_updates"] == 25_000
    assert config["lora_r"] == config["lora_alpha"] == 128
    assert config["model_name_or_path"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert "fp8" not in config
    assert config["quantization"] == "none"
    assert config["generation_perplexity"]["enabled"] is True


@pytest.mark.parametrize("model", ["llama", "gemma", "qwen"])
def test_quick_mode_preserves_the_full_recipe_except_model_rank_and_precision(model):
    paper = load_training_config(model, "paper")
    quick = load_training_config(model, "quick")
    allowed_differences = {
        "display_name", "lora_alpha", "lora_r", "model_name_or_path",
        "model_revision", "output_dir", "precision", "run_mode",
        "tokenizer_name_or_path",
        "prepared_dataset_config",
    }
    assert {
        key for key in paper | quick if paper.get(key) != quick.get(key)
    } == allowed_differences


@pytest.mark.parametrize(
    ("model", "expected_model", "quantization"),
    [
        ("llama", "meta-llama/Llama-3.2-1B-Instruct", "none"),
        ("gemma", "google/gemma-3-1b-it", "none"),
        ("qwen", "Qwen/Qwen2.5-1.5B-Instruct", "none"),
    ],
)
def test_quick_mode_selects_smaller_compatible_models(model, expected_model, quantization):
    config = load_training_config(model, "quick")
    assert config["model_name_or_path"] == expected_model
    assert config["quantization"] == quantization
    assert config["lora_r"] == config["lora_alpha"] == 128
    assert config["output_dir"].endswith({
        "llama": "llama-3.2-1b-mask",
        "gemma": "gemma-3-1b-mask",
        "qwen": "qwen-2.5-1.5b-mask",
    }[model])


def test_ministral_is_available_only_for_paper_reproduction():
    with pytest.raises(ValueError, match="no quick preset"):
        load_training_config("ministral", "quick")
    assert load_training_config("ministral", "paper")["model_name_or_path"] == (
        "mistralai/Ministral-8B-Instruct-2410"
    )


def test_llama_paper_config_preserves_historical_settings_with_confirmed_loss():
    root = Path(__file__).parents[1]
    reference = json.loads(
        (root / "reference/llama-3.1-8b-mask.resolved_config.json").read_text()
    )
    config = load_training_config("llama")
    runtime_fields = {
        "fp8", "effective_batch_size", "fp8_active", "fp8_compute_capability",
        "fp8_device_name", "fp8_requested", "learning_rate_scale",
        "resolved_learning_rate", "resolved_training_precision",
        "structured_marker_dropped", "training_sample_limit",
        "training_samples_used", "validation_samples_used", "output_dir",
    }
    for key, value in reference.items():
        if key not in runtime_fields:
            assert config[key] == value
    # The historical file predates the author-confirmed paper objective.
    assert "answer_padding_loss" not in reference
    assert config["answer_padding_loss"] == {
        "enabled": True, "answer_weight": 0.9, "padding_weight": 0.1,
    }


def test_every_external_input_is_revision_pinned():
    root = Path(__file__).parents[1]
    models = yaml.safe_load((root / "configs/models.yaml").read_text())
    paper = yaml.safe_load((root / "configs/paper.yaml").read_text())
    assert len(paper["dataset_revision"]) == 40
    assert all(len(preset["model_revision"]) == 40 for preset in models.values())
    assert all(
        len(preset["quick"]["model_revision"]) == 40
        for preset in models.values()
        if "quick" in preset
    )


def test_public_notebook_exposes_every_paper_model_and_has_no_saved_outputs():
    root = Path(__file__).parents[1]
    notebook = json.loads((root / "notebooks/train_diffusion.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for model in ("llama", "gemma", "qwen", "ministral"):
        assert f'"{model}"' in source
    assert "25,000 updates" in source
    assert "Dovelove/byod-llama-3.1-8b" in source
    assert all(
        not cell.get("outputs") and cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_readme_links_colab_and_all_live_demos():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    assert "colab.research.google.com/github/RuurdKuiper/BYOD" in readme
    for slug in (
        "byod-gemma-2-9b", "byod-llama-3.1-8b",
        "byod-qwen2.5-7b", "byod-ministral-8b",
    ):
        assert f"huggingface.co/spaces/Dovelove/{slug}" in readme


def test_print_config_exits_without_starting_training(monkeypatch, capsys):
    import sys
    from types import SimpleNamespace
    import train

    def unexpected_training(*args, **kwargs):
        pytest.fail("--print-config must not load a model or start training")

    monkeypatch.setitem(sys.modules, "diffusion_lm.training", SimpleNamespace(run_training=unexpected_training))
    monkeypatch.setattr(sys, "argv", ["train.py", "--model", "llama", "--print-config"])
    train.main()
    output = capsys.readouterr().out
    assert "answer_weight: 0.9" in output
    assert "padding_weight: 0.1" in output


def test_training_cli_forwards_cache_options(monkeypatch):
    import sys
    from types import SimpleNamespace
    import train

    configs = []
    monkeypatch.setitem(sys.modules, "diffusion_lm.training", SimpleNamespace(run_training=configs.append))
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--model", "llama", "--mode", "quick",
        "--prepared-data-cache-dir", "/tmp/local-prepared",
        "--prepared-data-persistent-cache-dir", "/tmp/persistent-prepared",
    ])
    train.main()
    config = configs[0]
    assert config["prepared_data_cache_dir"] == "/tmp/local-prepared"
    assert config["prepared_data_persistent_cache_dir"] == "/tmp/persistent-prepared"
    assert config["num_workers"] == 4
