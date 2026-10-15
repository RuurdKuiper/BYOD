"""Exercise the public setup cells without network access or a CUDA build."""
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def code_cell(marker):
    notebook = json.loads((ROOT / "notebooks/train_diffusion.ipynb").read_text())
    return next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    )


@pytest.mark.parametrize("torchao_version", [None, "0.10.0", "0.16.0"])
def test_default_setup_imports_checkout_in_live_kernel_without_optional_cuda_builds(
    tmp_path, monkeypatch, torchao_version,
):
    checkout = tmp_path / "BYOD"
    package = checkout / "src/diffusion_lm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (checkout / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.delitem(sys.modules, "diffusion_lm", raising=False)
    real_version = importlib.metadata.version

    def version(name):
        if name != "torchao":
            return real_version(name)
        if torchao_version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return torchao_version

    monkeypatch.setattr(importlib.metadata, "version", version)
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command))
    source = code_cell("def setup_step").replace(
        'Path("/content/BYOD")', f"Path({str(checkout)!r})",
    )
    namespace = {}
    exec(compile(source, "notebook-setup", "exec"), namespace)

    assert Path(namespace["diffusion_lm"].__file__) == package / "__init__.py"
    installs = [command for command in commands if "install" in command]
    assert len(installs) == 1 and ".[colab]" in installs[0]
    assert "-q" not in installs[0]
    assert bool([command for command in commands if "uninstall" in command]) == (
        torchao_version == "0.10.0"
    )


@pytest.mark.parametrize(
    "mode,capability,installed,expect_install",
    [
        ("quick", (9, 0), False, False),
        ("paper", (8, 0), False, False),
        ("paper", (8, 9), False, True),
        ("paper", (9, 0), False, True),
        ("paper", (12, 0), True, False),
    ],
)
def test_fp8_setup_only_builds_for_paper_mode_on_supported_gpu(
    monkeypatch, mode, capability, installed, expect_install,
):
    from diffusion_lm import notebook as notebook_helpers

    monkeypatch.setattr(notebook_helpers, "remove_unused_transformer_engine", lambda: [])
    monkeypatch.setattr(importlib, "reload", lambda module: module)
    cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: capability)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    def version(name):
        assert name == "transformer-engine"
        if not installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return "2.13.0"

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if installed else None)
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command))
    namespace = {"mode": SimpleNamespace(value=mode)}
    exec(compile(code_cell("native_fp8"), "notebook-fp8", "exec"), namespace)
    assert bool(commands) == expect_install
    if expect_install:
        assert "--no-build-isolation" in commands[0]
        assert "transformer-engine[pytorch]==2.13.0" in commands[0]


def test_default_extra_has_no_fp8_or_quantization_dependency():
    tomllib = pytest.importorskip("tomllib")

    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert extras["colab"] == ["ipywidgets>=8"]
    assert extras["fp8"] == ["transformer-engine[pytorch]==2.13.0"]
    assert extras["cuda"] == ["bitsandbytes>=0.45"]


@pytest.mark.parametrize("present", [[], ["transformer-engine-cu13"], [
    "transformer-engine", "transformer-engine-torch", "transformer-engine-cu12",
]])
def test_cleanup_only_uninstalls_present_fp8_distributions(monkeypatch, present):
    from diffusion_lm.notebook import remove_unused_transformer_engine

    def version(name):
        if name not in present:
            raise importlib.metadata.PackageNotFoundError(name)
        return "2.13.0"

    monkeypatch.setattr(importlib.metadata, "version", version)
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command))
    assert remove_unused_transformer_engine() == present
    if present:
        assert commands == [[sys.executable, "-m", "pip", "uninstall", "-y", *present]]
    else:
        assert not commands


@pytest.mark.parametrize("mode,capability,needs_cleanup", [
    ("quick", (9, 0), True),
    ("quick", (7, 5), True),
    ("paper", (8, 0), True),
    ("paper", (9, 0), False),
])
def test_training_cleans_non_fp8_environment_even_if_optional_cell_was_skipped(
    monkeypatch, mode, capability, needs_cleanup,
):
    from diffusion_lm import notebook as notebook_helpers

    events = []
    monkeypatch.setattr(importlib, "reload", lambda module: events.append("reload") or module)
    monkeypatch.setattr(notebook_helpers, "remove_unused_transformer_engine", lambda: events.append("cleanup"))
    monkeypatch.setattr(notebook_helpers, "run_with_dashboard", lambda *args, **kwargs: events.append("train"))
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda index: "Test GPU",
        get_device_capability=lambda: capability,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    def config(*args, **kwargs):
        # No model-training imports should precede the environment cleanup.
        assert ("cleanup" in events) == needs_cleanup
        return {"display_name": "Test", "max_updates": 25000, "lora_r": 128}

    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(load_training_config=config))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    namespace = {
        "REPOSITORY_DIR": ROOT,
        "mode": SimpleNamespace(value=mode),
        "model": SimpleNamespace(value="llama"),
        "save_to_drive": SimpleNamespace(value=False),
    }
    exec(compile(code_cell("run_with_dashboard"), "notebook-training", "exec"), namespace)
    assert events == (["reload", "cleanup", "train"] if needs_cleanup else ["reload", "train"])
