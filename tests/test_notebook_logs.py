import subprocess
import sys

import pytest

from diffusion_lm import notebook


def test_failure_preserves_root_traceback_and_full_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(notebook, "_show", lambda *args: None)
    run = tmp_path / "run"
    code = (
        f"from pathlib import Path; Path({str(run)!r}).mkdir(); "
        "print('Traceback (most recent call last):\\nRuntimeError: preparation failed'); "
        "print('Exception ignored: BufferError cleanup noise\\n' * 100); "
        "raise SystemExit(3)"
    )
    with pytest.raises(subprocess.CalledProcessError) as caught:
        notebook.run_with_dashboard([sys.executable, "-c", code], run, 25000, refresh_seconds=0.01)
    assert caught.value.returncode == 3
    assert "RuntimeError: preparation failed" in caught.value.output
    assert "RuntimeError: preparation failed" in capsys.readouterr().out
    assert (run / "colab_training.log").is_file()


def test_failure_before_run_directory_exists_keeps_log_accessible(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(notebook, "_show", lambda *args: None)
    run = tmp_path / "run"
    with pytest.raises(subprocess.CalledProcessError):
        notebook.run_with_dashboard([sys.executable, "-c", "raise RuntimeError('import failed')"], run, 25000, refresh_seconds=0.01)
    log = tmp_path / ".run.training.log"
    assert log.is_file()
    assert str(log) in capsys.readouterr().out
