import subprocess
import sys
from types import SimpleNamespace

import pytest

from diffusion_lm import notebook


def test_failure_preserves_root_traceback_and_full_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(notebook, "_TrainingDisplay", lambda *args: SimpleNamespace(status=lambda *a: None, record=lambda *a: None))
    run = tmp_path / "run"
    code = (
        f"from pathlib import Path; Path({str(run)!r}).mkdir(); "
        "print('Traceback (most recent call last):\\nRuntimeError: preparation failed'); "
        "print('Exception ignored: BufferError cleanup noise\\n' * 100); "
        "raise SystemExit(3)"
    )
    with pytest.raises(subprocess.CalledProcessError) as caught:
        notebook.run_with_dashboard([sys.executable, "-c", code], run, 25000)
    assert caught.value.returncode == 3
    assert "RuntimeError: preparation failed" in caught.value.output
    assert "RuntimeError: preparation failed" in capsys.readouterr().out
    assert (run / "colab_training.log").is_file()


def test_failure_before_run_directory_exists_keeps_log_accessible(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(notebook, "_TrainingDisplay", lambda *args: SimpleNamespace(status=lambda *a: None, record=lambda *a: None))
    run = tmp_path / "run"
    with pytest.raises(subprocess.CalledProcessError):
        notebook.run_with_dashboard([sys.executable, "-c", "raise RuntimeError('import failed')"], run, 25000)
    log = tmp_path / ".run.training.log"
    assert log.is_file()
    assert str(log) in capsys.readouterr().out


def test_messages_stream_and_metric_events_update_before_process_exits(tmp_path, monkeypatch, capsys):
    records, statuses = [], []
    acknowledgment = tmp_path / "first-update-seen"

    def record(value):
        records.append(value)
        acknowledgment.touch()

    monkeypatch.setattr(notebook, "_TrainingDisplay", lambda *args: SimpleNamespace(status=statuses.append, record=record))
    run = tmp_path / "run"
    code = f'''
import json, os, time
from pathlib import Path
assert os.environ["BYOD_NOTEBOOK"] == "1"
assert os.environ["USE_TF"] == "0"
assert os.environ["USE_TORCH"] == "1"
Path({str(run)!r}).mkdir()
print("Loading model", flush=True)
print("\\rtraining: 0%| | 0/2", end="", flush=True)
print('__BYOD_METRIC__' + json.dumps(dict(split="train", step=1, weighted_loss=2.0)), flush=True)
deadline = time.monotonic() + 5
while not Path({str(acknowledgment)!r}).exists():
    if time.monotonic() > deadline:
        raise RuntimeError("Notebook did not process the live update")
    time.sleep(.01)
print("__BYOD_METRIC__" + json.dumps(dict(split="train", step=2, weighted_loss=1.0)), flush=True)
print("Validation complete", flush=True)
'''
    notebook.run_with_dashboard([sys.executable, "-u", "-c", code], run, 2)
    assert [r["step"] for r in records] == [1, 2]
    output = capsys.readouterr().out
    assert "Loading model" in output and "Validation complete" in output
    assert "__BYOD_METRIC__" not in output
    assert "training: 0%" in output  # Native progress remains visible without widgets.
    assert "Loading model" in (run / "colab_training.log").read_text()
    assert statuses[-1] == "Finished: 2 updates"


def test_trainer_emits_metric_event_only_for_notebook_training(tmp_path, monkeypatch, capsys):
    import json
    from diffusion_lm.training import _append_jsonl
    value = {"split": "train", "step": 1, "weighted_loss": 1.5}
    path = tmp_path / "metrics.jsonl"
    monkeypatch.delenv("BYOD_NOTEBOOK", raising=False)
    _append_jsonl(path, value)
    assert capsys.readouterr().out == ""
    monkeypatch.setenv("BYOD_NOTEBOOK", "1")
    _append_jsonl(path, value)
    event = capsys.readouterr().out
    assert json.loads(event.removeprefix("__BYOD_METRIC__")) == value
    assert len(path.read_text().splitlines()) == 2


def test_graph_reuses_one_image_and_follows_validation_without_losing_points(monkeypatch):
    import IPython.display

    class PlotImage:
        def __init__(self, **kwargs):
            self.value = b""

    monkeypatch.setitem(sys.modules, "ipywidgets", SimpleNamespace(
        HTML=lambda value: SimpleNamespace(value=value), Image=PlotImage))
    displayed = []
    monkeypatch.setattr(IPython.display, "display", lambda *objects: displayed.extend(objects))
    clock = [0.0]
    monkeypatch.setattr(notebook.time, "monotonic", lambda: clock[0])
    view = notebook._TrainingDisplay(5)
    view.record({"split": "train", "step": 1, "weighted_loss": 100.0})
    figure = view._figure
    axis = figure.axes[0]
    first_png = view.plot_display.value
    assert first_png.startswith(b"\x89PNG")
    assert axis.get_ylim()[0] < 100 < axis.get_ylim()[1]
    view.record({"split": "train", "step": 2, "weighted_loss": 1.0})
    assert view.plot_display.value is first_png  # Record every update, throttle rendering.
    view.record({"split": "validation", "step": 2, "weighted_loss": 1.2})
    assert view._figure is figure
    assert view.plot_display.value != first_png  # Always refresh after validation.
    assert list(axis.lines[0].get_xdata()) == [1, 2]
    assert list(axis.lines[1].get_ydata()) == [1.2]
    assert axis.get_ylim() == pytest.approx((.2, 2.2))
    view.record({"split": "validation", "step": 2, "weighted_loss": float("nan")})
    assert axis.get_ylim() == pytest.approx((.2, 2.2))
    limits = axis.get_ylim()
    clock[0] = 6.0
    view.record({"split": "train", "step": 3, "weighted_loss": 1000.0})
    assert axis.get_ylim() == limits
    assert list(axis.lines[0].get_ydata()) == [100.0, 1.0, 1000.0]
    view.record({"split": "train_interval", "step": 3, "weighted_loss": 1.5})
    view.record({"split": "validation", "step": 4, "weighted_loss": .8})
    assert axis.get_ylim() == pytest.approx((-.2, 2.2))
    view.record({"split": "train", "step": 5, "weighted_loss": 1.0})
    assert list(axis.lines[0].get_xdata()) == [1, 2, 3, 5]  # Final update bypasses throttle.
    assert displayed == [view.status_display, view.plot_display]
    assert len(figure.axes) == 1 and len(axis.lines) == 2
    assert "5/5 updates" in view.status_display.value


def test_checkpoint_selection_after_failed_or_interrupted_training(tmp_path):
    def adapter(name):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "adapter_config.json").write_text("{}")
        (folder / "adapter_model.safetensors").write_bytes(b"saved")
        return folder
    with pytest.raises(FileNotFoundError, match="No saved adapter"):
        notebook.saved_checkpoint(tmp_path)
    old = adapter("checkpoint-250")
    latest = adapter("checkpoint-1000")
    assert notebook.saved_checkpoint(tmp_path) == latest
    best = adapter("best")
    assert notebook.saved_checkpoint(tmp_path) == best
    interrupted = adapter("interrupted")
    assert notebook.saved_checkpoint(tmp_path) == interrupted
    (interrupted / "adapter_model.safetensors").unlink()
    assert notebook.saved_checkpoint(tmp_path) == best


def test_notebook_stop_sends_interrupt_and_waits_for_saved_adapter(tmp_path, monkeypatch):
    import json
    run = tmp_path / "run"
    def interrupt(record):
        raise KeyboardInterrupt
    monkeypatch.setattr(notebook, "_TrainingDisplay", lambda *a: SimpleNamespace(record=interrupt))
    code = f'''
import time
from pathlib import Path
folder = Path({str(run)!r})
folder.mkdir()
try:
    print('__BYOD_METRIC__{{"split":"train","step":1,"weighted_loss":1.0}}', flush=True)
    while True:
        time.sleep(.05)
except KeyboardInterrupt:
    (folder / "interrupted").mkdir()
    (folder / "interrupted" / "state.json").write_text('{{"step":1}}')
    print("adapter saved to interrupted", flush=True)
'''
    with pytest.raises(KeyboardInterrupt):
        notebook.run_with_dashboard([sys.executable, "-u", "-c", code], run, 2)
    assert json.loads((run / "interrupted/state.json").read_text())["step"] == 1
    assert "adapter saved to interrupted" in (run / "colab_training.log").read_text()
