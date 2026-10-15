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
Path({str(run)!r}).mkdir()
print("Loading model", flush=True)
print("training: 0%| | 0/2", end="\\r", flush=True)
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
    assert "__BYOD_METRIC__" not in output and "training: 0%" not in output
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


def test_graph_keeps_all_loss_points_without_clearing_log_output(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import IPython.display

    class PlotOutput:
        clears = 0
        def clear_output(self, **kwargs):
            self.clears += 1
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "ipywidgets", SimpleNamespace(
        HTML=lambda value: SimpleNamespace(value=value), Output=PlotOutput))
    figures = []
    monkeypatch.setattr(IPython.display, "display", lambda *objects: figures.extend(
        o for o in objects if isinstance(o, Figure)))
    view = notebook._TrainingDisplay(3)
    for step, loss in ((1, 2.0), (2, 1.0)):
        view.record({"split": "train", "step": step, "weighted_loss": loss})
    view.record({"split": "validation", "step": 2, "weighted_loss": 1.2})
    view.record({"split": "train_interval", "step": 2, "weighted_loss": 1.5})
    assert view.plot_display.clears == 3
    assert len(figures) == 3
    assert list(figures[-1].axes[0].lines[0].get_xdata()) == [1, 2]
    assert list(figures[-1].axes[0].lines[1].get_ydata()) == [1.2]
    assert "2/3 updates" in view.status_display.value
