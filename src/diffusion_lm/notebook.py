"""Small live dashboard used by the public training notebook."""
from __future__ import annotations

import importlib
from importlib import metadata
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import signal
import sys
import time
from typing import Sequence


def remove_unused_transformer_engine() -> list[str]:
    """Remove incompatible optional libraries that PEFT may import on Colab."""
    installed = []
    for name in (
        "transformer-engine", "transformer-engine-torch",
        "transformer-engine-cu12", "transformer-engine-cu13",
    ):
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        installed.append(name)
    if installed:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", *installed],
            check=True, stdout=subprocess.DEVNULL,
        )
        importlib.invalidate_caches()
    return installed


def training_error_excerpt(text: str, limit: int = 16000) -> str:
    """Preserve the first traceback when shutdown noise fills the log tail."""
    start = text.find("Traceback (most recent call last)")
    if start >= 0:
        return text[max(0, start - 500):start + limit]
    return text[-limit:]


class _TrainingDisplay:
    """Update only the status and plot; ordinary log messages stay visible."""

    def __init__(self, max_updates: int):
        from IPython.display import display
        import ipywidgets as widgets
        self.max_updates = max_updates
        self.started = time.monotonic()
        self.status_display = widgets.HTML("Preparing data and model…")
        self.plot_display = widgets.Image(format="png")
        display(self.status_display, self.plot_display)
        self.train = []
        self.validation = []
        self._figure = None
        self._last_plot_at = float("-inf")

    def status(self, text: str) -> None:
        from html import escape
        self.status_display.value = "<pre>" + escape(text) + "</pre>"

    def record(self, record: dict) -> None:
        split = record.get("split")
        if split not in {"train", "validation"}:
            return
        points = self.train if split == "train" else self.validation
        points.append({"step": record["step"], "weighted_loss": record["weighted_loss"]})
        step = int(record["step"])
        elapsed = (time.monotonic() - self.started) / 60
        self.status(f"Training: {step:,}/{self.max_updates:,} updates · "
                    f"{split} loss {record['weighted_loss']:.4f} · elapsed {elapsed:.1f} min")
        now = time.monotonic()
        if split == "train" and step != self.max_updates and now - self._last_plot_at < 5:
            return
        # Reuse one figure and replace one widget's PNG bytes. Creating and
        # displaying a new figure every optimizer update is expensive in Colab.
        if self._figure is None:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            self._figure = Figure(figsize=(9, 3))
            FigureCanvasAgg(self._figure)
            axis = self._figure.subplots()
            for label in ("training", "validation"):
                axis.plot([], [], label=label, linewidth=1.3)
            axis.set(xlabel="update", ylabel="denoising loss")
            axis.grid(alpha=.25)
            axis.legend()
        figure = self._figure
        axis = figure.axes[0]
        for values, line in zip((self.train, self.validation), axis.lines):
            line.set_data([r["step"] for r in values], [r["weighted_loss"] for r in values])
        axis.relim()
        axis.autoscale_view(scaley=False)
        # Keep training spikes from compressing the validation curve.
        losses = [r["weighted_loss"] for r in self.validation
                  if math.isfinite(r["weighted_loss"])]
        if not losses:
            losses = [r["weighted_loss"] for r in self.train
                      if math.isfinite(r["weighted_loss"])]
        if losses:
            low, high = min(losses), max(losses)
            padding = max((high - low) * .1, abs(low) * .05, .01)
            axis.set_ylim(low - padding, high + padding)
        figure.tight_layout()
        with BytesIO() as image:
            figure.canvas.print_png(image)
            self.plot_display.value = image.getvalue()
        self._last_plot_at = time.monotonic()


def run_with_dashboard(command: Sequence[str], run_dir: str | Path, max_updates: int) -> None:
    """Stream all logs and refresh the plot when the trainer emits a metric."""
    import os

    run_dir = Path(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(f"Choose a new run directory; this one already exists: {run_dir}")
    log_path = run_dir.parent / f".{run_dir.name}.training.log"
    view = _TrainingDisplay(max_updates)
    environment = dict(os.environ, BYOD_NOTEBOOK="1", PYTHONUNBUFFERED="1", USE_TF="0", USE_TORCH="1")
    with log_path.open("w") as log:
        process = subprocess.Popen(
            list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1, env=environment, start_new_session=True,
        )
        # Keep tqdm carriage returns so the notebook can replace the same line.
        process.stdout.reconfigure(newline="")
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                message, marker, payload = line.partition("__BYOD_METRIC__")
                # tqdm does not end its progress line with a newline. A metric
                # event may therefore share that line; never discard it as a bar.
                if message:
                    print(message, end="", flush=True)
                if marker:
                    view.record(json.loads(payload))
            process.wait()
        except BaseException as error:
            if process.poll() is None:
                process.send_signal(signal.SIGINT if isinstance(error, KeyboardInterrupt) else signal.SIGTERM)
                print("\nStopping training; waiting for the trainer to exit…", flush=True)
                try:
                    remaining, _ = process.communicate(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    remaining, _ = process.communicate()
                log.write(remaining or "")
                for message in (remaining or "").splitlines():
                    if "adapter saved to" in message:
                        print(message, flush=True)
            if run_dir.is_dir():
                log.flush()
                log_path.replace(run_dir / "colab_training.log")
            raise
        finally:
            process.stdout.close()
    if run_dir.is_dir():
        log_path = log_path.replace(run_dir / "colab_training.log")
    if process.returncode:
        error = training_error_excerpt(log_path.read_text(errors="replace"))
        print(f"\nTraining failed. Full log: {log_path}\n\n{error}", flush=True)
        raise subprocess.CalledProcessError(process.returncode, command, output=error)
    view.status(f"Finished: {max_updates:,} updates")
    print(f"\nArtifacts: {run_dir}")


def saved_checkpoint(run_dir: str | Path) -> Path:
    """Prefer the interrupted adapter, then the best, then the latest snapshot."""
    root = Path(run_dir)
    candidates = [root / "interrupted", root / "best", root / "final"]
    snapshots = sorted(root.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
                       reverse=True)
    for path in [*candidates, *snapshots]:
        if (path / "adapter_config.json").is_file() and any(
            (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            return path
    raise FileNotFoundError("No saved adapter yet. Run training until a checkpoint is saved, then try again.")
