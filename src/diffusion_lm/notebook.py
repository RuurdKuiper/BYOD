"""Small live dashboard used by the public training notebook."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Sequence


def _records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            pass
    return records


def _tail(path: Path, lines: int = 12) -> str:
    if not path.is_file():
        return "Waiting for the training process to start…"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _show(run_dir: Path, log_path: Path, max_updates: int, started: float) -> None:
    from IPython.display import clear_output, display
    import matplotlib.pyplot as plt

    clear_output(wait=True)
    records = _records(run_dir / "metrics.jsonl")
    train = [r for r in records if r.get("split") == "train"]
    validation = [r for r in records if r.get("split") == "validation"]
    last_step = max([int(r.get("step", 0)) for r in records] or [0])
    elapsed = (time.time() - started) / 60
    print(f"Training: {last_step:,}/{max_updates:,} updates logged · elapsed {elapsed:.1f} min")

    if train or validation:
        figure, axis = plt.subplots(figsize=(9, 4))
        if train:
            axis.plot(
                [r["step"] for r in train],
                [r["weighted_loss"] for r in train],
                alpha=.7,
                linewidth=1.3,
                label="training loss",
            )
        if validation:
            axis.plot(
                [r["step"] for r in validation],
                [r["weighted_loss"] for r in validation],
                marker="o",
                linewidth=2,
                label="validation loss",
            )
        axis.set(xlabel="update", ylabel="denoising loss", title="Live training progress")
        axis.grid(alpha=.25)
        axis.legend()
        figure.tight_layout()
        display(figure)
        plt.close(figure)
    else:
        print("Preparing the dataset and model; loss points will appear shortly.")

    print("\nLatest training messages:\n" + _tail(log_path))


def run_with_dashboard(
    command: Sequence[str],
    run_dir: str | Path,
    max_updates: int,
    *,
    refresh_seconds: float = 5.0,
) -> None:
    """Run training and refresh loss curves until the subprocess exits."""
    run_dir = Path(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(f"Choose a new run directory; this one already exists: {run_dir}")
    # Keep the log beside the run initially. The trainer must create run_dir
    # itself, otherwise its collision protection deliberately selects a suffix.
    log_path = run_dir.parent / f".{run_dir.name}.training.log"
    started = time.time()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            while process.poll() is None:
                _show(run_dir, log_path, max_updates, started)
                time.sleep(refresh_seconds)
        except KeyboardInterrupt:
            process.terminate()
            process.wait(timeout=30)
            raise
    _show(run_dir, log_path, max_updates, started)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    if run_dir.is_dir():
        log_path.replace(run_dir / "colab_training.log")
    print(f"\nFinished. Artifacts are in {run_dir}")
