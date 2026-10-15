"""Shared ordered prompts for training and benchmark generation."""
from pathlib import Path


GENERATION_PROMPTS_PATH = Path(__file__).with_name("generation_prompts.txt")


def _load_generation_prompts(path: str | Path = GENERATION_PROMPTS_PATH) -> tuple[str, ...]:
    """Load the shared, ordered generation-validation prompt set."""
    prompts = tuple(
        line
        for raw_line in Path(path).read_text().splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    )
    if not prompts:
        raise ValueError(f"Generation prompt file is empty: {path}")
    return prompts


# Resolve once so every validation checkpoint in a run uses the exact same set.
DEFAULT_GENERATION_PROMPTS = _load_generation_prompts()
