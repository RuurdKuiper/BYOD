"""Small, reproducible benchmark adapters for pure diffusion evaluation."""
from __future__ import annotations

import json
import math
import random
import re
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from typing import Any, Callable

import torch

from .generation_prompts import DEFAULT_GENERATION_PROMPTS
from .metrics import distinct_n


MC_TASKS = {"mmlu", "mmlu_pro", "hellaswag", "arc_c", "gpqa"}
SUBJECT_CATEGORY_TASKS = {"mmlu", "mmlu_pro"}
ALL_TASKS = ["mmlu", "mmlu_pro", "hellaswag", "arc_c", "gsm8k", "math", "gpqa", "humaneval", "mbpp", "bidirectional_infilling"]
OPEN_ENDED_TASK = "open_ended"
BIDIRECTIONAL_INFILLING_TASK = "bidirectional_infilling"
AVAILABLE_TASKS = [*ALL_TASKS, OPEN_ENDED_TASK]
BENCHMARK_SAMPLE_SEED = 1234
DIFFUSION_SAMPLERS = {"denoise_stream", "llada_official"}

# Published pure-diffusion settings for LLaDA-8B-Instruct (paper Appendix B.4
# and the official OpenCompass reproduction configs).  The paper profiles use
# one full generation block, so they contain no semi-autoregressive decoding.
LLADA_INSTRUCT_TASK_SETTINGS: dict[str, dict[str, Any]] = {
    "mmlu": {"max_new_tokens": 3, "num_steps": 3, "block_length": 3},
    "mmlu_pro": {"max_new_tokens": 256, "num_steps": 256, "block_length": 256},
    "hellaswag": {"max_new_tokens": 3, "num_steps": 3, "block_length": 3},
    "arc_c": {"max_new_tokens": 512, "num_steps": 512, "block_length": 512},
    "gsm8k": {"max_new_tokens": 512, "num_steps": 512, "block_length": 512, "confidence_eos_eot_inf": True},
    "math": {"max_new_tokens": 512, "num_steps": 512, "block_length": 512, "confidence_eos_eot_inf": True},
    "gpqa": {"max_new_tokens": 64, "num_steps": 64, "block_length": 64, "confidence_eos_eot_inf": True},
    "humaneval": {"max_new_tokens": 512, "num_steps": 512, "block_length": 512, "logits_eos_inf": True},
    "mbpp": {"max_new_tokens": 256, "num_steps": 256, "block_length": 256, "confidence_eos_eot_inf": True},
}


def _path_slug(value: str) -> str:
    """Turn a model/task label into a stable, filesystem-safe component."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-.").lower()
    return slug or "unnamed"


class BenchmarkRunReporter:
    """Write one benchmark invocation into an isolated, structured directory."""

    schema_version = 1

    def __init__(self, results_dir: str | Path, config: dict[str, Any], run_name: str | None = None):
        self.started_at = datetime.now(timezone.utc)
        timestamp = self.started_at.strftime("%Y%m%dT%H%M%S.%fZ")
        self.run_id = timestamp + (f"--{_path_slug(run_name)}" if run_name else "")
        self.path = Path(results_dir) / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.summaries: list[dict[str, Any]] = []
        self._write_manifest("running")

    def _write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")

    def _write_manifest(self, status: str, completed_at: str | None = None) -> None:
        manifest = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": status,
            "started_at": self.started_at.isoformat(),
            "completed_at": completed_at,
            "config": self.config,
        }
        self._write_json(self.path / "run.json", manifest)

    def group_path(self, model: str, task: str, method: str) -> Path:
        """Return the directory for one model/task/method result group."""
        return self.path / "models" / _path_slug(model) / _path_slug(task) / _path_slug(method)

    def save_result(self, result: dict[str, Any]) -> None:
        """Append one example only to its model/task/method result file."""
        path = self.group_path(result["model"], result["task"], result["method"]) / "results.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_result(path, result)

    def save_summary(self, summary: dict[str, Any]) -> None:
        """Save a group summary and retain it for run/model rollups."""
        self.summaries.append(summary)
        path = self.group_path(summary["model"], summary["task"], summary["method"]) / "summary.json"
        self._write_json(path, summary)

    def save_run_json(self, filename: str, value: Any) -> None:
        """Save a structured artifact at the root of this benchmark run."""
        self._write_json(self.path / filename, value)

    def save_run_records(self, filename: str, records: list[dict[str, Any]]) -> None:
        """Save newline-delimited records at the root of this benchmark run."""
        path = self.path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def complete(self) -> Path:
        """Write model and run rollups, then mark the invocation complete."""
        by_model: dict[str, list[dict[str, Any]]] = {}
        for summary in self.summaries:
            by_model.setdefault(str(summary["model"]), []).append(summary)
        models = []
        for model, summaries in by_model.items():
            model_summary = {"model": model, "results": summaries}
            models.append(model_summary)
            self._write_json(self.path / "models" / _path_slug(model) / "summary.json", model_summary)
        completed_at = datetime.now(timezone.utc).isoformat()
        self._write_json(self.path / "summary.json", {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": completed_at,
            "models": models,
        })
        self._write_manifest("completed", completed_at)
        return self.path


def resolve_generation_settings(config: dict[str, Any], task: str, mode: str) -> dict[str, Any]:
    """Resolve generation settings for a task and corruption mode."""
    settings = dict(config.get("generation", {}))
    mode_settings = config.get("generation_by_corruption", {}).get(mode, {})
    if mode == "legacy" and not mode_settings:
        mode_settings = config.get("generation_by_corruption", {}).get("structured", {})
    settings.update(mode_settings)
    settings.update(config.get("task_generation", {}).get(task, {}))
    settings.update(config.get("task_generation_by_corruption", {}).get(mode, {}).get(task, {}))
    if mode == "mask_only":
        # Mask-only training is evaluated with the full-remasking setup used
        # by the training-time generation validation. Retention behavior stays
        # explicitly configurable when the denoise-stream sampler is selected.
        settings["noise_level"] = 1.0
        settings.setdefault("permanent_unmask", True)
        settings.setdefault("confidence_guided", True)
    if "diffusion_sampler" in config:
        settings["sampler"] = config["diffusion_sampler"]
    if "sampler" in settings:
        sampler = str(settings["sampler"])
        if sampler not in DIFFUSION_SAMPLERS:
            raise ValueError(
                f"diffusion sampler must be one of {sorted(DIFFUSION_SAMPLERS)}; received {sampler!r}"
            )
        settings["sampler"] = sampler
    return settings


def resolve_autoregressive_generation_settings(config: dict[str, Any], task: str) -> dict[str, Any]:
    """Resolve settings for the independent autoregressive baseline."""
    settings = dict(config.get("autoregressive_generation", {}))
    settings.update(config.get("autoregressive_task_generation", {}).get(task, {}))
    settings.setdefault("max_new_tokens", 256)
    return settings


def resolve_llada_generation_settings(config: dict[str, Any], task: str) -> dict[str, Any]:
    """Resolve selectable denoise-stream or official decoding for hosted LLaDA."""
    settings = resolve_generation_settings(config, task, "mask_only")
    profile = LLADA_INSTRUCT_TASK_SETTINGS.get(task, {})
    settings.update(profile)
    family_settings = dict(config.get("llada_generation", {}))
    task_settings = dict(config.get("llada_task_generation", {}).get(task, {}))
    sampler_settings = dict(settings)
    sampler_settings.update(family_settings)
    sampler_settings.update(task_settings)
    sampler = str(config.get("diffusion_sampler", sampler_settings.get("sampler", "llada_official")))
    if sampler not in DIFFUSION_SAMPLERS:
        raise ValueError(
            f"diffusion sampler must be one of {sorted(DIFFUSION_SAMPLERS)}; received {sampler!r}"
        )
    official_defaults = {
        "sampler": "llada_official",
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
        "logits_eos_inf": bool(profile.get("logits_eos_inf", False)),
        "confidence_eos_eot_inf": bool(profile.get("confidence_eos_eot_inf", False)),
        "eot_token_id": 126348,
        "proportional_unmask": False,
    }
    if sampler == "llada_official":
        # Official defaults supersede generic denoise-stream settings, while
        # explicit family/task overrides retain their existing precedence.
        settings.update(official_defaults)
        settings.update(family_settings)
        settings.update(task_settings)
        for unused in (
            "noise_level", "top_k", "permanent_unmask", "confidence_guided",
            "early_stopping", "freeze_retained_tokens",
        ):
            settings.pop(unused, None)
        settings["proportional_unmask"] = False
    else:
        # Family-wide llada_generation contains official-only controls. Shared
        # denoise controls come from generation/generation_by_corruption, while
        # task overrides remain useful to both samplers.
        settings.update(task_settings)
    settings["sampler"] = sampler
    settings["block_length"] = int(settings.get("block_length", settings.get("max_new_tokens", 128)))
    return settings


def resolve_mask_only_generation_settings(config: dict[str, Any], task: str) -> dict[str, Any]:
    """Resolve selectable denoise-stream or official decoding for a mask-only adapter."""
    settings = resolve_generation_settings(config, task, "mask_only")
    profile = LLADA_INSTRUCT_TASK_SETTINGS.get(task, {})
    settings.update(profile)
    family_settings = dict(config.get("mask_only_generation", {}))
    task_settings = dict(config.get("mask_only_task_generation", {}).get(task, {}))
    sampler_settings = dict(settings)
    sampler_settings.update(family_settings)
    sampler_settings.update(task_settings)
    sampler = str(config.get("diffusion_sampler", sampler_settings.get("sampler", "llada_official")))
    if sampler not in DIFFUSION_SAMPLERS:
        raise ValueError(
            f"diffusion sampler must be one of {sorted(DIFFUSION_SAMPLERS)}; received {sampler!r}"
        )
    official_defaults = {
        "sampler": "llada_official",
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
        "logits_eos_inf": bool(profile.get("logits_eos_inf", False)),
        "confidence_eos_eot_inf": bool(profile.get("confidence_eos_eot_inf", False)),
        "proportional_unmask": False,
    }
    if sampler == "llada_official":
        settings.update(official_defaults)
        settings.update(family_settings)
        settings.update(task_settings)
        for unused in (
            "noise_level", "top_k", "permanent_unmask", "confidence_guided",
            "early_stopping", "freeze_retained_tokens",
        ):
            settings.pop(unused, None)
        settings["proportional_unmask"] = False
    else:
        settings.update(task_settings)
    settings["sampler"] = sampler
    settings["block_length"] = int(settings.get("block_length", settings.get("max_new_tokens", 128)))
    return settings

# Fixed prompts make comparisons between runs reproducible.  `limit` can be
# used to evaluate a smaller prefix, while the default benchmark config uses
# all 30 questions.
OPEN_ENDED_PROMPTS = list(DEFAULT_GENERATION_PROMPTS)


@dataclass
class BenchmarkExample:
    """Normalized benchmark item consumed by both diffusion and AR evaluators."""
    task: str
    example_id: str
    prompt: str
    answer: str
    kind: str
    metadata: dict[str, Any]


def _choice_prompt(name: str, question: str, choices: list[Any], category: str | None = None) -> str:
    """Format multiple choice and explicitly request an extractable answer label."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "\n".join(f"{letters[i]}: {choice}" for i, choice in enumerate(choices))
    if name in {"mmlu_pro", "gpqa"}:
        answer_format = (
            "Think through the problem concisely, then end with exactly one final line containing "
            "`ANSWER:` followed by the correct option label. Put no option text or punctuation "
            "after the label on that line."
        )
    elif name == "mmlu":
        answer_format = "Start your response with the correct option label followed by a colon."
    else:
        answer_format = "Start your response with the correct option label followed by a colon, for example `A:`."
    if name == "hellaswag":
        instruction = f"Choose the option that most plausibly continues the described event. {answer_format}"
        task_input = f"Beginning of the event:\n{question.strip()}\n\nWhat most plausibly happens next?\n{options}"
    else:
        instruction = f"Answer the following multiple-choice question. {answer_format}"
        task_input = f"{question.strip()}\n\n{options}"
    if category and str(category).strip():
        category_label = str(category).strip().replace("_", " ")
        instruction += f" Subject category: {category_label}."
    return f"{instruction}\n\n{task_input}"


def _multiple_choice_fields(name: str, row: dict[str, Any], index: int) -> tuple[str, list[Any], str]:
    """Normalize task-specific question, choice, and answer schemas."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if name == "hellaswag":
        question = row.get("ctx", "")
        choices = row.get("endings")
        answer = row.get("label")
    elif name == "arc_c":
        question = row.get("question", "")
        choice_group = row.get("choices") or {}
        choices = choice_group.get("text") if isinstance(choice_group, dict) else choice_group
        labels = [str(label) for label in choice_group.get("label", [])] if isinstance(choice_group, dict) else []
        answer_key = str(row.get("answerKey", ""))
        answer = letters[labels.index(answer_key)] if answer_key in labels else answer_key
    elif name == "gpqa" and not row.get("choices") and not row.get("options"):
        question = row.get("Question", row.get("question", ""))
        choices = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        correct = choices[0]
        # Make option order stable for a question even when evaluating a
        # different subset, whose local enumeration indices may change.
        random.Random(f"gpqa:{question}").shuffle(choices)
        answer = letters[choices.index(correct)]
    else:
        question = row.get("question", row.get("Question", row.get("query", row.get("ctx", ""))))
        choices = row.get("choices", row.get("options"))
        answer = row.get("answer", row.get("answerKey", row.get("label", row.get("answer_index"))))
    if not isinstance(choices, (list, tuple)) or not choices:
        raise ValueError(f"{name} example {index} has no usable answer choices")
    if isinstance(answer, int) or str(answer).isdigit():
        answer_index = int(answer)
        if not 0 <= answer_index < len(choices):
            raise ValueError(f"{name} example {index} has out-of-range answer index {answer_index}")
        answer = letters[answer_index]
    return str(question), list(choices), str(answer).upper()


def _boxed(text: str) -> str:
    """Extract the last boxed/math answer, including nested LaTeX braces."""
    text = text or ""
    openings = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    for opening in reversed(openings):
        start = opening.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index].strip()
    hashes = re.findall(r"####\s*([^\n]+)", text)
    return hashes[-1].strip() if hashes else text.strip()


def _last_number(text: str) -> str:
    """Extract the final numeric candidate, following common GSM8K evaluation."""
    candidates = re.findall(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text or "")
    return candidates[-1].replace(",", "").rstrip(".") if candidates else ""


def _normalize_math_answer(text: str) -> str:
    """Normalize a generated or reference final MATH answer for comparison."""
    has_box = re.search(r"\\(?:boxed|fbox)\s*\{", text) is not None
    value = _boxed(text).strip()
    if not has_box:
        # Accept only a terminal inline expression as a fallback. This recovers
        # answers such as "Therefore ... $(3, \\frac{\\pi}{2}).$" without
        # accidentally selecting an intermediate expression from a rationale.
        terminal_math = re.search(r"\$([^$\n]+)\$\s*[.!]?\s*\Z", value)
        if terminal_math:
            value = terminal_math.group(1).strip()
    answer_match = re.search(r"(?is)(?:final\s+answer|answer)\s*(?:is|:)\s*(.+)$", value)
    if answer_match:
        value = answer_match.group(1).strip()
    value = re.sub(r"^\$|\$$", "", value.strip())
    value = value.rstrip(".。;,!").strip()
    value = value.replace("\\left", "").replace("\\right", "")
    # Repair duplicated command escapes occasionally emitted by diffusion
    # decoding, while retaining legitimate LaTeX row separators such as `\\`.
    value = re.sub(r"\\\\(?=[A-Za-z])", r"\\", value)
    value = re.sub(r"\s+", "", value)
    # Remove commas only inside conventional thousands-grouped numerals. A
    # blanket removal corrupts tuples, coordinate pairs, intervals, and sets.
    value = re.sub(
        r"(?<![\d,])([+-]?\d{1,3}(?:,\d{3})+)(?![\d,])",
        lambda match: match.group(1).replace(",", ""),
        value,
    )
    # Normalize common answer-only presentation variants without attempting
    # broad unit conversion. Redundant grouping braces and degree notation do
    # not change the mathematical value of these terminal answers.
    while value.startswith("{") and value.endswith("}"):
        depth = 0
        encloses_all = True
        for index, character in enumerate(value):
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0 and index != len(value) - 1:
                    encloses_all = False
                    break
        if not encloses_all or depth != 0:
            break
        value = value[1:-1]
    value = re.sub(r"(?:\^\{?\\circ\}?|\\circ|°|degrees?)\Z", "", value, flags=re.IGNORECASE)
    return value


def _numeric_answers_equal(left: str, right: str) -> bool:
    """Compare normalized decimal answers exactly when both are numeric."""
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return False


def _math_answers_equal(prediction: str, target: str) -> bool:
    """Use symbolic MATH verification when installed, with a strict fallback."""
    normalized_prediction = _normalize_math_answer(prediction)
    normalized_target = _normalize_math_answer(target)
    if normalized_prediction == normalized_target or _numeric_answers_equal(normalized_prediction, normalized_target):
        return True
    try:
        from math_verify import parse, verify

        return bool(verify(parse(target), parse(prediction)))
    except (ImportError, TypeError, ValueError):
        return False


def _sample_indices(size: int, limit: int | None = None, limit_fraction: float | None = None, shuffle: bool = False) -> list[int]:
    """Select a prefix/fraction, optionally shuffling grouped datasets first."""
    if limit is not None and limit_fraction is not None:
        raise ValueError("Set either limit or limit_fraction, not both")
    if limit_fraction is not None:
        fraction = float(limit_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("limit_fraction must be greater than 0 and at most 1")
        count = min(size, max(1, math.ceil(size * fraction))) if size else 0
    elif limit is not None:
        count = int(limit)
        if count < 1:
            raise ValueError("limit must be positive")
        count = min(count, size)
    else:
        count = size
    if count >= size:
        return list(range(size))
    if shuffle:
        indices = list(range(size))
        random.Random(BENCHMARK_SAMPLE_SEED).shuffle(indices)
        return indices[:count]
    if limit_fraction is not None:
        return [(index * size) // count for index in range(count)]
    return list(range(count))


def _benchmark_spec(name: str, split: str) -> tuple[str, str | None, str]:
    """Resolve the dataset configuration and locally scoreable task split."""
    specs = {
        "mmlu": ("cais/mmlu", "all", split),
        "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, split),
        # HellaSwag's public test labels are withheld, so validation is the
        # standard locally-scoreable evaluation split.
        "hellaswag": ("Rowan/hellaswag", None, "validation" if split == "test" else split),
        "arc_c": ("allenai/ai2_arc", "ARC-Challenge", "test" if split == "test" else split),
        "gsm8k": ("openai/gsm8k", "main", split),
        "math": ("HuggingFaceH4/MATH-500", None, "test" if split == "test" else split),
        # The Hugging Face GPQA release exposes its 448 benchmark examples as
        # `train`; they are the evaluation set, not model-training data here.
        "gpqa": ("Idavidrein/gpqa", "gpqa_main", "train"),
        "humaneval": ("openai/openai_humaneval", None, split),
        "mbpp": ("google-research-datasets/mbpp", "sanitized", split),
    }
    if name not in specs:
        raise ValueError(f"Unknown benchmark {name}; available: {AVAILABLE_TASKS}")
    return specs[name]


def _mbpp_prompt(row: dict[str, Any]) -> str:
    """Build a test-informed MBPP prompt that emphasizes exact semantics."""
    description = str(row.get("text") or row.get("prompt") or "").strip()
    test_imports = [str(statement) for statement in (row.get("test_imports") or [])]
    tests = [str(test) for test in (row.get("test_list") or [])]
    sections = [description]
    if test_imports or tests:
        test_block = "\n".join(test_imports + tests)
        sections.append(
            "Your function must use the name and interface demonstrated by these tests:\n"
            f"```python\n{test_block}\n```"
        )
    sections.append(
        "Carefully infer the exact required behavior from the description and every assertion. "
        "Pay particular attention to the exact function name and number of positional arguments; "
        "words such as remove/keep, first/last/all, and ascending/descending; and the direction of "
        "arithmetic relationships. Silently check the implementation against every shown assertion "
        "before answering.\n\n"
        "Return exactly one complete Markdown code block tagged `python`. Do not write any text "
        "outside that block."
    )
    return "\n\n".join(section for section in sections if section)


def _humaneval_prompt(prompt: str) -> str:
    """Wrap canonical HumanEval source for instruction-tuned chat models."""
    return (
        "Implement the Python function described below. Preserve the exact function name, signature, "
        "and return type. Carefully follow the entire docstring, including edge cases and examples. "
        "Silently trace the implementation against every shown example before answering.\n\n"
        "Return exactly one complete Markdown code block tagged `python`, containing the complete "
        "function and any required imports. Do not write any text outside that block.\n\n"
        "Function specification:\n\n"
        + prompt.strip()
    )


def _math_prompt(problem: str) -> str:
    """Request checked, concise reasoning followed by an exact answer marker."""
    return (
        "Solve the following mathematics problem step by step. Keep the reasoning concise. "
        "Check every arithmetic and algebraic step, and verify that the final result satisfies "
        "all conditions in the problem. Simplify fractions, radicals, and expressions completely.\n\n"
        "End with exactly one final line in this format:\n\n"
        "FINAL: \\boxed{answer}\n\n"
        "Put only the answer inside the box. Do not omit the final line.\n\n"
        "Problem:\n\n" + problem.strip()
    )


def _gsm8k_prompt(question: str) -> str:
    """Request GSM8K reasoning followed by its canonical numeric answer marker."""
    return (
        "Solve the following math problem step by step. End your response with a final line in the "
        "form `#### number`, containing only the final numeric answer after `####`.\n\n"
        + question.strip()
    )


def _bidirectional_infilling_examples() -> list[BenchmarkExample]:
    """Build a minimal test of whether a masked word can use a later clue."""
    # Every target, including its leading-space boundary, is one token in the
    # Gemma 2, Llama 3.1, Qwen2.5, and Ministral tokenizers used in this study.
    words = (
        "apple", "bridge", "candle", "banana", "engine",
        "forest", "garden", "hammer", "island", "jacket",
        "chair", "lemon", "mirror", "needle", "orange",
        "piano", "quartz", "rocket", "silver", "turtle",
    )
    instruction = (
        'Replace the masked token in the partial assistant response with exactly the word written '
        'after "COPY:". Copy the spelling exactly. Do not use a synonym or a related word.'
    )
    return [
        BenchmarkExample(
            BIDIRECTIONAL_INFILLING_TASK,
            f"copy-{index:03d}",
            instruction,
            word,
            "bidirectional_infilling",
            {
                "system_prompt": "You are a helpful assistant.",
                "answer_prefix": "Answer:",
                "target_text": f" {word}",
                "answer_suffix": f". COPY: {word}.",
            },
        )
        for index, word in enumerate(words)
    ]


def load_benchmark(name: str, split: str, limit: int | None, cache_dir: str, token: str | None, limit_fraction: float | None = None) -> list[BenchmarkExample]:
    """Download one configured benchmark split and normalize its records."""
    if name == OPEN_ENDED_TASK:
        indices = _sample_indices(len(OPEN_ENDED_PROMPTS), limit, limit_fraction)
        return [BenchmarkExample(name, str(index), OPEN_ENDED_PROMPTS[index], "", "open_ended", {}) for index in indices]
    if name == BIDIRECTIONAL_INFILLING_TASK:
        examples = _bidirectional_infilling_examples()
        indices = _sample_indices(len(examples), limit, limit_fraction)
        return [examples[index] for index in indices]
    from datasets import load_dataset
    path, config, actual_split = _benchmark_spec(name, split)
    dataset = load_dataset(path, config, split=actual_split, cache_dir=cache_dir, token=token)
    indices = _sample_indices(len(dataset), limit, limit_fraction, shuffle=name in SUBJECT_CATEGORY_TASKS)
    if len(indices) != len(dataset):
        dataset = dataset.select(indices)
    items = []
    for index, row in enumerate(dataset):
        if name in MC_TASKS:
            question, choices, answer = _multiple_choice_fields(name, row, index)
            answer_index = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(answer)
            target = f"{answer}: {choices[answer_index]}"
            category = row.get("subject", row.get("category")) if name in SUBJECT_CATEGORY_TASKS else None
            items.append(BenchmarkExample(name, str(index), _choice_prompt(name, question, choices, category), target, "multiple_choice", row))
        elif name == "gsm8k":
            prompt = _gsm8k_prompt(row["question"])
            items.append(BenchmarkExample(name, str(index), prompt, row["answer"].strip(), "gsm8k", row))
        elif name == "math":
            problem = row.get("problem", row.get("question", ""))
            solution = row.get("solution", row.get("answer", ""))
            prompt = _math_prompt(problem)
            items.append(BenchmarkExample(name, str(index), prompt, solution.strip(), "math", row))
        elif name == "humaneval":
            items.append(BenchmarkExample(name, str(index), _humaneval_prompt(row["prompt"]), row.get("canonical_solution", ""), "code", row))
        elif name == "mbpp":
            items.append(BenchmarkExample(name, str(index), _mbpp_prompt(row), row.get("code", ""), "code", row))
    return items


def _declared_option_text_matches(text: str, reference: str) -> bool:
    """Match a terminal textual ANS/ANSWER against one labelled reference option."""
    declarations = re.findall(r"(?im)^\s*(?:ANS|ANSWER)\s*:\s*(.*?)\s*$", text or "")
    reference_match = re.match(r"^\s*[A-Z]\s*:\s*(.+?)\s*$", reference or "", flags=re.DOTALL)
    if not declarations or not reference_match:
        return False

    def normalize(value: str) -> str:
        value = re.sub(r"\s+", " ", value.strip()).casefold()
        return value.rstrip(" .。;,:!?")

    return normalize(declarations[-1]) == normalize(reference_match.group(1))


def extract_answer(text: str, kind: str, reference: str | None = None) -> str:
    """Extract a comparable answer from free-form model output."""
    if kind == "multiple_choice":
        # Prefer the requested leading `A: ...` format. If a model ignores that
        # instruction, accept only an explicit answer declaration rather than
        # searching for an arbitrary capital letter later in its explanation.
        match = re.match(r"\s*([A-Z])(?=\s*(?::|[.)-]|$))", text.upper())
        if match:
            return match.group(1)
        answer_line = re.search(r"(?im)^\s*ANSWER\s*:\s*[*_`(\[]*([A-Z])(?=\s*(?::|[.)\]`*_]|$))", text)
        if answer_line:
            return answer_line.group(1).upper()
        declared = re.search(
            r"\b(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s+(?:IS|WOULD\s+BE)\s+"
            r"(?:OPTION\s+)?[*_`(\[]*([A-Z])(?=\s*(?::|[.)\]-]|$))",
            text.upper(),
        )
        if declared:
            return declared.group(1)
        option = re.search(
            r"\b(?:CHOOSE|SELECT)\s+(?:OPTION\s+)?[*_`(\[]*([A-Z])"
            r"(?=\s*(?::|[.)\]-]|$))",
            text.upper(),
        )
        if option:
            return option.group(1)
        if reference and _declared_option_text_matches(text, reference):
            return extract_answer(reference, kind)
        return ""
    if kind == "gsm8k":
        return _last_number(_boxed(text))
    if kind == "math":
        return _normalize_math_answer(text)
    return text.strip()


def _extract_python_code(candidate: str, entry_point: str | None = None) -> str:
    """Extract a Python block while preserving body-completion indentation."""
    fenced = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", candidate, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        if entry_point:
            definition = re.compile(rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(")
            matching = next((block for block in fenced if definition.search(block)), None)
            candidate = matching if matching is not None else fenced[0]
        else:
            candidate = fenced[0]
    else:
        # Remove a standalone final closing fence before looking for an
        # unterminated opening fence; otherwise the closing fence itself would
        # be mistaken for the opening and all preceding Python would be lost.
        candidate = re.sub(r"\n?[ \t]*```[ \t]*\Z", "", candidate)
        # Also handle an unterminated Markdown fence, which is common when a
        # fixed generation budget cuts off just after otherwise valid code.
        opening = re.search(r"```(?:python|py)?\s*\n?", candidate, flags=re.IGNORECASE)
        if opening:
            candidate = candidate[opening.end():]
        elif entry_point:
            # If prose precedes a complete function, discard only that prose.
            definition = re.search(rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", candidate)
            if definition:
                candidate = candidate[definition.start():]
    return candidate.strip("\n")


def _run_code(candidate: str, example: BenchmarkExample, timeout: float = 10.0) -> bool:
    """Execute one generated code answer with its benchmark tests in a timeout."""
    metadata = example.metadata
    if example.task == "humaneval":
        entry_point = str(metadata["entry_point"])
        candidate = _extract_python_code(candidate, entry_point)
        full_function = re.search(
            rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", candidate
        )
        if full_function:
            solution = candidate
        else:
            # HumanEval's canonical answer is a function-body completion. Join
            # it to the benchmark prompt exactly as the reference harness does.
            completion = candidate
            first_line = next((line for line in completion.splitlines() if line.strip()), "")
            if first_line and not first_line[:1].isspace():
                completion = "\n".join(f"    {line}" if line else line for line in completion.splitlines())
            prompt = str(metadata["prompt"])
            solution = prompt + ("" if prompt.endswith("\n") else "\n") + completion.lstrip("\n")
        program = solution + "\n\n" + metadata["test"] + f"\ncheck({entry_point})\n"
    else:
        candidate = _extract_python_code(candidate)
        tests = metadata.get("test_list", [])
        setup_parts = metadata.get("test_imports", []) or []
        legacy_setup = metadata.get("test_setup_code", "")
        if legacy_setup:
            setup_parts = [*setup_parts, legacy_setup]
        setup = "\n".join(str(statement) for statement in setup_parts)
        program = setup + "\n" + candidate + "\n" + "\n".join(tests)
    with tempfile.TemporaryDirectory(prefix="diffusion-lm-eval-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(program)
        try:
            result = subprocess.run([sys.executable, "-I", str(path)], capture_output=True, timeout=timeout, cwd=directory)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False


def score_prediction(example: BenchmarkExample, generated: str) -> bool:
    """Score one normalized prediction with exact-match or benchmark tests."""
    if example.kind == "multiple_choice":
        return extract_answer(generated, example.kind, example.answer) == extract_answer(example.answer, example.kind)
    if example.kind == "gsm8k":
        prediction = extract_answer(generated, example.kind)
        target = extract_answer(example.answer, example.kind)
        return prediction == target or _numeric_answers_equal(prediction, target)
    if example.kind == "math":
        return _math_answers_equal(generated, example.answer)
    return _run_code(generated, example)


def save_result(path: Path, result: dict[str, Any]) -> None:
    """Append one per-example benchmark result as JSONL."""
    with path.open("a") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")


@torch.no_grad()
def score_texts_with_model(model: Any, tokenizer: Any, device: torch.device, texts: list[str]) -> dict[str, Any]:
    """Score texts with one fixed causal reference model.

    This deliberately does not disable adapters or restore normalization
    parameters: the supplied model is the shared perplexity reference model.
    """
    import torch.nn.functional as F

    total_nll = 0.0
    total_tokens = 0
    per_text = []
    model.eval()
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            perplexity = None
        else:
            outputs = model(input_ids=input_ids, use_cache=False)
            labels = input_ids[:, 1:]
            logits = outputs.logits[:, :-1].float()
            nll = F.cross_entropy(logits.transpose(1, 2), labels, reduction="sum")
            text_nll = float(nll.cpu())
            text_tokens = int(labels.numel())
            total_nll += text_nll
            total_tokens += text_tokens
            perplexity = float(torch.exp(torch.tensor(text_nll / text_tokens)))
        per_text.append({
            "perplexity": perplexity,
        })
    mean_nll = total_nll / max(total_tokens, 1)
    valid_perplexities = [item["perplexity"] for item in per_text if item["perplexity"] is not None]
    return {
        "perplexity": float(torch.exp(torch.tensor(mean_nll))),
        "mean_perplexity": float(sum(valid_perplexities) / len(valid_perplexities)) if valid_perplexities else None,
        "median_perplexity": float(median(valid_perplexities)) if valid_perplexities else None,
        "mean_nll": mean_nll,
        "tokens": total_tokens,
        "per_text": per_text,
    }


@torch.no_grad()
def score_open_ended_generations(session: Any, texts: list[str]) -> dict[str, Any]:
    """Score generated texts with base-model perplexity and Distinct-n metrics.

    Perplexity is measured with adapters disabled and the saved initial
    normalization weights restored, matching training-time generation
    perplexity.  The aggregate perplexity is token-weighted; each text also
    receives its own perplexity in ``per_text``.
    """
    import torch.nn.functional as F

    model = session.model
    tokenizer = session.tokenizer
    trained_norms = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if "norm" in name.lower()}
    initial_path = Path(session.adapter_path) / "normalization_initial_state.pt"
    initial_norms = torch.load(initial_path, map_location="cpu", weights_only=True) if initial_path.is_file() else trained_norms
    total_nll = 0.0
    total_tokens = 0
    per_text = []
    try:
        named = dict(model.named_parameters())
        for name, value in initial_norms.items():
            if name in named:
                named[name].data.copy_(value.to(named[name].device, dtype=named[name].dtype))
        adapter_context = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
        with adapter_context:
            for text in texts:
                encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
                input_ids = encoded["input_ids"].to(session.device)
                if input_ids.shape[1] < 2:
                    perplexity = None
                else:
                    outputs = model(input_ids=input_ids) if getattr(session, "llada", False) else model(input_ids=input_ids, use_cache=False)
                    labels = input_ids[:, 1:]
                    logits = outputs.logits[:, :-1].float()
                    nll = F.cross_entropy(logits.transpose(1, 2), labels, reduction="sum")
                    text_nll = float(nll.cpu())
                    text_tokens = int(labels.numel())
                    total_nll += text_nll
                    total_tokens += text_tokens
                    perplexity = float(torch.exp(torch.tensor(text_nll / text_tokens)))
                per_text.append({
                    "perplexity": perplexity,
                    "distinct_1": distinct_n(text, tokenizer, 1),
                    "distinct_2": distinct_n(text, tokenizer, 2),
                    "distinct_3": distinct_n(text, tokenizer, 3),
                })
    finally:
        named = dict(model.named_parameters())
        for name, value in trained_norms.items():
            if name in named:
                named[name].data.copy_(value.to(named[name].device, dtype=named[name].dtype))
    mean_nll = total_nll / max(total_tokens, 1)
    valid_perplexities = [item["perplexity"] for item in per_text if item["perplexity"] is not None]
    return {
        "perplexity": float(torch.exp(torch.tensor(mean_nll))),
        "mean_perplexity": float(sum(valid_perplexities) / len(valid_perplexities)) if valid_perplexities else None,
        "median_perplexity": float(median(valid_perplexities)) if valid_perplexities else None,
        "mean_nll": mean_nll,
        "tokens": total_tokens,
        "per_text": per_text,
    }
