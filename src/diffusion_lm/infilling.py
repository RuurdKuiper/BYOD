"""A minimal right-context copy test for bidirectional denoisers."""
from __future__ import annotations

from typing import Any

import torch

from .inference import _prompt_ids, forward_denoising


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Encode one deliberately boundary-aware benchmark segment."""
    return list(tokenizer.encode(text, add_special_tokens=False))


def _encoded_example(session: Any, example: Any) -> tuple[list[int], list[int], list[int]]:
    """Return chat/prefix IDs, clean target IDs, and visible suffix IDs."""
    system_prompt = str(example.metadata.get("system_prompt", "You are a helpful assistant."))
    chat = _prompt_ids(session.tokenizer, example.prompt, system_prompt, session.prompt_format)
    answer_prefix = _encode(session.tokenizer, str(example.metadata["answer_prefix"]))
    target = _encode(session.tokenizer, str(example.metadata["target_text"]))
    suffix = _encode(session.tokenizer, str(example.metadata["answer_suffix"]))
    if not target:
        raise ValueError(f"Infilling example {example.example_id} has an empty tokenized target")
    return chat + answer_prefix, target, suffix


@torch.inference_mode()
def score_bidirectional_example(session: Any, example: Any) -> dict[str, Any]:
    """Predict an earlier masked word with and without the later copy clue."""
    left, target, suffix = _encoded_example(session, example)
    target_start = len(left)
    masked_target = [int(session.mask_token_id)] * len(target)

    def predict(input_ids: list[int]) -> tuple[list[int], int]:
        tokens = torch.tensor([input_ids], dtype=torch.long, device=session.device)
        padding = torch.zeros_like(tokens, dtype=torch.bool)
        logits = forward_denoising(session, tokens, padding)[0, target_start:target_start + len(target)].float()
        predicted = logits.argmax(dim=-1).tolist()
        correct = sum(int(actual == expected) for actual, expected in zip(predicted, target))
        return predicted, correct

    predicted, correct = predict(left + masked_target + suffix)
    without_clue_predicted, without_clue_correct = predict(left + masked_target)
    return {
        "target_ids": target,
        "prediction_ids": predicted,
        "prediction": session.tokenizer.decode(predicted, skip_special_tokens=True).strip(),
        "target": str(example.metadata["target_text"]).strip(),
        "target_tokens": len(target),
        "correct_tokens": correct,
        "exact_match": predicted == target,
        "without_clue_prediction_ids": without_clue_predicted,
        "without_clue_prediction": session.tokenizer.decode(without_clue_predicted, skip_special_tokens=True).strip(),
        "without_clue_correct_tokens": without_clue_correct,
        "without_clue_exact_match": without_clue_predicted == target,
        "clue_changed_prediction": predicted != without_clue_predicted,
    }


def summarize_infilling(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Report accuracy with the later clue, without it, and their difference."""
    examples = len(results)
    tokens = sum(int(result["target_tokens"]) for result in results)
    exact_match = sum(int(result["exact_match"]) for result in results) / max(examples, 1)
    token_accuracy = sum(int(result["correct_tokens"]) for result in results) / max(tokens, 1)
    without_clue_exact = sum(int(result["without_clue_exact_match"]) for result in results) / max(examples, 1)
    without_clue_token = sum(int(result["without_clue_correct_tokens"]) for result in results) / max(tokens, 1)
    clue_changed = sum(int(result["clue_changed_prediction"]) for result in results) / max(examples, 1)
    summary = {
        "accuracy": exact_match,
        "exact_match": exact_match,
        "token_accuracy": token_accuracy,
        "without_clue_exact_match": without_clue_exact,
        "without_clue_token_accuracy": without_clue_token,
        "right_context_gain_exact_match": exact_match - without_clue_exact,
        "right_context_gain_token_accuracy": token_accuracy - without_clue_token,
        "clue_changed_prediction_rate": clue_changed,
        "correct": sum(int(result["exact_match"]) for result in results),
        "total": examples,
        "tokens": tokens,
    }
    return summary
