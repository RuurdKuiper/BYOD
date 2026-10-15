from types import SimpleNamespace

import torch

from diffusion_lm.infilling import score_bidirectional_example, summarize_infilling


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        mapping = {
            "left": [1],
            " target": [2],
            " suffix target": [4, 2],
        }
        return mapping[text]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(token) for token in ids)


def example():
    return SimpleNamespace(
        example_id="example",
        prompt="instruction",
        metadata={
            "system_prompt": "You are a helpful assistant.",
            "answer_prefix": "left",
            "target_text": " target",
            "answer_suffix": " suffix target",
        },
    )


def test_bidirectional_score_reports_right_context_gain(monkeypatch):
    import diffusion_lm.infilling as module

    monkeypatch.setattr(module, "_prompt_ids", lambda *_args: [10])

    def forward(_session, input_ids, _padding):
        logits = torch.zeros((1, input_ids.shape[1], 8))
        # The later clue makes the model predict target ID 2; without it, it predicts 0.
        logits[0, 2, 2 if input_ids.shape[1] == 5 else 0] = 10.0
        return logits

    monkeypatch.setattr(module, "forward_denoising", forward)
    session = SimpleNamespace(
        tokenizer=Tokenizer(), prompt_format="test", mask_token_id=7,
        device=torch.device("cpu"),
    )

    result = score_bidirectional_example(session, example())

    assert result["exact_match"] is True
    assert result["without_clue_exact_match"] is False
    assert result["clue_changed_prediction"] is True
    summary = summarize_infilling([result])
    assert summary["exact_match"] == 1.0
    assert summary["without_clue_exact_match"] == 0.0
    assert summary["right_context_gain_exact_match"] == 1.0
    assert summary["clue_changed_prediction_rate"] == 1.0


def test_infilling_summary_is_token_weighted():
    results = [
        {
            "exact_match": True, "correct_tokens": 1, "target_tokens": 1,
            "without_clue_exact_match": False, "without_clue_correct_tokens": 0,
            "clue_changed_prediction": True,
        },
        {
            "exact_match": False, "correct_tokens": 1, "target_tokens": 3,
            "without_clue_exact_match": False, "without_clue_correct_tokens": 0,
            "clue_changed_prediction": False,
        },
    ]

    summary = summarize_infilling(results)

    assert summary["exact_match"] == 0.5
    assert summary["token_accuracy"] == 0.5
    assert summary["without_clue_exact_match"] == 0.0
    assert summary["right_context_gain_token_accuracy"] == 0.5
    assert summary["clue_changed_prediction_rate"] == 0.5
