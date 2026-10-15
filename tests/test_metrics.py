import pytest

from diffusion_lm.metrics import distinct_n


class Tokenizer:
    all_special_ids = [99]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [int(token) for token in text.split()]


def test_distinct_n_uses_overlapping_model_token_windows_and_excludes_specials():
    tokenizer = Tokenizer()

    assert distinct_n("1 2 1 2", tokenizer, 1) == pytest.approx(0.5)
    # Sliding bigrams: (1,2), (2,1), (1,2), hence 2 / 3 distinct.
    assert distinct_n("1 2 1 2", tokenizer, 2) == pytest.approx(2 / 3)
    # The special token is removed before windows are constructed.
    assert distinct_n("1 99 1", tokenizer, 1) == pytest.approx(0.5)


def test_distinct_n_returns_zero_without_a_complete_ngram():
    assert distinct_n("1", Tokenizer(), 2) == 0.0


def test_distinct_n_rejects_non_positive_n():
    with pytest.raises(ValueError, match="at least 1"):
        distinct_n("1 2", Tokenizer(), 0)
