import torch
from diffusion_lm.data import DEFAULT_SYSTEM_PROMPT, DenoisingCollator, knowledge_neutral_chat_template, prepare_mask_only_cache_record, source_to_tokens
from diffusion_lm.loss import masked_denoising_loss, selected_denoising_loss


class ToyTokenizer:
    name_or_path = "toy-llama"
    eos_token_id = 2
    chat_template = "toy"
    def encode(self, text, add_special_tokens=False):
        table = {"MASK": [9], "<|start_header_id|>assistant<|end_header_id|>": [7, 8, 6], "\n": [5]}
        return table.get(text, [3 + (ord(c) % 4) for c in text])
    def decode(self, ids): return "MASK" if ids == [9] else str(ids)
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True): return [1, 4]


def test_knowledge_neutral_template_removes_dates_only():
    tokenizer = ToyTokenizer()
    tokenizer.chat_template = (
        'before\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n'
        '{{- "Today Date: " + date_string + "\\n\\n" }}\nafter'
    )
    assert knowledge_neutral_chat_template(tokenizer) == "before\nafter"


def row(index=0):
    # assistant header/newline, answer content, genuine EOS, EOS padding
    labels = [1, 7, 8, 6, 5, 3, 4, 2, 2, 2]
    stored = [1, 7, 8, 6, 9, 9, 4, 9, 9, 9]
    return {"labels": labels, "input_ids": stored, "instruction": "i", "input": "", "output": "ab", "_index": index}


def test_structured_uses_stored_inputs_and_excludes_delimiter_eos_padding():
    b = DenoisingCollator(ToyTokenizer(), "structured", 32, structured_loss_behavior="all_answer_tokens")([row()])
    assert b["input_ids"].tolist()[0] == row()["input_ids"]
    assert b["answer_mask"].tolist()[0] == [False, False, False, False, False, True, True, True, False, False]
    assert b["padding_mask"].tolist()[0] == [False] * 8 + [True, True]
    assert b["loss_mask"].equal(b["answer_mask"])


def test_structured_online_corrupts_clean_stored_inputs_deterministically():
    clean = row()
    clean["input_ids"] = list(clean["labels"])
    collator = DenoisingCollator(ToyTokenizer(), "structured", 32, seed=1, deterministic=True)
    first = collator([clean])
    second = collator([clean])
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert torch.equal(first["input_ids"][:, :5], first["labels"][:, :5])
    assert (first["input_ids"][:, 5:] != first["labels"][:, 5:]).any()


def test_structured_all_tokens_can_include_prompt_and_padding():
    b = DenoisingCollator(ToyTokenizer(), "structured", 32, structured_loss_behavior="all_tokens")([row()])
    assert b["loss_mask"].all()


def test_mask_only_starts_clean_and_never_changes_prompt_or_padding():
    b = DenoisingCollator(ToyTokenizer(), "mask_only", 32, seed=4, deterministic=True, t_min=1.0)([row()])
    # Source reconstruction, rather than the stored noised Llama IDs, is the clean input.
    assert b["labels"].tolist()[0] == [1, 4, 4, 5, 2]
    assert b["labels"].tolist()[0] != row()["input_ids"]
    changed = b["input_ids"] != b["labels"]
    assert torch.equal(changed, b["loss_mask"])
    assert not changed[0, :2].any()
    assert changed[0, 2:].all()


def test_source_tokenization_does_not_append_eos_to_builder_truncated_answer():
    complete, start = source_to_tokens(row(), ToyTokenizer())
    truncated, truncated_start = source_to_tokens(
        {**row(), "answer_truncated": True}, ToyTokenizer()
    )
    assert truncated_start == start
    assert complete[-1] == ToyTokenizer.eos_token_id
    assert truncated == complete[:-1]


def test_cached_mask_only_preparation_matches_online_tokenization_exactly():
    tokenizer = ToyTokenizer()
    features = [row(2), {**row(7), "output": "abcdef"}]
    cached = [prepare_mask_only_cache_record(feature, tokenizer, 32) for feature in features]
    online_batch = DenoisingCollator(
        tokenizer, "mask_only", 32, seed=11, deterministic=True, t_min=0.2
    )(features)
    cached_batch = DenoisingCollator(
        tokenizer, "mask_only", 32, seed=11, deterministic=True, t_min=0.2
    )(cached)

    assert online_batch.keys() == cached_batch.keys()
    for key in online_batch:
        assert torch.equal(online_batch[key], cached_batch[key]), key


def test_mask_only_all_tokens_supervises_prompt_answer_and_padding():
    # A longer peer forces EOS padding onto the short example.  At t=1,
    # all-token mask-only corruption must mask that padding too, so EOS is
    # learned as a denoising target rather than copied from the input.
    longer = {**row(1), "output": "abcdef"}
    b = DenoisingCollator(ToyTokenizer(), "mask_only", 32, structured_loss_behavior="all_tokens", seed=4, deterministic=True, t_min=1.0)([row(), longer])
    assert b["loss_mask"].all()
    assert b["loss_mask"].shape == b["labels"].shape
    changed = b["input_ids"] != b["labels"]
    assert not changed[:, :2].any()  # prompt remains clean
    assert changed.all(dim=1).tolist() == [False, False]
    assert changed[:, 2:].all()
    assert b["padding_mask"][0].any()
    assert changed[0][b["padding_mask"][0]].all()


def test_eos_padding_loss_switch_applies_to_each_loss_behavior():
    longer = {**row(1), "output": "abcdef"}
    enabled = DenoisingCollator(
        ToyTokenizer(), "mask_only", 32, structured_loss_behavior="corrupted_answer_tokens",
        eos_padding_loss=True, seed=4, deterministic=True, t_min=1.0,
    )([row(), longer])
    changed = enabled["input_ids"] != enabled["labels"]
    padding = enabled["padding_mask"]
    assert changed[0][padding[0]].all()
    assert enabled["loss_mask"][0][padding[0]].all()

    disabled = DenoisingCollator(
        ToyTokenizer(), "mask_only", 32, structured_loss_behavior="all_tokens",
        eos_padding_loss=False, seed=4, deterministic=True, t_min=1.0,
    )([row(), longer])
    padding = disabled["padding_mask"]
    assert not (disabled["input_ids"] != disabled["labels"])[0][padding[0]].any()
    assert not disabled["loss_mask"][0][padding[0]].any()


def test_deterministic_eval_and_training_resampling():
    c = DenoisingCollator(ToyTokenizer(), "mask_only", 32, seed=9, deterministic=True, t_min=.2)
    assert torch.equal(c([row(5)])["input_ids"], c([row(5)])["input_ids"])
    train = DenoisingCollator(ToyTokenizer(), "mask_only", 32, seed=9, deterministic=False, t_min=.2)
    seen = {tuple(train([row(5)])["input_ids"].flatten().tolist()) for _ in range(8)}
    assert len(seen) > 1


def test_source_retokenization_falls_back_for_systemless_chat_template():
    class SystemlessTokenizer:
        chat_template = "gemma-like"
        name_or_path = "test/gemma-like"
        eos_token_id = 99

        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **_):
            self.calls.append(messages)
            if any(message["role"] == "system" for message in messages):
                class TemplateError(Exception):
                    pass
                raise TemplateError("System role not supported")
            return [10, 11]

        def encode(self, text, **_):
            return [len(text)]

    tokenizer = SystemlessTokenizer()
    labels, answer_start = source_to_tokens(
        {"instruction": "Answer clearly", "input": "about birds", "output": "They fly."}, tokenizer
    )
    assert tokenizer.calls[0][0]["role"] == "system"
    assert tokenizer.calls[1] == [{"role": "user", "content": f"{DEFAULT_SYSTEM_PROMPT}\n\nAnswer clearly\n\nabout birds"}]
    assert labels == [10, 11, len("They fly."), 99]
    assert answer_start == 2


def test_source_retokenization_uses_per_example_system_prompt():
    class RecordingTokenizer(ToyTokenizer):
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            self.calls.append(messages)
            return super().apply_chat_template(messages, tokenize, add_generation_prompt)

    tokenizer = RecordingTokenizer()
    source_to_tokens(
        {"system": "Be precise.", "instruction": "Answer", "input": "this", "output": "Okay"},
        tokenizer,
    )
    assert tokenizer.calls[0][0] == {"role": "system", "content": "Be precise."}


def test_every_usable_example_has_a_mask_and_multitoken_mask_is_rejected():
    b = DenoisingCollator(ToyTokenizer(), "mask_only", 32, seed=1, deterministic=True, t_min=0.0)([row()])
    assert b["loss_mask"].any()
    class BadTokenizer(ToyTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [9, 10] if text == "MASK" else super().encode(text, add_special_tokens)
    try:
        DenoisingCollator(BadTokenizer(), "mask_only", 32)
    except ValueError as exc:
        assert "exactly one token" in str(exc)
    else:
        raise AssertionError("expected MASK validation failure")


def test_configured_single_token_mask_is_used():
    class ConfigurableMaskTokenizer(ToyTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text == "MASK":
                return [9, 10]
            if text == "mask":
                return [11]
            return super().encode(text, add_special_tokens)

        def decode(self, ids):
            return "mask" if ids == [11] else super().decode(ids)

    collator = DenoisingCollator(
        ConfigurableMaskTokenizer(), "mask_only", 32, mask_token="mask"
    )
    assert collator.mask_info["mask_token"] == "mask"
    assert collator.mask_info["mask_token_id"] == 11


def test_per_example_inverse_t_weighting_same_position():
    # p(correct)=e^-1 at row 0; p(correct)=e^-2 at row 1.
    logits = torch.tensor([[[0., 0.]], [[0., -2.]]])
    labels = torch.tensor([[1], [1]])
    mask = torch.tensor([[True], [True]])
    normalization = torch.tensor([[True], [True]])
    loss, metrics = masked_denoising_loss(logits, labels, mask, torch.tensor([.5, 1.]), normalization)
    ce0 = torch.logsumexp(torch.tensor([0., 0.]), 0)
    ce1 = -torch.tensor(-2.) + torch.logsumexp(torch.tensor([0., -2.]), 0)
    assert torch.allclose(loss, ((ce0 / .5) + ce1) / 2)
    assert torch.allclose(metrics["unweighted_masked_token_ce"], (ce0 + ce1) / 2)


def test_inverse_t_weighting_uses_full_response_length_not_masked_count():
    logits = torch.tensor([[[0., 0.]], [[0., -2.]]])
    labels = torch.tensor([[1], [1]])
    loss_mask = torch.tensor([[True], [True]])
    normalization = torch.tensor([
        [True, True, True, True],
        [True, True, False, False],
    ])
    sampled_t = torch.tensor([.5, 1.])
    loss, _ = masked_denoising_loss(logits, labels, loss_mask, sampled_t, normalization)
    ce0 = torch.logsumexp(torch.tensor([0., 0.]), 0)
    ce1 = -torch.tensor(-2.) + torch.logsumexp(torch.tensor([0., -2.]), 0)
    expected = ((ce0 / .5 / 4) + (ce1 / 1. / 2)) / 2
    assert torch.allclose(loss, expected)


def test_same_position_logits_not_shifted():
    labels = torch.tensor([[1, 0]])
    logits = torch.tensor([[[0., 8.], [8., 0.]]])
    mask = torch.tensor([[True, False]])
    loss, _ = masked_denoising_loss(logits, labels, mask)
    assert loss < .01


def test_selected_position_loss_matches_dense_reference_and_gradients():
    torch.manual_seed(4)
    labels = torch.randint(0, 7, (3, 5))
    mask = torch.tensor([
        [True, False, True, False, False],
        [False, False, False, False, False],
        [False, True, True, True, False],
    ])
    normalization = torch.tensor([
        [True, True, True, False, False],
        [True, True, False, False, False],
        [True, True, True, True, True],
    ])
    sampled_t = torch.tensor([0.5, 0.8, 1.0])
    optimized_logits = torch.randn(3, 5, 7, requires_grad=True)
    reference_logits = optimized_logits.detach().clone().requires_grad_(True)

    optimized, _ = masked_denoising_loss(
        optimized_logits, labels, mask, sampled_t, normalization
    )
    dense_ce = torch.nn.functional.cross_entropy(
        reference_logits.transpose(1, 2), labels, reduction="none"
    )
    counts = mask.sum(dim=1)
    valid = counts > 0
    dense_weighted = (
        (dense_ce * mask).sum(dim=1)
        / sampled_t
        / normalization.sum(dim=1).clamp_min(1)
    )
    reference = dense_weighted[valid].mean()

    optimized.backward()
    reference.backward()
    assert torch.allclose(optimized, reference)
    assert torch.allclose(optimized_logits.grad, reference_logits.grad)


def test_preselected_logit_loss_matches_masked_logit_loss():
    torch.manual_seed(8)
    logits = torch.randn(3, 4, 9)
    labels = torch.randint(0, 9, (3, 4))
    mask = torch.tensor([
        [True, False, True, False],
        [False, True, False, False],
        [True, True, True, False],
    ])
    normalization = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
        [True, True, True, True],
    ])
    sampled_t = torch.tensor([0.4, 0.7, 1.0])
    expected, expected_metrics = masked_denoising_loss(
        logits, labels, mask, sampled_t, normalization
    )
    example_ids, token_ids = mask.nonzero(as_tuple=True)
    actual, actual_metrics = selected_denoising_loss(
        logits[example_ids, token_ids],
        labels[example_ids, token_ids],
        example_ids,
        mask.sum(dim=1),
        sampled_t,
        normalization,
    )
    assert torch.allclose(actual, expected)
    for key in expected_metrics:
        assert torch.allclose(actual_metrics[key], expected_metrics[key]), key


def test_empty_loss_mask_is_differentiable_and_training_metrics_are_optional():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.zeros((2, 3), dtype=torch.long)
    mask = torch.zeros((2, 3), dtype=torch.bool)
    loss, metrics = masked_denoising_loss(
        logits, labels, mask, compute_unweighted_metric=False
    )
    loss.backward()
    assert loss == 0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0
    assert "unweighted_masked_token_ce" not in metrics


def test_dense_all_token_fast_path_matches_selected_path():
    torch.manual_seed(5)
    labels = torch.randint(0, 6, (2, 4))
    mask = torch.ones((2, 4), dtype=torch.bool)
    logits = torch.randn(2, 4, 6)
    selected, selected_metrics = masked_denoising_loss(logits, labels, mask)
    dense, dense_metrics = masked_denoising_loss(
        logits, labels, mask, sparse_positions=False
    )
    assert torch.allclose(selected, dense)
    assert torch.allclose(
        selected_metrics["unweighted_masked_token_ce"],
        dense_metrics["unweighted_masked_token_ce"],
    )
