import json

import pytest
import torch
import torch.nn.functional as F

from diffusion_lm.loss import masked_denoising_loss, selected_denoising_loss
from diffusion_lm.training import _resolve_answer_padding_weights


def objective(logits, labels, selected, answer, padding, t, corrections=None):
    return masked_denoising_loss(
        logits, labels, selected, t, answer | padding,
        answer_padding_weights=(0.9, 0.1), answer_mask=answer, padding_mask=padding,
        token_weights=corrections,
    )


def test_padding_repetition_does_not_dilute_answer_loss_or_gradient():
    results = []
    for padding_length in (1, 200):
        # Two content tokens and their terminating EOS, then repeated EOS.
        logits = torch.tensor([[[0., 2.], [1., 0.], [0., 1.]] + [[1., 0.]] * padding_length], requires_grad=True)
        labels = torch.tensor([[1, 1, 0] + [0] * padding_length])
        answer = torch.tensor([[True] * 3 + [False] * padding_length])
        padding = ~answer
        loss, metrics = objective(logits, labels, answer | padding, answer, padding, torch.ones(1))
        loss.backward()
        ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        assert torch.allclose(loss, 0.9 * ce[:, :3].mean() + 0.1 * ce[:, 3:].mean())
        assert torch.allclose(metrics["answer_loss"], ce[:, :3].mean())
        results.append((loss.detach(), logits.grad[:, :3], logits.grad[:, 3:].sum(dim=1)))
    for small, large in zip(*results):
        assert torch.allclose(small, large)


def test_separate_loss_uses_eligible_lengths_and_actual_frontier_probabilities():
    torch.manual_seed(14)
    logits = torch.randn(2, 6, 4, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, 3, 3, 3], [0, 2, 1, 1, 3, 3]])
    answer = torch.tensor([[False, True, True, True, False, False], [False, True, True, True, True, False]])
    padding = torch.tensor([[False, False, False, False, True, True], [False, False, False, False, False, True]])
    selected = torch.tensor([[False, True, False, True, True, False], [False, True, False, True, False, True]])
    t = torch.tensor([0.4, 0.5])
    # First row has frontier probabilities on answer tokens; second is IID.
    probabilities = torch.tensor([[1., .03, .3, .9, .4, .4], [1., .5, .5, .5, .5, .5]])
    corrections = t[:, None] / probabilities
    ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    expected_answer = ((ce * selected * answer / probabilities).sum(dim=1) / answer.sum(dim=1)).mean()
    expected_padding = ((ce * selected * padding / probabilities).sum(dim=1) / padding.sum(dim=1)).mean()
    expected = .9 * expected_answer + .1 * expected_padding
    expected_grad, = torch.autograd.grad(expected, logits, retain_graph=True)

    full, metrics = objective(logits, labels, selected, answer, padding, t, corrections)
    ids, positions = selected.nonzero(as_tuple=True)
    compact, compact_metrics = selected_denoising_loss(
        logits[ids, positions], labels[ids, positions], ids, selected.sum(dim=1), t,
        token_weights=corrections[ids, positions], answer_padding_weights=(.9, .1),
        selected_answer_mask=answer[ids, positions], selected_padding_mask=padding[ids, positions],
        answer_lengths=answer.sum(dim=1), padding_lengths=padding.sum(dim=1),
    )
    for loss, result in ((full, metrics), (compact, compact_metrics)):
        assert torch.allclose(loss, expected)
        assert torch.allclose(result["answer_loss"], expected_answer)
        assert torch.allclose(result["padding_loss"], expected_padding)
        assert torch.allclose(result["unweighted_masked_token_ce"], ce[selected].mean())
        grad, = torch.autograd.grad(loss, logits, retain_graph=True)
        assert torch.allclose(grad, expected_grad)
        assert torch.equal(grad[:, 0], torch.zeros_like(grad[:, 0]))  # Prompt has no loss.


def test_empty_padding_and_zero_mask_draws_do_not_redistribute_weights():
    logits = torch.zeros(2, 3, 2, requires_grad=True)
    labels = torch.zeros(2, 3, dtype=torch.long)
    answer = torch.tensor([[False, True, True], [False, True, True]])
    padding = torch.zeros_like(answer)
    selected = torch.tensor([[False, True, True], [False, False, False]])
    loss, metrics = objective(logits, labels, selected, answer, padding, torch.ones(2))
    assert loss.item() == pytest.approx(.9 * torch.log(torch.tensor(2.)).item() / 2)
    assert metrics["padding_loss"] == 0
    assert metrics["valid_examples"] == 2
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_all_zero_mask_draws_have_finite_zero_loss_and_gradients():
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    labels = torch.zeros(1, 3, dtype=torch.long)
    answer = torch.tensor([[False, True, False]])
    padding = torch.tensor([[False, False, True]])
    loss, metrics = objective(logits, labels, torch.zeros_like(answer), answer, padding, torch.tensor([.001]))
    loss.backward()
    assert loss == metrics["answer_loss"] == metrics["padding_loss"] == 0
    assert metrics["valid_examples"] == 1
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def base_config():
    return dict(corruption_mode="mask_only", structured_loss_behavior="corrupted_answer_tokens",
                eos_padding_loss=True, answer_padding_loss=dict(enabled=True, answer_weight=.9, padding_weight=.1))


def test_separate_objective_is_opt_in():
    assert _resolve_answer_padding_weights({}) is None
    assert _resolve_answer_padding_weights(base_config()) == (.9, .1)


@pytest.mark.parametrize("updates", [
    {"corruption_mode": "structured"}, {"structured_loss_behavior": "all_tokens"},
    {"eos_padding_loss": False}, {"answer_padding_loss": "enabled"},
    {"answer_padding_loss": {"enabled": "true"}},
    {"answer_padding_loss": {"enabled": True, "answer_weight": 9, "padding_weight": 1}},
    {"answer_padding_loss": {"enabled": True, "answer_weight": -0.1, "padding_weight": 1.1}},
    {"answer_padding_loss": {"enabled": True, "answer_weight": float("nan")}},
])
def test_invalid_separate_loss_configuration_fails(updates):
    with pytest.raises(ValueError, match="answer_padding_loss"):
        _resolve_answer_padding_weights({**base_config(), **updates})


@pytest.mark.parametrize("selected_logits", [False, True])
def test_tiny_training_saves_answer_padding_components(tmp_path, monkeypatch, selected_logits):
    from datasets import Dataset, DatasetDict
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model
    from test_data_corruption_loss import ToyTokenizer
    from diffusion_lm import training

    for name in ("LAD_STORAGE", "LAD_OUTPUT_ROOT"):
        monkeypatch.delenv(name, raising=False)
    tokenizer = ToyTokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = Dataset.from_list([
        {"instruction": "Answer", "input": "", "output": "abcd" * (i % 3 + 1)}
        for i in range(8)
    ])
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: DatasetDict(
        train=dataset, validation=dataset.select(range(4)), test=dataset.select(range(4)),
    ))
    monkeypatch.setattr(training.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    model = get_peft_model(LlamaForCausalLM(LlamaConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2,
    )), LoraConfig(r=2, target_modules=["q_proj", "v_proj"], bias="none"))
    before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    monkeypatch.setattr(training, "load_denoising_model", lambda config: (model, {}))
    saved = []
    monkeypatch.setattr(training, "_save_adapter", lambda model, tokenizer, path, norms: saved.append(path))
    config = {
        **base_config(), "dataset_name": "tiny-local", "model_name_or_path": "tiny-local",
        "output_dir": str(tmp_path / "output"), "cache_dir": str(tmp_path / "cache"),
        "base_model_cache_dir": str(tmp_path / "models"), "prepared_data_cache_dir": str(tmp_path / "prepared"),
        "precision": "fp32", "batch_size": 2, "eval_batch_size": 3,
        "max_updates": 2, "gradient_accumulation_steps": 2, "max_sequence_length": 32,
        "pad_to_multiple_of": 8, "learning_rate": .01, "scheduler": "constant",
        "logging_steps": 1, "validation_steps": 1, "num_workers": 0, "preprocessing_num_workers": 1,
        "frontier_masking_probability": .75, "t_min": .2,
        "selected_logit_optimization": selected_logits,
    }
    training.run_training(config)
    output = tmp_path / "output"
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert {record["split"] for record in records} == {"train", "train_interval", "validation"}
    assert len([record for record in records if record["split"] == "train_interval"]) == 2
    for record in [*records, json.loads((output / "test_metrics.json").read_text())]:
        assert record["weighted_loss"] == pytest.approx(.9 * record["answer_loss"] + .1 * record["padding_loss"], rel=1e-6)
    resolved = json.loads((output / "resolved_config.json").read_text())
    assert resolved["answer_padding_loss"] == config["answer_padding_loss"]
    assert saved  # Best-checkpoint selection ran using the new validation objective.
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters() if name in before)
