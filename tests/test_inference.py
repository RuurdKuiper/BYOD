import json
from pathlib import Path

import pytest
import torch

import diffusion_lm.inference as inference_module
from diffusion_lm.inference import InferenceSession, _apply_eos_eot_prediction_penalty, _apply_repetition_penalty, _block_step_plan, _llada_transfer_schedule, _precision_dtype, _prompt_ids, _remask_offsets, _safe_adapter_path, decode_denoising_state, denoise_stream, find_adapters, forward_denoising, llada_generate, load_local_legacy_session, preflight_session, render_denoising_step
from diffusion_lm.legacy_compat import LegacyCustomTransformerConfig, LegacyCustomTransformerModel, install_legacy_pickle_modules, patch_legacy_lora_modules, restore_legacy_pickle_modules


def test_adapter_discovery_only_lists_valid_saved_adapters(tmp_path):
    valid = tmp_path / "run-a" / "best"
    valid.mkdir(parents=True)
    (valid / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "tiny"}))
    incomplete = tmp_path / "run-b" / "checkpoint-1"
    incomplete.mkdir(parents=True)
    assert find_adapters(tmp_path) == ["run-a/best"]
    assert _safe_adapter_path(tmp_path, "run-a/best") == valid.resolve()
    with pytest.raises(ValueError):
        _safe_adapter_path(tmp_path, "../outside")


def test_hub_adapter_loader_downloads_then_uses_shared_loader(monkeypatch, tmp_path):
    adapter = tmp_path / "snapshot"
    adapter.mkdir()
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(adapter)

    sentinel = object()
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        inference_module,
        "_load_adapter_path",
        lambda path, device, quantization, preflight=True: (
            sentinel,
            Path(path),
            device,
            quantization,
            preflight,
        ),
    )

    result = inference_module.load_hub_adapter_session(
        "Ruurd/BYOD-Llama-3.1-8B",
        device_name="cuda",
        quantization="none",
        revision="main",
        cache_dir=tmp_path / "cache",
    )

    assert result == (sentinel, adapter.resolve(), "cuda", "none", True)
    assert calls["repo_id"] == "Ruurd/BYOD-Llama-3.1-8B"
    assert calls["repo_type"] == "model"
    assert calls["revision"] == "main"


def test_prompt_ids_falls_back_for_systemless_chat_template():
    class SystemlessTokenizer:
        chat_template = "gemma-like"
        name_or_path = "test/gemma-like"

        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **_):
            self.calls.append(messages)
            if any(message["role"] == "system" for message in messages):
                class TemplateError(Exception):
                    pass
                raise TemplateError("System role not supported")
            return [3, 4]

    tokenizer = SystemlessTokenizer()
    assert _prompt_ids(tokenizer, "What?", "Be brief.") == [3, 4]
    assert tokenizer.calls[1] == [{"role": "user", "content": "Be brief.\n\nWhat?"}]


def test_legacy_prompt_ids_do_not_require_a_chat_template():
    class BaseLlamaTokenizer:
        name_or_path = "base-llama"
        chat_template = None

        def encode(self, text, **_):
            self.prompt = text
            return [1, 2]

    tokenizer = BaseLlamaTokenizer()
    assert _prompt_ids(tokenizer, "What?", "Be brief.", "legacy_llama") == [1, 2]
    assert "<|start_header_id|>assistant<|end_header_id|>" in tokenizer.prompt
    assert "<|start_header_id|>system<|end_header_id|>\nBe brief.\n<|start_header_id|>user" in tokenizer.prompt


def test_bf16_inference_falls_back_to_fp16_on_non_bf16_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert _precision_dtype("bf16", torch.device("cuda")) == torch.float16


def test_confidence_guided_remasking_targets_the_least_confident_tokens():
    confidence = torch.tensor([.9, .2, .7, .1])
    assert _remask_offsets(confidence, .5, True).tolist() == [3, 1]


def test_llada_linear_schedule_transfers_every_mask_once():
    assert _llada_transfer_schedule(10, 4) == [3, 3, 2, 2]


def test_block_step_plan_distributes_steps_and_supports_a_short_final_block():
    assert _block_step_plan(10, 5, 4) == [
        (0, 0, 4, 0, 2),
        (0, 0, 4, 1, 2),
        (1, 4, 8, 0, 2),
        (1, 4, 8, 1, 2),
        (2, 8, 10, 0, 1),
    ]
    with pytest.raises(ValueError, match="at least the number of blocks"):
        _block_step_plan(10, 2, 4)


def test_denoising_state_is_copyable_with_explicit_spaced_masks():
    class Tokenizer:
        eos_token_id = 2

        def decode(self, token_ids, **_kwargs):
            vocabulary = {0: "Kill", 1: " first", 4: " chicken", 5: " quickly"}
            return "".join(vocabulary.get(token, "") for token in token_ids)

    assert decode_denoising_state(
        [0, 1, 9, 4, 9, 5, 2, 9], Tokenizer(), mask_token_id=9
    ) == "Kill first MASK chicken MASK quickly"
    assert decode_denoising_state(
        [9, 9, 4], Tokenizer(), mask_token_id=9
    ) == "MASK MASK chicken"
    assert decode_denoising_state(
        [0, 2, 4, 2], Tokenizer(), mask_token_id=9, show_eos_tokens=True
    ) == "Kill <EOS> chicken <EOS>"

    class EotTokenizer(Tokenizer):
        unk_token_id = 8

        def convert_tokens_to_ids(self, token):
            return {"<|eot_id|>": 7}.get(token, self.unk_token_id)

    assert decode_denoising_state(
        [0, 7, 4, 2], EotTokenizer(), mask_token_id=9, show_eos_tokens=True
    ) == "Kill <EOT> chicken <EOS>"


def test_denoising_html_uses_probability_colors_and_mask_chips():
    class Tokenizer:
        eos_token_id = 2

        def decode(self, token_ids, **_kwargs):
            return {4: " answer", 9: "MASK"}.get(token_ids[0], "")

    html = render_denoising_step(
        [1, 4, 9],
        [0.0, 0.75],
        answer_start=1,
        tokenizer=Tokenizer(),
        mask_token_id=9,
        step=1,
        total_steps=2,
        retained={0},
        frozen={0: 4},
        frozen_confidences={0: 0.75},
        frozen_steps={0: 1},
    )

    assert "sampling probability 75.0%" in html
    assert "predicted at iteration 1" in html
    assert "retained and locked" not in html
    assert "border-bottom" not in html
    assert ">mask</span>" in html

    iteration_html = render_denoising_step(
        [1, 4], [0.75], 1, Tokenizer(), 9, 1, 2,
        frozen={0: 4}, frozen_confidences={0: 0.75}, frozen_steps={0: 1},
        color_mode="Prediction iteration",
    )
    assert "hsl(210,90%" in iteration_html
    assert "Light-to-dark blue" in iteration_html

    plain_html = render_denoising_step(
        [1, 4], [0.75], 1, Tokenizer(), 9, 1, 2, color_mode="No coloring"
    )
    assert "color:inherit" in plain_html
    assert "Token coloring is disabled" in plain_html


def test_repetition_penalty_scales_probability_weight_and_excludes_current_position():
    logits = torch.zeros((1, 3, 6))
    logits[0, :, 2] = torch.tensor([4.0, -4.0, 2.0])
    logits[0, :, 3] = 6.0
    answer_ids = torch.tensor([[2, 2, 5]])

    penalized = _apply_repetition_penalty(
        logits, answer_ids, 2.0, mask_token_id=5, exclude_self=True
    )

    # At an existing occurrence, the current position is excluded, leaving one
    # other occurrence. At MASK, predicting a third copy divides its softmax
    # weight by 2**2, which subtracts log(2**2) from the logit.
    assert penalized[0, 0, 2].item() == pytest.approx(4.0 - torch.log(torch.tensor(2.0)).item())
    assert penalized[0, 1, 2].item() == pytest.approx(-4.0 - torch.log(torch.tensor(2.0)).item())
    assert penalized[0, 2, 2].item() == pytest.approx(2.0 - torch.log(torch.tensor(4.0)).item())
    # Tokens absent from the current answer are unchanged; MASK is excluded.
    assert torch.equal(penalized[0, :, 3], logits[0, :, 3])


def test_repetition_penalty_rejects_values_below_one():
    with pytest.raises(ValueError, match="at least 1.0"):
        _apply_repetition_penalty(
            torch.zeros((1, 1, 3)), torch.tensor([[1]]), 0.9, mask_token_id=2
        )


def test_repetition_penalty_excludes_special_tokens():
    logits = torch.zeros((1, 2, 6))
    logits[0, :, 2] = 4.0

    penalized = _apply_repetition_penalty(
        logits,
        torch.tensor([[2, 5]]),
        2.0,
        mask_token_id=5,
        excluded_token_ids={2},
    )

    assert torch.equal(penalized, logits)


def test_eos_eot_prediction_penalty_only_reduces_special_token_logits():
    logits = torch.zeros((2, 6), dtype=torch.bfloat16)
    logits[:, 2] = 4.0
    logits[:, 4] = 3.0

    penalized = _apply_eos_eot_prediction_penalty(
        logits, 10.0, eos_token_id=2, eot_token_id=4
    )

    expected_delta = torch.log(torch.tensor(10.0)).item()
    assert penalized.dtype == logits.dtype
    assert penalized[0, 2].float().item() == pytest.approx(4.0 - expected_delta, abs=.02)
    assert penalized[0, 4].float().item() == pytest.approx(3.0 - expected_delta, abs=.02)
    assert torch.equal(penalized[:, [0, 1, 3, 5]], logits[:, [0, 1, 3, 5]])
    assert torch.equal(_apply_eos_eot_prediction_penalty(logits, 1.0, 2, 4), logits)
    with pytest.raises(ValueError, match="at least 1.0"):
        _apply_eos_eot_prediction_penalty(logits, 0.5, 2, 4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_repetition_penalty_preserves_low_precision_logits_dtype(dtype):
    logits = torch.zeros((1, 3, 6), dtype=dtype)
    logits[0, :, 2] = 4.0

    penalized = _apply_repetition_penalty(
        logits,
        torch.tensor([[2, 2, 5]]),
        1.25,
        mask_token_id=5,
        exclude_self=True,
    )

    assert penalized.dtype == dtype
    assert penalized[0, 2, 2].float().item() == pytest.approx(
        4.0 - 2 * torch.log(torch.tensor(1.25)).item(), rel=0.01
    )


def test_legacy_wrapper_uses_its_own_forward_without_duplicate_keywords():
    class InnerModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask, output_hidden_states, use_cache):
            self.called = {"input_ids": input_ids, "attention_mask": attention_mask, "output_hidden_states": output_hidden_states, "use_cache": use_cache}
            return type("Output", (), {"logits": torch.zeros((*input_ids.shape, 3))})()

    class LegacyPeftOuter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = InnerModel()

        def forward(self, *_args, **_kwargs):
            raise AssertionError("version-sensitive PeftModel.forward should be bypassed")

    class LegacyModel(LegacyCustomTransformerModel):
        def __init__(self):
            super().__init__(LegacyCustomTransformerConfig(vocab_size=3))
            self.llama = LegacyPeftOuter()

    model = LegacyModel()
    session = InferenceSession(model, None, torch.device("cpu"), Path("."), {}, 0, legacy_wrapper=True)
    logits = forward_denoising(session, torch.tensor([[1, 2]]), torch.zeros((1, 2), dtype=torch.bool))
    assert logits.shape == (1, 2, 3)
    assert model.llama.base_model.called["use_cache"] is False


def test_legacy_pickle_compatibility_registers_main_module_aliases():
    import __main__
    previous = install_legacy_pickle_modules()
    try:
        assert hasattr(__main__, "CustomTransformerModel")
        assert hasattr(__main__, "CustomTransformerConfig")
    finally:
        restore_legacy_pickle_modules(previous)


def test_legacy_lora_patch_enables_the_current_peft_vanilla_branch():
    class OldLoraLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.ModuleDict()

    model = torch.nn.Sequential(OldLoraLinear(), OldLoraLinear())
    assert patch_legacy_lora_modules(model) == 2
    assert all(module.lora_variant == {} for module in model)


def test_local_legacy_loader_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        load_local_legacy_session(tmp_path / "missing.pth", "tokenizer", "cpu")


def test_preflight_runs_a_real_forward_pass():
    class Tokenizer:
        eos_token_id = 2
        chat_template = "template"
        name_or_path = "toy"

        def apply_chat_template(self, *_args, **_kwargs):
            return [1, 2]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask, use_cache):
            return type("Output", (), {"logits": torch.zeros((*input_ids.shape, 4))})()

    session = InferenceSession(Model(), Tokenizer(), torch.device("cpu"), Path("."), {}, 3)
    assert preflight_session(session) == (3, 4)


def test_early_stopping_requires_two_consecutive_identical_complete_predictions():
    class Tokenizer:
        eos_token_id = 2
        chat_template = "template"
        name_or_path = "toy"

        def apply_chat_template(self, *_args, **_kwargs):
            return [1, 2]

        def decode(self, token_ids, **_kwargs):
            return " ".join(map(str, token_ids))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask, use_cache):
            logits = torch.full((*input_ids.shape, 4), -100.0)
            logits[..., 1] = 100.0
            return type("Output", (), {"logits": logits})()

    session = InferenceSession(Model(), Tokenizer(), torch.device("cpu"), Path("."), {}, 3)
    states = list(denoise_stream(session, "Test", "System", 2, 6, .5, 1., 1, 1234, early_stopping=True))
    assert len(states) == 2
    assert "stopped early" in states[-1][1]
    assert "2 output tokens" in states[-1][1]


def test_early_stopping_ignores_changes_after_first_eos():
    class Tokenizer:
        eos_token_id = 2
        chat_template = "template"
        name_or_path = "toy"

        def apply_chat_template(self, *_args, **_kwargs):
            return [1]

        def decode(self, token_ids, **_kwargs):
            return " ".join(map(str, token_ids))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward(self, input_ids, attention_mask, use_cache):
            self.calls += 1
            logits = torch.full((*input_ids.shape, 7), -100.0)
            answer = [3, 2, 4 + self.calls % 2, 6 - self.calls % 2]
            for position, token in enumerate(answer, start=1):
                logits[:, position, token] = 100.0
            return type("Output", (), {"logits": logits})()

    model = Model()
    session = InferenceSession(model, Tokenizer(), torch.device("cpu"), Path("."), {}, 6)
    states = list(denoise_stream(
        session, "Test", "System", 4, 6, .5, 1., 1, 1234, early_stopping=True
    ))

    assert len(states) == 2
    assert "same answer for 2 consecutive iterations" in states[-1][1]


def test_early_stopping_continues_when_first_eos_moves():
    class Tokenizer:
        eos_token_id = 2
        chat_template = "template"
        name_or_path = "toy"

        def apply_chat_template(self, *_args, **_kwargs):
            return [1]

        def decode(self, token_ids, **_kwargs):
            return " ".join(map(str, token_ids))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward(self, input_ids, attention_mask, use_cache):
            self.calls += 1
            logits = torch.full((*input_ids.shape, 7), -100.0)
            answer = [3, 2, 4, 4] if self.calls != 2 else [3, 4, 2, 4]
            for position, token in enumerate(answer, start=1):
                logits[:, position, token] = 100.0
            return type("Output", (), {"logits": logits})()

    model = Model()
    session = InferenceSession(model, Tokenizer(), torch.device("cpu"), Path("."), {}, 6)
    states = list(denoise_stream(
        session, "Test", "System", 4, 6, .5, 1., 1, 1234, early_stopping=True
    ))

    assert len(states) == 4
    assert "same answer for 2 consecutive iterations" in states[-1][1]


def test_llada_session_uses_the_app_denoising_loop():
    class Tokenizer:
        eos_token_id = 2
        name_or_path = "toy-llada"

        def apply_chat_template(self, messages, **_kwargs):
            self.messages = messages
            return [1]

        def decode(self, token_ids, **_kwargs):
            return " ".join(map(str, token_ids))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask):
            self.attention_mask = attention_mask
            self.inference_mode = torch.is_inference_mode_enabled()
            logits = torch.full((*input_ids.shape, 10), -100.0)
            logits[..., 3] = 100.0
            return type("Output", (), {"logits": logits})()

    tokenizer = Tokenizer()
    model = Model()
    session = InferenceSession(model, tokenizer, torch.device("cpu"), Path("."), {}, 9, llada=True, prompt_format="llada")
    states = list(denoise_stream(session, "Question", "System", 3, 2, .5, 0., 20, 1234))

    assert len(states) == 2
    assert states[-1][0] == "3 3 3"
    assert "Denoising step 1/2" in states[0][1]
    assert tokenizer.messages == [{"role": "user", "content": "System\n\nQuestion"}]
    assert model.attention_mask.dtype == torch.long
    assert model.inference_mode is True


def test_streaming_denoising_delays_eos_retention_when_configured():
    class Tokenizer:
        eos_token_id = 2
        unk_token_id = 0
        chat_template = "template"
        name_or_path = "toy"

        def apply_chat_template(self, *_args, **_kwargs):
            return [1]

        def convert_tokens_to_ids(self, token):
            return {"<|eot_id|>": 4}.get(token, self.unk_token_id)

        def decode(self, token_ids, **_kwargs):
            return " ".join(map(str, token_ids))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.inputs = []

        def forward(self, input_ids, attention_mask, use_cache):
            self.inputs.append(input_ids.detach().clone())
            logits = torch.full((*input_ids.shape, 6), -10.0)
            logits[:, -2, 2] = 10.0  # EOS would otherwise be retained first.
            logits[:, -1, 3] = 8.0
            return type("Output", (), {"logits": logits})()

    model = Model()
    session = InferenceSession(model, Tokenizer(), torch.device("cpu"), Path("."), {}, 5)
    list(denoise_stream(
        session, "Question", "System", 2, 3, 1.0, 0.0, 1, 1234,
        permanent_unmask=True,
        confidence_guided=False,
        proportional_unmask=False,
        confidence_eos_eot_inf=True,
    ))

    assert model.inputs[1].tolist() == [[1, 5, 3]]


def test_eos_prediction_penalty_does_not_enable_confidence_guided_remasking(monkeypatch):
    from diffusion_lm import inference

    class Tokenizer:
        eos_token_id = 2
        unk_token_id = 0
        chat_template = "template"
        name_or_path = "toy"

        def apply_chat_template(self, *_args, **_kwargs):
            return [1]

        def convert_tokens_to_ids(self, token):
            return {"<|eot_id|>": 4}.get(token, self.unk_token_id)

        def decode(self, token_ids, **_kwargs):
            return " ".join(map(str, token_ids))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask, use_cache):
            logits = torch.full((*input_ids.shape, 6), -10.0)
            logits[..., 2] = 5.0
            logits[..., 3] = 4.0
            return type("Output", (), {"logits": logits})()

    guided_arguments = []
    original_remask_offsets = inference._remask_offsets

    def recording_remask_offsets(confidence, mask_probability, confidence_guided):
        guided_arguments.append(confidence_guided)
        return original_remask_offsets(confidence, mask_probability, confidence_guided)

    monkeypatch.setattr(inference, "_remask_offsets", recording_remask_offsets)
    session = InferenceSession(Model(), Tokenizer(), torch.device("cpu"), Path("."), {}, 5)
    list(denoise_stream(
        session,
        "Question",
        "System",
        2,
        2,
        1.0,
        0.0,
        1,
        1234,
        confidence_guided=False,
        confidence_eos_eot_inf=False,
        eos_eot_prediction_penalty=10.0,
    ))

    assert guided_arguments == [False]


def test_copyable_trajectory_can_show_prediction_before_and_after_remasking():
    class Tokenizer:
        eos_token_id = 2
        chat_template = "template"
        name_or_path = "toy"

        def apply_chat_template(self, *_args, **_kwargs):
            return [1]

        def decode(self, token_ids, **_kwargs):
            return " ".join(map(str, token_ids))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask, use_cache):
            logits = torch.full((*input_ids.shape, 6), -10.0)
            logits[..., 3] = 10.0
            return type("Output", (), {"logits": logits})()

    session = InferenceSession(Model(), Tokenizer(), torch.device("cpu"), Path("."), {}, 5)
    states = list(denoise_stream(
        session,
        "Question",
        "System",
        4,
        2,
        1.0,
        0.0,
        1,
        1234,
        include_pre_remask_prediction=True,
    ))

    assert states[0][0].startswith("Predicted (before re-mask):\n3 3 3 3\nState after re-mask:\n")
    assert "MASK" in states[0][0]
    assert states[-1][0] == (
        "Predicted (before re-mask):\n3 3 3 3\n"
        "State after re-mask (unchanged; final state):\n3 3 3 3"
    )


def test_streaming_block_generation_finishes_each_block_before_the_next():
    class Tokenizer:
        eos_token_id = 2
        chat_template = "template"
        name_or_path = "toy"

        def apply_chat_template(self, *_args, **_kwargs):
            return [1]

        def decode(self, token_ids, **_kwargs):
            return " ".join(map(str, token_ids))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.inputs = []

        def forward(self, input_ids, attention_mask, use_cache):
            self.inputs.append(input_ids.detach().clone())
            logits = torch.full((*input_ids.shape, 6), -10.0)
            logits[..., 3] = 10.0
            return type("Output", (), {"logits": logits})()

    model = Model()
    session = InferenceSession(model, Tokenizer(), torch.device("cpu"), Path("."), {}, 5)
    states = list(denoise_stream(
        session,
        "Question",
        "System",
        4,
        4,
        1.0,
        0.0,
        1,
        1234,
        include_pre_remask_prediction=True,
        block_length=2,
    ))

    assert len(states) == 4
    assert model.inputs[0].tolist() == [[1, 5, 5, 5, 5]]
    assert model.inputs[2].tolist() == [[1, 3, 3, 5, 5]]
    assert "Predicted (before re-mask):\n3 3 MASK MASK" in states[1][0]
    assert "Predicted (before re-mask):\n3 3 3 3" in states[2][0]
    assert states[-1][0].endswith("State after re-mask (unchanged; final state):\n3 3 3 3")
    assert "block 2/2" in states[-1][1]


def test_retained_positions_can_remain_editable_or_lock_their_token_values():
    class Tokenizer:
        eos_token_id = 2
        chat_template = "template"
        name_or_path = "toy"

        def apply_chat_template(self, *_args, **_kwargs):
            return [1]

        def decode(self, token_ids, **_kwargs):
            return " ".join(map(str, token_ids))

    class ChangingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.inputs = []

        def forward(self, input_ids, attention_mask, use_cache):
            self.inputs.append(input_ids.detach().clone())
            call = len(self.inputs) - 1
            logits = torch.full((*input_ids.shape, 10), -10.0)
            logits[:, -2, 3 + call] = 10.0
            logits[:, -1, 8] = 8.0
            return type("Output", (), {"logits": logits})()

    def generate(freeze_retained_tokens):
        model = ChangingModel()
        session = InferenceSession(model, Tokenizer(), torch.device("cpu"), Path("."), {}, 9)
        states = list(denoise_stream(
            session,
            "Question",
            "System",
            2,
            3,
            1.0,
            0.0,
            1,
            1234,
            permanent_unmask=True,
            confidence_guided=True,
            proportional_unmask=False,
            freeze_retained_tokens=freeze_retained_tokens,
        ))
        return model, states

    editable_model, editable_states = generate(False)
    locked_model, locked_states = generate(True)

    assert editable_model.inputs[1].tolist() == [[1, 3, 9]]
    assert editable_model.inputs[2].tolist() == [[1, 4, 9]]
    assert editable_states[-1][0].startswith("5 ")
    assert "retained 1 tokens" in editable_states[-1][1]
    assert "editable" not in editable_states[-1][1]
    assert locked_model.inputs[1].tolist() == [[1, 3, 9]]
    assert locked_model.inputs[2].tolist() == [[1, 3, 9]]
    assert locked_states[-1][0].startswith("3 ")
    assert "retained 1 tokens" in locked_states[-1][1]
    assert "locked" not in locked_states[-1][1]


def test_official_llada_sampler_delays_eos_when_configured():
    class Tokenizer:
        eos_token_id = 2

        def apply_chat_template(self, messages, **_kwargs):
            self.messages = messages
            return [1]

        def decode(self, token_ids, skip_special_tokens=False):
            if skip_special_tokens:
                token_ids = [token for token in token_ids if token != self.eos_token_id]
            return " ".join(map(str, token_ids))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.inputs = []

        def forward(self, input_ids, attention_mask):
            self.inputs.append(input_ids.detach().clone())
            logits = torch.full((*input_ids.shape, 6), -10.0)
            logits[:, 1, 2] = 10.0  # EOS is most confident at the first answer position.
            logits[:, 2, 3] = 8.0
            return type("Output", (), {"logits": logits})()

    tokenizer = Tokenizer()
    model = Model()
    session = InferenceSession(model, tokenizer, torch.device("cpu"), Path("."), {}, 5, llada=True, prompt_format="llada")

    text = llada_generate(session, "Question", gen_length=2, steps=2, confidence_eos_eot_inf=True, eot_token_id=4)

    assert model.inputs[1].tolist() == [[1, 5, 3]]
    assert text == "3"
    assert tokenizer.messages == [{"role": "user", "content": "Question"}]


def test_official_llada_sampler_also_supports_native_adapter_prompts():
    class Tokenizer:
        eos_token_id = 2
        chat_template = "toy"
        name_or_path = "toy-adapter"

        def apply_chat_template(self, messages, **_kwargs):
            self.messages = messages
            return [1]

        def decode(self, token_ids, **_kwargs):
            return " ".join(map(str, token_ids))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, **_kwargs):
            logits = torch.full((*input_ids.shape, 6), -10.0)
            logits[..., 3] = 10.0
            return type("Output", (), {"logits": logits})()

    tokenizer = Tokenizer()
    session = InferenceSession(Model(), tokenizer, torch.device("cpu"), Path("."), {}, 5)

    text = llada_generate(session, "Question", gen_length=2, steps=2, system_prompt="System")

    assert text == "3 3"
    assert tokenizer.messages == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Question"},
    ]
