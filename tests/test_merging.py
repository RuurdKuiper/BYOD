import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from diffusion_lm.merging import _load_normalization_state
from diffusion_lm.modeling import forward_bidirectional


def test_lora_merge_preserves_bidirectional_logits():
    torch.manual_seed(7)
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    ))
    model = get_peft_model(base, LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )).eval()
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.1)

    input_ids = torch.tensor([[3, 4, 5]])
    padding = torch.zeros_like(input_ids, dtype=torch.bool)
    expected = forward_bidirectional(model, input_ids, padding)
    merged = model.merge_and_unload(safe_merge=True).eval()
    actual = forward_bidirectional(merged, input_ids, padding)

    assert not any("lora_" in name for name, _ in merged.named_parameters())
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_saved_normalization_state_is_restored_before_merge(tmp_path):
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    ))
    model = get_peft_model(base, LoraConfig(r=2, target_modules=["q_proj"], bias="none"))
    name, parameter = next((n, p) for n, p in model.named_parameters() if "norm" in n)
    expected = torch.full_like(parameter, 0.25)
    torch.save({name: expected}, tmp_path / "normalization_state.pt")

    assert _load_normalization_state(model, tmp_path) == 1
    assert torch.equal(dict(model.named_parameters())[name], expected)
