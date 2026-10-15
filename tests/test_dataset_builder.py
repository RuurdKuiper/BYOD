import json

from datasets import Dataset, DatasetDict, concatenate_datasets

import diffusion_lm.dataset_builder as dataset_builder
from diffusion_lm.dataset_builder import BuildConfig, _repeat_dataset, _take, _targets, choose_system_prompt, format_hellaswag, format_mc, format_tulu, prompt_hash, row_prompt_hashes, write_manifest


class TinyTokenizer:
    eos_token_id = 99
    name_or_path = "tiny"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, truncation=False, max_length=None):
        size = sum(len(m["content"].split()) for m in messages)
        ids = list(range(size + int(add_generation_prompt)))
        return ids[:max_length] if truncation else ids

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = list(range(20, 20 + len(text.split())))
        return {"input_ids": ids[:max_length] if truncation else ids}

    def encode(self, text, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return " ".join(f"token{index}" for index in ids)


class FailingTokenizer(TinyTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("extreme text should be rejected before tokenization")


def test_default_targets_are_exact_and_match_requested_mix():
    assert _targets(100, BuildConfig().weights) == {"general": 45, "reasoning": 18, "math": 18, "code": 19}


def test_formatters_reject_bad_mc_and_keep_final_tulu_exchange():
    assert format_mc("q", ["x", "y"], 1)["output"] == "B: y"
    assert format_mc("q", ["x"], 4) is None
    row = {"messages": [{"role": "user", "content": "first"}, {"role": "assistant", "content": "one"},
                        {"role": "user", "content": "second"}, {"role": "assistant", "content": "two"}]}
    result = format_tulu(row)
    assert result["output"] == "two"
    assert "first" in result["input"] and "second" in result["input"]


def test_hellaswag_asks_an_explicit_continuation_question():
    result = format_hellaswag({"ctx": "A person opens a door.", "endings": ["They enter.", "The moon explodes."], "label": "0"})
    assert "What most plausibly happens next?" in result["input"]
    assert result["output"] == "A: They enter."


def test_native_system_is_preserved_and_generated_variation_is_deterministic():
    assert choose_system_prompt({"system": "Be a pirate."}, "general:tulu-3", "0" * 64) == "Be a pirate."
    first = choose_system_prompt({}, "math:orca", "7" * 64)
    assert first == choose_system_prompt({}, "math:orca", "7" * 64)
    assert first != "You are a helpful assistant."


def test_take_filters_heldout_and_lengths_and_stores_clean_ids_twice():
    rows = [{"q": "held out", "a": "no"}, {"q": "one two three four", "a": "too long"}, {"q": "usable", "a": "answer"}]
    formatter = lambda x: {"instruction": "", "input": x["q"], "output": x["a"]}
    config = BuildConfig(total_examples=1, max_prompt_tokens=8, max_sequence_tokens=9)
    result = _take(rows, formatter, 1, TinyTokenizer(), config, {prompt_hash("held out")}, "general:test")
    assert len(result) == 1
    assert result[0]["input"] == "usable"
    assert result[0]["input_ids"] == result[0]["labels"]
    assert result[0]["labels"][-1] == TinyTokenizer.eos_token_id
    assert result[0]["answer_truncated"] is False


def test_take_optionally_truncates_long_answers_without_false_eos():
    rows = [{"q": "usable", "a": "one two three four five"}]
    formatter = lambda x: {"instruction": "", "input": x["q"], "output": x["a"]}
    base = BuildConfig(total_examples=1, max_prompt_tokens=8, max_sequence_tokens=10)
    assert _take(rows, formatter, 1, TinyTokenizer(), base, set(), "general:test") == []

    enabled = BuildConfig(
        total_examples=1,
        max_prompt_tokens=8,
        max_sequence_tokens=10,
        truncate_long_answers=True,
    )
    result = _take(rows, formatter, 1, TinyTokenizer(), enabled, set(), "general:test")
    assert len(result) == 1
    assert result[0]["answer_truncated"] is True
    assert len(result[0]["labels"]) == enabled.max_sequence_tokens
    assert result[0]["labels"][-1] != TinyTokenizer.eos_token_id


def test_take_rejects_pathological_text_before_normalization_and_tokenization():
    config = BuildConfig(total_examples=1, max_prompt_tokens=2, max_sequence_tokens=4)
    rows = [{"q": "x" * 101, "a": "answer"}]
    formatter = lambda x: {"instruction": "", "input": x["q"], "output": x["a"]}
    assert _take(rows, formatter, 1, FailingTokenizer(), config, set(), "general:test") == []


def test_prompt_hashes_ignore_empty_raw_input_for_instruction_only_rows():
    first = row_prompt_hashes({"instruction": "Write function A", "input": ""})
    second = row_prompt_hashes({"instruction": "Write function B", "input": ""})
    assert prompt_hash("") not in first
    assert first.isdisjoint(second)


def test_take_excludes_prior_prompts_and_deduplicates_current_sources():
    rows = [
        {"q": "old prompt", "a": "old"},
        {"q": "new prompt", "a": "first"},
        {"q": "NEW, PROMPT!", "a": "normalized duplicate"},
        {"q": "another prompt", "a": "second"},
    ]
    formatter = lambda x: {"instruction": "", "input": x["q"], "output": x["a"]}
    config = BuildConfig(total_examples=2, max_prompt_tokens=20, max_sequence_tokens=30)
    used, stats = set(), {}
    result = _take(
        rows, formatter, 2, TinyTokenizer(), config, set(), "general:test",
        excluded=row_prompt_hashes({"instruction": "", "input": "old prompt"}),
        used=used, stats=stats,
    )
    assert [row["input"] for row in result] == ["new prompt", "another prompt"]
    assert all(row["sample_origin"] == "new_source" for row in result)
    assert stats == {"excluded_overlap": 1, "within_build_duplicate": 1}


def test_repeat_dataset_uses_every_unique_row_before_balanced_repeats():
    source = Dataset.from_dict({"value": [0, 1, 2]})
    repeated = _repeat_dataset(source, 8, 42, concatenate_datasets)
    values = repeated["value"]
    assert len(values) == 8
    assert values[:6] == [0, 1, 2, 0, 1, 2]
    assert max(values.count(x) for x in range(3)) - min(values.count(x) for x in range(3)) <= 1


def test_excluded_fallback_reuses_only_training_rows_and_preserves_heldout(tmp_path, monkeypatch):
    def prior_row(category, split):
        prompt = f"prior {category} {split}"
        return {
            "system": "You are a helpful assistant.", "instruction": "", "input": prompt,
            "output": "answer", "input_ids": [1, 2, 99], "labels": [1, 2, 99],
            "category": category, "source": f"{category}:prior", "answer_truncated": False,
        }

    prior = DatasetDict({
        "train": Dataset.from_list([prior_row(category, "train") for category in ("general", "reasoning", "math", "code")]),
        "validation": Dataset.from_list([prior_row(category, "validation") for category in ("general", "reasoning", "math", "code")]),
        "test": Dataset.from_list([prior_row(category, "test") for category in ("general", "reasoning", "math", "code")]),
    })
    exclusion_path = tmp_path / "prior"
    prior.save_to_disk(str(exclusion_path))

    def fake_load_dataset(path, name=None, split="train", **_):
        suffix = str(name or split).replace(" ", "-")
        if path == "allenai/tulu-3-sft-mixture":
            return Dataset.from_list([{"source": "general", "messages": [
                {"role": "user", "content": f"new tulu {suffix}"},
                {"role": "assistant", "content": "answer"},
            ]}])
        if path in {"crumb/Clean-Instruct-3M", "vicgalle/alpaca-gpt4", "tatsu-lab/alpaca"}:
            return Dataset.from_list([{"instruction": f"new {path} {suffix}", "input": "", "output": "answer"}])
        if path == "cais/mmlu":
            return Dataset.from_list([{"question": f"new mmlu {suffix}", "choices": ["x", "y"], "answer": 0}])
        if path == "Rowan/hellaswag":
            return Dataset.from_list([{"ctx": f"new hellaswag {suffix}", "endings": ["x", "y"], "label": "0"}])
        if path == "allenai/ai2_arc":
            return Dataset.from_list([{"question": f"new arc {suffix}", "choices": {"text": ["x", "y"], "label": ["A", "B"]}, "answerKey": "A"}])
        if path in {"microsoft/orca-math-word-problems-200k", "openai/gsm8k"}:
            return Dataset.from_list([{"question": f"new math {path} {suffix}", "answer": "1"}])
        if path == "OpenCoder-LLM/opc-sft-stage2":
            return Dataset.from_list([{"instruction": f"new code {suffix}", "output": "pass"}])
        if path == "google-research-datasets/mbpp":
            return Dataset.from_list([{"prompt": f"new mbpp {suffix}", "code": "pass"}])
        raise AssertionError(path)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_, **__: TinyTokenizer())
    monkeypatch.setattr(dataset_builder, "_evaluation_hashes", lambda _: set())
    config = BuildConfig(
        total_examples=40, max_prompt_tokens=100, max_sequence_tokens=200,
        weights={name: 0.25 for name in ("general", "reasoning", "math", "code")},
        exclude_dataset=str(exclusion_path), allow_excluded_fallback=True,
    )
    result = dataset_builder.build_dataset(config)
    assert {split: len(rows) for split, rows in result.items()} == {"train": 32, "validation": 4, "test": 4}
    assert set(result["validation"]["sample_origin"]) == {"excluded_heldout"}
    assert set(result["test"]["sample_origin"]) == {"excluded_heldout"}
    assert "excluded_training_fallback" in result["train"]["sample_origin"]
    assert all("prior" not in text or "train" in text for text in result["train"]["input"])

    config.build_report["verified"] = True
    manifest_path = tmp_path / "manifest.json"
    write_manifest(result, config, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["novelty"]["verified"] is True
    assert "build_report" not in manifest["config"]
    assert manifest["sample_origins"]["validation"] == {"excluded_heldout": 4}
