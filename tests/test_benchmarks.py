import json

import pytest
import torch

from diffusion_lm.benchmarks import BenchmarkExample, BenchmarkRunReporter, _benchmark_spec, _boxed, _choice_prompt, _extract_python_code, _gsm8k_prompt, _humaneval_prompt, _math_prompt, _mbpp_prompt, _multiple_choice_fields, _sample_indices, extract_answer, load_benchmark, resolve_autoregressive_generation_settings, resolve_generation_settings, resolve_llada_generation_settings, resolve_mask_only_generation_settings, score_prediction, score_texts_with_model


def test_benchmark_reporter_isolates_and_structures_each_run(tmp_path):
    reporter = BenchmarkRunReporter(tmp_path, {"tasks": ["arc_c"]}, "smoke run")
    record = {"model": "llada:GSAI-ML/LLaDA-8B-Instruct", "task": "arc_c", "method": "diffusion", "prediction": "A"}
    summary = {"model": record["model"], "task": "arc_c", "method": "diffusion", "accuracy": 1.0}

    reporter.save_result(record)
    reporter.save_summary(summary)
    run_path = reporter.complete()

    assert run_path.parent == tmp_path
    assert run_path.name.endswith("--smoke-run")
    assert json.loads((run_path / "run.json").read_text())["status"] == "completed"
    assert json.loads((run_path / "summary.json").read_text())["models"][0]["results"] == [summary]
    group = run_path / "models" / "llada-gsai-ml-llada-8b-instruct" / "arc_c" / "diffusion"
    assert json.loads((group / "results.jsonl").read_text()) == record
    assert json.loads((group / "summary.json").read_text()) == summary


def test_open_ended_benchmark_has_reproducible_hundred_prompt_suite():
    examples = load_benchmark("open_ended", "test", None, "unused", None)

    assert len(examples) == 100
    assert examples[0].example_id == "0"
    assert examples[0].kind == "open_ended"
    assert examples[0].prompt == "What do you know about Amsterdam?"


def test_open_ended_benchmark_respects_limit():
    examples = load_benchmark("open_ended", "test", 3, "unused", None)

    assert len(examples) == 3


def test_open_ended_benchmark_supports_fractional_smoke_suite():
    examples = load_benchmark("open_ended", "test", None, "unused", None, limit_fraction=0.05)

    assert len(examples) == 5
    assert [example.example_id for example in examples] == ["0", "20", "40", "60", "80"]


def test_bidirectional_infilling_benchmark_is_a_simple_right_context_copy_test():
    examples = load_benchmark("bidirectional_infilling", "test", None, "unused", None)

    assert len(examples) == 20
    assert len({example.example_id for example in examples}) == 20
    assert all(example.metadata["target_text"].strip() in example.metadata["answer_suffix"] for example in examples)
    assert all(example.metadata["target_text"].strip() not in example.metadata["answer_prefix"] for example in examples)
    assert all(example.metadata["system_prompt"] == "You are a helpful assistant." for example in examples)
    assert all(example.metadata["answer_prefix"] == "Answer:" for example in examples)
    assert all("COPY:" in example.metadata["answer_suffix"] for example in examples)
    assert "exactly" in examples[0].prompt


def test_bidirectional_infilling_benchmark_respects_limit():
    examples = load_benchmark("bidirectional_infilling", "test", 7, "unused", None)

    assert len(examples) == 7
    assert examples[0].example_id == "copy-000"


def test_open_ended_scoring_reports_median_per_response_perplexity():
    class Tokenizer:
        all_special_ids = []

        def __call__(self, text, **_kwargs):
            token = 0 if text == "easy" else 1
            length = 2 if text != "long-hard" else 8
            return {"input_ids": torch.tensor([[token] * length])}

        def encode(self, text, **_kwargs):
            return [0 if text == "easy" else 1]

    class Model(torch.nn.Module):
        def forward(self, input_ids, use_cache=False):
            logits = torch.zeros((*input_ids.shape, 2))
            logits[..., 0] = 2.0
            return type("Output", (), {"logits": logits})()

    scores = score_texts_with_model(
        Model(), Tokenizer(), torch.device("cpu"), ["easy", "hard", "long-hard"]
    )
    per_response = sorted(item["perplexity"] for item in scores["per_text"])

    assert scores["median_perplexity"] == per_response[1]
    assert scores["mean_perplexity"] == pytest.approx(sum(per_response) / len(per_response))
    assert scores["median_perplexity"] != pytest.approx(scores["perplexity"])


def test_fractional_sampling_is_evenly_spaced_and_rounded_up():
    assert _sample_indices(101, limit_fraction=0.05) == [0, 16, 33, 50, 67, 84]


def test_limited_sampling_spans_grouped_subjects():
    indices = _sample_indices(400, limit=50, shuffle=True)

    assert indices == _sample_indices(400, limit=50, shuffle=True)
    assert {index // 100 for index in indices} == {0, 1, 2, 3}


def test_limited_sampling_keeps_the_prefix_unless_shuffle_is_requested():
    assert _sample_indices(100, limit=5) == [0, 1, 2, 3, 4]


def test_benchmark_limits_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either limit or limit_fraction"):
        _sample_indices(100, limit=5, limit_fraction=0.05)


def test_hellaswag_uses_endings_as_answer_choices():
    row = {"ctx": "A person starts cooking.", "endings": ["A", "B", "C", "D"], "label": "2"}

    question, choices, answer = _multiple_choice_fields("hellaswag", row, 0)

    assert question == "A person starts cooking."
    assert choices == ["A", "B", "C", "D"]
    assert answer == "C"


def test_arc_maps_dataset_choice_labels_to_output_letters():
    row = {
        "question": "Which answer is correct?",
        "choices": {"text": ["First", "Second", "Third", "Fourth"], "label": ["1", "2", "3", "4"]},
        "answerKey": "3",
    }

    question, choices, answer = _multiple_choice_fields("arc_c", row, 0)

    assert question == "Which answer is correct?"
    assert choices == ["First", "Second", "Third", "Fourth"]
    assert answer == "C"


def test_multiple_choice_prompt_and_target_match_training_contract():
    prompt = _choice_prompt("mmlu", "Question?", ["First", "Second"])
    example = BenchmarkExample("mmlu", "0", prompt, "B: Second", "multiple_choice", {})

    assert prompt == "Answer the following multiple-choice question. Start your response with the correct option label followed by a colon.\n\nQuestion?\n\nA: First\nB: Second"
    assert "example `A:`" not in prompt
    assert score_prediction(example, "B: Second")
    assert score_prediction(example, "B. Second")
    assert score_prediction(example, "The correct answer is B: Second")
    assert score_prediction(example, "The answer is B: Second")
    assert score_prediction(example, "I would select option B.")
    assert not score_prediction(example, "I considered A and B, but the latter seems stronger.")


def test_reasoning_multiple_choice_tasks_request_answer_on_final_line():
    for task in ("mmlu_pro", "gpqa"):
        prompt = _choice_prompt(task, "Question?", ["First", "Second"])

        assert "Think through the problem concisely" in prompt
        assert "put no option text or punctuation after the label" in prompt.casefold()
        assert "Start your response" not in prompt

    for task in ("mmlu_pro", "gpqa"):
        assert "`ANSWER: A`" not in _choice_prompt(task, "Question?", ["First", "Second"])


def test_mmlu_prompts_include_available_subject_category_without_label_anchor():
    mmlu = _choice_prompt("mmlu", "Question?", ["First", "Second"], "college_biology")
    mmlu_pro = _choice_prompt("mmlu_pro", "Question?", ["First", "Second"], "computer science")

    assert "Subject category: college biology." in mmlu
    assert "Subject category: computer science." in mmlu_pro
    assert "`ANSWER: A`" not in mmlu_pro


def test_multiple_choice_extraction_accepts_official_answer_line():
    assert extract_answer("Reasoning here.\nANSWER: B", "multiple_choice") == "B"
    assert extract_answer("Reasoning here.\nANSWER: **B**", "multiple_choice") == "B"
    assert extract_answer("ANSWER: C: Third option", "multiple_choice") == "C"


def test_multiple_choice_scoring_accepts_an_exact_declared_option_text():
    example = BenchmarkExample(
        "mmlu_pro", "0", "prompt", "E: Full-service agency.", "multiple_choice", {}
    )

    assert extract_answer("ANS: Full-service agency.", "multiple_choice", example.answer) == "E"
    assert score_prediction(example, "ANS: Full-service agency.")
    assert not score_prediction(example, "ANS: Full service agency.")
    assert not score_prediction(example, "A full-service agency is likely.")


def test_multiple_choice_extraction_ignores_unmarked_letters_in_explanations():
    assert extract_answer("After considering A and B, the latter is stronger.", "multiple_choice") == ""
    assert extract_answer("The correct answer is B: 12.", "multiple_choice") == "B"


def test_hellaswag_uses_training_prompt_and_labelled_validation_split():
    prompt = _choice_prompt("hellaswag", "A person opens a door.", ["They enter.", "They leave."])

    assert prompt.startswith("Choose the option that most plausibly continues the described event. Start your response with the correct option label followed by a colon")
    assert _benchmark_spec("hellaswag", "test")[2] == "validation"


def test_gpqa_uses_its_published_train_named_evaluation_split():
    assert _benchmark_spec("gpqa", "test")[2] == "train"


def test_gpqa_option_shuffle_is_stable_across_evaluation_subsets():
    row = {
        "Question": "Which option is correct?",
        "Correct Answer": "Correct",
        "Incorrect Answer 1": "Wrong 1",
        "Incorrect Answer 2": "Wrong 2",
        "Incorrect Answer 3": "Wrong 3",
    }

    assert _multiple_choice_fields("gpqa", row, 0) == _multiple_choice_fields("gpqa", row, 87)


def test_gsm8k_scores_the_last_numeric_answer_from_a_rationale():
    example = BenchmarkExample("gsm8k", "0", "problem", "work shown\n#### 3", "gsm8k", {})

    assert extract_answer(example.answer, "gsm8k") == "3"
    assert score_prediction(example, "Two blue plus one white gives 3 bolts.")
    assert not score_prediction(example, "A robe takes 2 bolts and then 1 bolt.")


def test_gsm8k_prompt_requests_canonical_final_answer_marker():
    prompt = _gsm8k_prompt("What is 20 + 22?")

    assert "step by step" in prompt
    assert "final line in the form `#### number`" in prompt
    assert prompt.endswith("What is 20 + 22?")


def test_math_ignores_terminal_punctuation_for_final_answer():
    example = BenchmarkExample("math", "0", "problem", "Therefore, $\\boxed{27}$.", "math", {})

    assert extract_answer(example.answer, "math") == "27"
    assert score_prediction(example, "27.")


def test_math_extracts_boxed_answers_with_nested_latex_braces():
    answer = r"Therefore, the answer is $\boxed{\left(3, \frac{\pi}{2}\right)}$."

    assert _boxed(answer) == r"\left(3, \frac{\pi}{2}\right)"
    assert extract_answer(answer, "math") == r"(3,\frac{\pi}{2})"


def test_math_accepts_only_a_terminal_inline_expression_when_box_is_missing():
    target = r"Therefore, $\boxed{\left(3, \frac{\pi}{2}\right)}$."
    prediction = r"The radius is 3. Therefore the coordinates are $(3,\\frac{\pi}{2}).$"

    assert score_prediction(BenchmarkExample("math", "0", "problem", target, "math", {}), prediction)
    assert extract_answer(r"An intermediate value is $27$, but more work remains.", "math") != "27"


def test_math_removes_thousands_separators_without_corrupting_structured_answers():
    assert extract_answer(r"\boxed{1,234,567}", "math") == "1234567"
    assert extract_answer(r"\boxed{(3,4)}", "math") == "(3,4)"


def test_math_normalizes_redundant_braces_and_degree_notation():
    assert extract_answer(r"\boxed{{9}}", "math") == "9"
    assert extract_answer(r"\boxed{90 degrees}", "math") == "90"
    assert extract_answer(r"\boxed{90^\circ}", "math") == "90"


def test_math_prompt_requests_a_boxed_answer_after_reasoning():
    prompt = _math_prompt("What is 1 + 1?")

    assert prompt.startswith("Solve the following mathematics problem step by step.")
    assert "Check every arithmetic and algebraic step" in prompt
    assert "Simplify fractions, radicals, and expressions completely" in prompt
    assert "FINAL: \\boxed{answer}" in prompt
    assert "Put only the answer inside the box" in prompt
    assert prompt.endswith("What is 1 + 1?")


def _humaneval_add_example():
    prompt = 'def add(a, b):\n    """Return the sum."""\n'
    return BenchmarkExample("humaneval", "0", prompt, "    return a + b\n", "code", {
        "prompt": prompt,
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
    })


def test_humaneval_prompt_requests_a_complete_fenced_function():
    specification = 'def add(a, b):\n    """Return the sum."""\n'

    prompt = _humaneval_prompt(specification)

    assert prompt.startswith("Implement the Python function described below.")
    assert "Preserve the exact function name, signature, and return type" in prompt
    assert "Silently trace the implementation against every shown example" in prompt
    assert "exactly one complete Markdown code block tagged `python`" in prompt
    assert prompt.endswith(specification.strip())


def test_humaneval_executes_canonical_body_completions_with_the_prompt():
    assert score_prediction(_humaneval_add_example(), "    return a + b")
    assert score_prediction(_humaneval_add_example(), "return a + b")


def test_humaneval_extracts_full_function_from_prose_and_fences():
    prediction = "Here is the implementation:\n```python\ndef add(a, b):\n    return a + b\n```"

    assert _extract_python_code(prediction, "add").startswith("def add")
    assert score_prediction(_humaneval_add_example(), prediction)


def test_humaneval_rejects_broken_full_functions():
    prediction = "```python\ndef add(a, b):\n    return a - b\n```"

    assert not score_prediction(_humaneval_add_example(), prediction)


def test_mbpp_prompt_exposes_required_interface_through_tests():
    prompt = _mbpp_prompt({
        "prompt": "Write a function that sorts a matrix by row sum.",
        "test_imports": ["from copy import deepcopy"],
        "test_list": ["assert sort_matrix([[2], [1]]) == [[1], [2]]"],
    })

    assert "from copy import deepcopy" in prompt
    assert "assert sort_matrix(" in prompt
    assert "exact function name and number of positional arguments" in prompt
    assert "remove/keep, first/last/all" in prompt
    assert "Silently check the implementation against every shown assertion" in prompt
    assert prompt.endswith("Do not write any text outside that block.")


def test_mbpp_prompt_preserves_the_dataset_description_verbatim():
    description = "Write a function to check if the given number is woodball or not."

    prompt = _mbpp_prompt({"prompt": description, "test_list": []})

    assert prompt.startswith(description + "\n\n")


def test_mbpp_executes_fenced_code_with_official_test_imports():
    example = BenchmarkExample(
        "mbpp",
        "0",
        "prompt",
        "",
        "code",
        {
            "test_imports": ["from math import sqrt"],
            "test_list": ["assert root(9) == 3"],
        },
    )
    prediction = "Here is the implementation:\n```python\ndef root(value):\n    return sqrt(value)\n```"

    assert score_prediction(example, prediction)


def test_mbpp_ignores_a_standalone_trailing_fence_without_an_opening_fence():
    example = BenchmarkExample(
        "mbpp",
        "0",
        "prompt",
        "",
        "code",
        {"test_list": ["assert square_perimeter(5) == 20"]},
    )
    prediction = "def square_perimeter(side):\n    return 4 * side\n```"

    assert _extract_python_code(prediction) == "def square_perimeter(side):\n    return 4 * side"
    assert score_prediction(example, prediction)


def test_generation_settings_use_corruption_specific_overrides():
    config = {
        "generation": {"temperature": 0.7, "top_k": 20},
        "generation_by_corruption": {
            "structured": {"temperature": 0.9},
            "mask_only": {"temperature": 0.5, "noise_level": 0.8},
        },
    }

    structured = resolve_generation_settings(config, "open_ended", "structured")
    masked = resolve_generation_settings(config, "open_ended", "mask_only")

    assert structured["temperature"] == 0.9
    assert masked["temperature"] == 0.5
    assert masked["noise_level"] == 1.0
    assert masked["permanent_unmask"] is True
    assert masked["confidence_guided"] is True


def test_legacy_generation_settings_fall_back_to_structured_settings():
    config = {"generation": {"temperature": 0.7}, "generation_by_corruption": {"structured": {"temperature": 0.9}}}

    settings = resolve_generation_settings(config, "open_ended", "legacy")

    assert settings["temperature"] == 0.9


def test_autoregressive_generation_has_independent_task_budgets():
    config = {
        "generation": {"max_new_tokens": 128},
        "task_generation": {"mmlu_pro": {"max_new_tokens": 256}},
        "autoregressive_generation": {"max_new_tokens": 512},
        "autoregressive_task_generation": {"mmlu_pro": {"max_new_tokens": 2048}},
    }

    assert resolve_autoregressive_generation_settings(config, "mmlu_pro")["max_new_tokens"] == 2048
    assert resolve_autoregressive_generation_settings(config, "mmlu")["max_new_tokens"] == 512
    assert resolve_autoregressive_generation_settings({}, "mmlu_pro")["max_new_tokens"] == 256


def test_llada_uses_published_task_specific_inference_settings():
    gsm8k = resolve_llada_generation_settings({"generation": {"temperature": 0.7}}, "gsm8k")
    math = resolve_llada_generation_settings({}, "math")
    humaneval = resolve_llada_generation_settings({}, "humaneval")

    assert gsm8k["sampler"] == "llada_official"
    assert gsm8k["max_new_tokens"] == gsm8k["num_steps"] == gsm8k["block_length"] == 512
    assert gsm8k["temperature"] == 0.0
    assert gsm8k["confidence_eos_eot_inf"] is True
    assert gsm8k["eot_token_id"] == 126348
    assert gsm8k["proportional_unmask"] is False
    assert math["max_new_tokens"] == math["num_steps"] == math["block_length"] == 512
    assert humaneval["logits_eos_inf"] is True
    assert humaneval["confidence_eos_eot_inf"] is False


def test_mask_only_adapters_use_the_same_official_task_sampler():
    gsm8k = resolve_mask_only_generation_settings({}, "gsm8k")
    math = resolve_mask_only_generation_settings({}, "math")
    overridden = resolve_mask_only_generation_settings(
        {"mask_only_task_generation": {"gsm8k": {"block_length": 8}}}, "gsm8k"
    )

    assert gsm8k["sampler"] == "llada_official"
    assert gsm8k["max_new_tokens"] == gsm8k["num_steps"] == gsm8k["block_length"] == 512
    assert gsm8k["temperature"] == 0.0
    assert gsm8k["confidence_eos_eot_inf"] is True
    assert gsm8k["proportional_unmask"] is False
    assert math["max_new_tokens"] == math["num_steps"] == math["block_length"] == 512
    assert overridden["block_length"] == 8


def test_global_diffusion_sampler_switch_applies_to_every_model_family():
    config = {
        "diffusion_sampler": "denoise_stream",
        "generation": {
            "temperature": 0.6,
            "top_k": 7,
            "proportional_unmask": True,
            "permanent_unmask": False,
            "confidence_guided": True,
        },
    }

    structured = resolve_generation_settings(config, "open_ended", "structured")
    masked = resolve_mask_only_generation_settings(config, "open_ended")
    llada = resolve_llada_generation_settings(config, "open_ended")

    for settings in (structured, masked, llada):
        assert settings["sampler"] == "denoise_stream"
        assert settings["temperature"] == 0.6
        assert settings["top_k"] == 7
        assert settings["proportional_unmask"] is True
        assert settings["permanent_unmask"] is False
        assert settings["confidence_guided"] is True


def test_unknown_global_diffusion_sampler_is_rejected():
    with pytest.raises(ValueError, match="diffusion sampler"):
        resolve_mask_only_generation_settings(
            {"diffusion_sampler": "not-a-sampler"}, "open_ended"
        )
