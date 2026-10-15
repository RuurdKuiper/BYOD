#!/usr/bin/env python
"""Evaluate selected saved adapters: python evaluate_benchmarks.py --config ..."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import yaml
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Prefer this checkout over a stale non-editable package in site-packages when
# the script is run directly in Colab.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diffusion_lm.benchmarks import ALL_TASKS, BIDIRECTIONAL_INFILLING_TASK, BenchmarkRunReporter, extract_answer, load_benchmark, resolve_autoregressive_generation_settings, resolve_generation_settings, resolve_llada_generation_settings, resolve_mask_only_generation_settings, score_prediction, score_texts_with_model
from diffusion_lm.infilling import score_bidirectional_example, summarize_infilling
from diffusion_lm.metrics import distinct_n
from diffusion_lm.inference import denoise_stream, find_adapters, llada_generate, load_hosted_legacy_session, load_llada_session, load_local_legacy_session, load_merged_session, load_session, release_session, select_device
from diffusion_lm.judging import judge_open_ended_groups

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_PERPLEXITY_MODEL = "microsoft/phi-4"


def _generate_ar(session, prompt: str, max_new_tokens: int, original_base: bool = True) -> str:
    """Generate one autoregressive answer from a loaded adapter."""
    from diffusion_lm.data import apply_neutral_chat_template
    prefix = apply_neutral_chat_template(session.tokenizer, [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
    if isinstance(prefix, str):
        prefix = session.tokenizer.encode(prefix, add_special_tokens=False)
    elif hasattr(prefix, "input_ids"):
        prefix = prefix.input_ids
    if prefix and isinstance(prefix[0], list):
        prefix = prefix[0]
    import torch
    context = session.model.disable_adapter() if original_base else None
    trained = {name: parameter.detach().cpu().clone() for name, parameter in session.model.named_parameters() if "norm" in name.lower()}
    config = session.model.config
    original_config = {
        name: getattr(config, name)
        for name in ("use_cache", "is_causal", "use_bidirectional_attention")
        if hasattr(config, name)
    }
    try:
        # load_session prepares this shared model instance for denoising. Put
        # it back in causal generation mode for the untouched base baseline.
        config.use_cache = True
        if hasattr(config, "is_causal"):
            config.is_causal = True
        if hasattr(config, "use_bidirectional_attention"):
            config.use_bidirectional_attention = False
        with torch.inference_mode(), (context if context is not None else _null_context()):
            if original_base:
                initial_path = session.adapter_path / "normalization_initial_state.pt"
                if initial_path.is_file():
                    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
                    named = dict(session.model.named_parameters())
                    for name, value in initial.items():
                        if name in named:
                            named[name].data.copy_(value.to(named[name].device, dtype=named[name].dtype))
            input_ids = torch.tensor([prefix], device=session.device)
            attention_mask = torch.ones_like(input_ids)
            pad_token_id = session.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = session.tokenizer.eos_token_id
            generated = session.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_token_id,
            )
    finally:
        for name, value in original_config.items():
            setattr(config, name, value)
        named = dict(session.model.named_parameters())
        for name, value in trained.items():
            if name in named:
                named[name].data.copy_(value.to(named[name].device, dtype=named[name].dtype))
    return session.tokenizer.decode(generated[0, len(prefix):], skip_special_tokens=True).strip()


class _null_context:
    """Minimal no-op context manager for optional adapter disabling."""
    def __enter__(self):
        """Enter without modifying model state."""
        return self
    def __exit__(self, *args):
        """Leave without suppressing exceptions."""
        return False


def _autoregressive_baseline_key(
    run_config: dict,
    task: str,
    settings: dict,
    fallback_model: str,
) -> tuple[str, str, str, str, str]:
    """Identify one reusable AR baseline within a benchmark invocation.

    Multiple diffusion adapters may share an unchanged parent.  Their AR
    comparison is identical when the parent, tokenizer, revision, task, and
    generation settings match, so it should be generated only once.
    """
    base_model = str(run_config.get("model_name_or_path", run_config.get("repo_id", fallback_model)))
    tokenizer = str(run_config.get("tokenizer_name_or_path", base_model))
    revision = str(run_config.get("model_revision", run_config.get("revision", "")))
    serialized_settings = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)
    return base_model, tokenizer, revision, str(task), serialized_settings


def _generate_diffusion(session, prompt: str, settings: dict, mode: str) -> str:
    """Generate one pure-diffusion answer using the run's corruption strategy."""
    sampler = str(settings.get("sampler", "denoise_stream"))
    if sampler == "llada_official":
        return llada_generate(
            session,
            prompt,
            gen_length=int(settings.get("max_new_tokens", 128)),
            steps=int(settings.get("num_steps", settings.get("max_new_tokens", 128))),
            block_length=int(settings.get("block_length", settings.get("max_new_tokens", 128))),
            temperature=float(settings.get("temperature", 0.0)),
            cfg_scale=float(settings.get("cfg_scale", 0.0)),
            remasking=str(settings.get("remasking", "low_confidence")),
            logits_eos_inf=bool(settings.get("logits_eos_inf", False)),
            confidence_eos_eot_inf=bool(settings.get("confidence_eos_eot_inf", False)),
            eot_token_id=settings.get("eot_token_id"),
            system_prompt=str(settings.get("system_prompt", "")),
            seed=int(settings.get("seed", 1234)),
            repetition_penalty=float(settings.get("repetition_penalty", 1.0)),
        )
    if sampler != "denoise_stream":
        raise ValueError(f"Unknown diffusion sampler: {sampler!r}")
    structured = mode in {"structured", "legacy"}
    # Mask-only (LLaDA-style) evaluation always begins with the answer fully
    # masked; configured noise levels remain applicable to structured runs.
    noise_level = 1.0 if mode == "mask_only" else float(settings.get("noise_level", .5))
    final = ""
    for final, _status, _html in denoise_stream(
        session,
        prompt,
        settings.get("system_prompt", "You are a helpful assistant."),
        int(settings.get("max_new_tokens", 256)),
        int(settings.get("num_steps", settings.get("max_new_tokens", 256))),
        noise_level,
        float(settings.get("temperature", .7)),
        int(settings.get("top_k", 20)),
        int(settings.get("seed", 1234)),
        permanent_unmask=bool(settings.get("permanent_unmask", structured)),
        confidence_guided=bool(settings.get("confidence_guided", structured)),
        proportional_unmask=bool(settings.get("proportional_unmask", True)),
        early_stopping=bool(settings.get("early_stopping", False)),
        confidence_eos_eot_inf=bool(settings.get("confidence_eos_eot_inf", False)),
        freeze_retained_tokens=bool(settings.get("freeze_retained_tokens", True)),
        repetition_penalty=float(settings.get("repetition_penalty", 1.0)),
    ):
        pass
    return final


def _native_eot_token_id(tokenizer) -> int | None:
    """Find the model-native end-of-turn token used for delayed transfer."""
    unknown = getattr(tokenizer, "unk_token_id", None)
    vocabulary_size = len(tokenizer)
    for token in ("<|eot_id|>", "<end_of_turn>", "<|end_of_turn|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != unknown and 0 <= int(token_id) < vocabulary_size:
            return int(token_id)
    return None


def _show_open_ended_answer(progress, method: str, index: int, total: int, prompt: str, answer: str) -> None:
    """Print one completed open-ended generation without disrupting tqdm."""
    progress.write(
        f"\n[{method} {index}/{total}] Prompt:\n{prompt}\n"
        f"Final answer:\n{answer or '[empty response]'}\n"
    )
    print(f"[{method} {index}/{total}] sample complete", flush=True)


def _save_infilling_results(reporter, model_label: str, mode: str, examples, results, protocol: dict) -> dict:
    """Write per-example and aggregate records for the right-context copy test."""
    for example, result in zip(examples, results):
        reporter.save_result({
            "model": model_label,
            "corruption_mode": mode,
            "task": BIDIRECTIONAL_INFILLING_TASK,
            "example_id": example.example_id,
            "method": "diffusion",
            "prompt": example.prompt,
            "answer_prefix": example.metadata["answer_prefix"],
            "answer_suffix": example.metadata["answer_suffix"],
            "inference_settings": protocol,
            **result,
        })
    summary = {
        "model": model_label,
        "corruption_mode": mode,
        "task": BIDIRECTIONAL_INFILLING_TASK,
        "method": "diffusion",
        "inference_settings": protocol,
        **summarize_infilling(results),
    }
    reporter.save_summary(summary)
    return summary


def _load_perplexity_reference(config: dict):
    """Load the single model used for all benchmark perplexity scores."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = dict(config.get("perplexity", {}))
    model_name = settings.get("model_name_or_path", DEFAULT_PERPLEXITY_MODEL)
    tokenizer_name = settings.get("tokenizer_name_or_path", model_name)
    device = select_device(settings.get("device", config.get("device", "auto")))
    cache_dir = settings.get("cache_dir", "base_models")
    token = os.getenv("HF_TOKEN")
    precision = str(settings.get("precision", "fp16" if device.type == "cuda" else "fp32")).lower()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision == "fp16" and device.type != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=token, cache_dir=cache_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {"torch_dtype": dtype, "token": token, "cache_dir": cache_dir}
    if str(settings.get("quantization", "none")).lower() in {"4bit", "4-bit", "qlora"}:
        if device.type != "cuda":
            raise RuntimeError("Perplexity 4-bit quantization requires CUDA.")
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(settings.get("quantization_type", "nf4")),
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = {"": device.index if device.index is not None else 0}
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if "device_map" not in load_kwargs:
        model.to(device)
    model.eval()
    return model, tokenizer, device, {"model_name_or_path": model_name, "tokenizer_name_or_path": tokenizer_name, "device": str(device), "quantization": str(settings.get("quantization", "none")), "precision": precision}


def main() -> None:
    """Load configured benchmark data and evaluate selected best adapters."""
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    outputs = Path(config.get("outputs_dir", "outputs"))
    selections = config.get("models") or find_adapters(outputs)
    if not selections:
        raise SystemExit("No adapter directories found. Train at least one model to create outputs/<run>/best/.")
    if config.get("tasks") is None:
        config["tasks"] = ALL_TASKS
    run_diffusion = bool(config.get("run_diffusion", True))
    include_autoregressive = bool(config.get("include_autoregressive", False))
    if not run_diffusion and not include_autoregressive:
        raise ValueError("At least one of run_diffusion or include_autoregressive must be enabled")
    token = os.getenv("HF_TOKEN")
    cache = config.get("cache_dir", "data/huggingface")
    if "results_path" in config and "results_dir" not in config:
        legacy_path = Path(config["results_path"])
        results_dir = legacy_path.parent / "benchmark_runs"
        print(f"results_path is deprecated; this run will be isolated under {results_dir}", flush=True)
    else:
        results_dir = Path(config.get("results_dir", "outputs/benchmark_runs"))
    reporter = BenchmarkRunReporter(results_dir, config, config.get("run_name"))
    print(f"Benchmark run: {reporter.run_id}\nResults: {reporter.path}", flush=True)
    show_open_ended_answers = bool(config.get("show_open_ended_answers", False))
    open_ended_pending = []
    completed_autoregressive: dict[tuple[str, str, str, str, str], str] = {}
    for selection in selections:
        selection = str(selection)
        if selection.startswith("llada:"):
            model_name = selection.split(":", 1)[1].strip() or "GSAI-ML/LLaDA-8B-Instruct"
            model_label = f"llada:{model_name}"
            run_config = {"model_source": "llada", "repo_id": model_name}
            mode = "mask_only"
            session = load_llada_session(model_name, config.get("device", "auto"))
            supports_autoregressive = False
        elif selection.startswith("legacy:"):
            checkpoint = selection.split(":", 1)[1].strip()
            if not checkpoint:
                raise ValueError("Legacy model entries must use legacy:/path/to/checkpoint.pth")
            tokenizer_name = config.get("legacy_tokenizer_name_or_path", "meta-llama/Llama-3.2-3B")
            model_label = f"legacy:{checkpoint}"
            run_config = {"model_source": "local_legacy", "checkpoint": checkpoint, "tokenizer_name_or_path": tokenizer_name}
            mode = "structured"
            session = load_local_legacy_session(checkpoint, tokenizer_name, config.get("device", "auto"))
            supports_autoregressive = False
        elif selection.startswith("legacy-hf:"):
            descriptor = selection.split(":", 1)[1].strip()
            try:
                repo_id, filename = [part.strip() for part in descriptor.split("|", 1)]
            except ValueError as exc:
                raise ValueError("Hosted legacy entries must use legacy-hf:repo_id|filename.pth") from exc
            tokenizer_name = config.get("legacy_tokenizer_name_or_path", "meta-llama/Llama-3.2-3B")
            model_label = f"legacy-hf:{repo_id}/{filename}"
            run_config = {"model_source": "huggingface_legacy", "repo_id": repo_id, "filename": filename, "tokenizer_name_or_path": tokenizer_name}
            mode = "structured"
            session = load_hosted_legacy_session(repo_id, filename, tokenizer_name, config.get("device", "auto"))
            supports_autoregressive = False
        elif selection.startswith("merged:"):
            merged_descriptor = selection.split(":", 1)[1].strip()
            if not merged_descriptor:
                raise ValueError("Merged model entries must use merged:/path/to/model")
            merged_path = Path(merged_descriptor).expanduser().resolve()
            saved_config_path = merged_path / "lad_run_config.json"
            if saved_config_path.is_file():
                run_config = json.loads(saved_config_path.read_text())
            else:
                # Merges created before lad_run_config.json was introduced can
                # recover it from the adapter path recorded in the report.
                report_path = merged_path / "lad_merge_report.json"
                report = json.loads(report_path.read_text()) if report_path.is_file() else {}
                source_adapter = Path(report.get("adapter_path", ""))
                source_config_path = source_adapter.parent / "resolved_config.json"
                run_config = json.loads(source_config_path.read_text()) if source_config_path.is_file() else {}
            model_label = f"merged:{merged_path}"
            mode = run_config.get("corruption_mode", "mask_only")
            session = load_merged_session(
                merged_path,
                config.get("device", "auto"),
                config.get("quantization"),
                run_config,
            )
            # A merged model cannot disable LoRA to recover the original base.
            supports_autoregressive = False
        else:
            model_label = selection
            adapter_path = outputs / selection
            run_config_path = adapter_path.parent / "resolved_config.json"
            run_config = json.loads(run_config_path.read_text())
            mode = run_config.get("corruption_mode", "mask_only")
            session = load_session(selection, outputs, config.get("device", "auto"), config.get("quantization"))
            supports_autoregressive = True
        ar_model_name = str(run_config.get("model_name_or_path", run_config.get("repo_id", model_label)))
        if not run_diffusion and not supports_autoregressive:
            print(f"Skipping {model_label}: this selection has no recoverable autoregressive parent.", flush=True)
            release_session(session)
            del session
            continue
        for task in config["tasks"]:
            examples = load_benchmark(task, config.get("split", "test"), config.get("limit"), cache, token, config.get("limit_fraction"))
            if session.llada:
                task_settings = resolve_llada_generation_settings(config, task)
            elif mode == "mask_only":
                task_settings = resolve_mask_only_generation_settings(config, task)
            else:
                task_settings = resolve_generation_settings(config, task, mode)
            if task_settings.get("sampler") == "llada_official" and "eot_token_id" not in task_settings:
                task_settings["eot_token_id"] = _native_eot_token_id(session.tokenizer)
            autoregressive_settings = resolve_autoregressive_generation_settings(config, task)
            autoregressive_key = _autoregressive_baseline_key(
                run_config, task, autoregressive_settings, model_label
            )
            autoregressive_already_run = autoregressive_key in completed_autoregressive
            if task == BIDIRECTIONAL_INFILLING_TASK:
                if not run_diffusion:
                    print(f"[{ar_model_name}] {task}: skipped because the test requires bidirectional masked prediction", flush=True)
                    continue
                total = len(examples)
                protocol = {
                    "type": "right_context_copy",
                    "decoding": "single_forward_argmax",
                    "system_prompt": "You are a helpful assistant.",
                    "with_clue": "predict the earlier mask while the copied word is visible later",
                    "without_clue": "predict the same mask after removing the later clue",
                    "seed": None,
                }
                print(f"\n[{model_label}] {task}: {total} deterministic right-context copy samples", flush=True)
                diffusion_results = []
                progress = tqdm(examples, desc=f"{model_label}/{task} bidirectional", unit="sample")
                for index, example in enumerate(progress, start=1):
                    result = score_bidirectional_example(session, example)
                    diffusion_results.append(result)
                    exact = sum(int(item["exact_match"]) for item in diffusion_results)
                    progress.set_postfix(exact=f"{exact}/{index}")
                summary = _save_infilling_results(
                    reporter, model_label, mode, examples, diffusion_results, protocol
                )
                message = (
                    f"{model_label} | {task} | with-clue exact={summary['exact_match']:.4f} "
                    f"| token accuracy={summary['token_accuracy']:.4f} "
                    f"| without-clue exact={summary['without_clue_exact_match']:.4f} "
                    f"| right-context gain={summary['right_context_gain_exact_match']:+.4f} "
                    f"| clue-changed prediction={summary['clue_changed_prediction_rate']:.4f}"
                )
                print(message, flush=True)
                continue
            if task == "open_ended":
                messages = []
                if run_diffusion:
                    print(f"\n[{model_label}] {task}: {len(examples)} validation samples (diffusion)", flush=True)
                    diffusion_texts = []
                    diffusion_progress = tqdm(examples, desc=f"{model_label}/{task} diffusion", unit="sample")
                    for index, example in enumerate(diffusion_progress, start=1):
                        if show_open_ended_answers:
                            print(f"\n[diffusion {index}/{len(examples)}] generating: {example.prompt}", flush=True)
                        text = _generate_diffusion(session, example.prompt, task_settings, mode)
                        diffusion_texts.append(text)
                        if show_open_ended_answers:
                            _show_open_ended_answer(diffusion_progress, "diffusion", index, len(examples), example.prompt, text)
                    diffusion_distinct = [
                        {f"distinct_{n}": distinct_n(text, session.tokenizer, n) for n in (1, 2, 3)}
                        for text in diffusion_texts
                    ]
                    open_ended_pending.append({"model": model_label, "corruption_mode": mode, "task": task, "method": "diffusion", "examples": examples, "texts": diffusion_texts, "distinct": diffusion_distinct, "inference_settings": task_settings})
                    messages.append(f"{model_label} | {task} | diffusion generation complete")
                if include_autoregressive and supports_autoregressive and not autoregressive_already_run:
                    print(f"[{ar_model_name}] {task}: {len(examples)} validation samples (autoregressive base)", flush=True)
                    ar_texts = []
                    ar_progress = tqdm(examples, desc=f"{ar_model_name}/{task} autoregressive", unit="sample")
                    for index, example in enumerate(ar_progress, start=1):
                        if show_open_ended_answers:
                            print(f"\n[autoregressive {index}/{len(examples)}] generating: {example.prompt}", flush=True)
                        text = _generate_ar(session, example.prompt, int(autoregressive_settings["max_new_tokens"]), original_base=True)
                        ar_texts.append(text)
                        if show_open_ended_answers:
                            _show_open_ended_answer(ar_progress, "autoregressive", index, len(examples), example.prompt, text)
                    ar_distinct = [
                        {f"distinct_{n}": distinct_n(text, session.tokenizer, n) for n in (1, 2, 3)}
                        for text in ar_texts
                    ]
                    open_ended_pending.append({"model": ar_model_name, "evaluation_model": ar_model_name, "model_variant": "original_base", "corruption_mode": mode, "task": task, "method": "autoregressive", "examples": examples, "texts": ar_texts, "distinct": ar_distinct, "inference_settings": autoregressive_settings})
                    completed_autoregressive[autoregressive_key] = ar_model_name
                    messages.append(f"{ar_model_name} | {task} | autoregressive generation complete")
                if include_autoregressive and supports_autoregressive and autoregressive_already_run:
                    messages.append(f"duplicate autoregressive baseline skipped (already evaluated as {completed_autoregressive[autoregressive_key]})")
                if include_autoregressive and not supports_autoregressive:
                    messages.append("autoregressive comparison skipped")
                print(" | ".join(messages))
                continue
            correct = 0; ar_correct = 0
            total = len(examples)
            if run_diffusion:
                print(f"\n[{model_label}] {task}: {total} validation samples (diffusion)", flush=True)
                # Complete the diffusion pass before switching to the optional
                # autoregressive baseline, avoiding per-example model switching.
                diffusion_progress = tqdm(examples, desc=f"{model_label}/{task} diffusion", unit="sample")
                for index, example in enumerate(diffusion_progress, start=1):
                    diffusion_text = _generate_diffusion(session, example.prompt, task_settings, mode)
                    diffusion_ok = score_prediction(example, diffusion_text)
                    record = {"model": model_label, "corruption_mode": mode, "task": task, "example_id": example.example_id, "method": "diffusion", "prompt": example.prompt, "prediction": diffusion_text, "target": example.answer, "correct": diffusion_ok, "inference_settings": task_settings}
                    if example.kind != "code":
                        record.update(extracted_prediction=extract_answer(diffusion_text, example.kind, example.answer), extracted_target=extract_answer(example.answer, example.kind))
                    reporter.save_result(record); correct += int(diffusion_ok)
                    diffusion_progress.set_postfix(correct=f"{correct}/{index}", accuracy=f"{correct / index:.3f}")
            if include_autoregressive and supports_autoregressive and not autoregressive_already_run:
                print(f"[{ar_model_name}] {task}: {total} validation samples (autoregressive base)", flush=True)
                ar_progress = tqdm(examples, desc=f"{ar_model_name}/{task} autoregressive", unit="sample")
                for index, example in enumerate(ar_progress, start=1):
                    ar_text = _generate_ar(session, example.prompt, int(autoregressive_settings["max_new_tokens"]), original_base=True)
                    ar_ok = score_prediction(example, ar_text); ar_correct += int(ar_ok)
                    record = {"model": ar_model_name, "evaluation_model": ar_model_name, "model_variant": "original_base", "corruption_mode": mode, "task": task, "example_id": example.example_id, "method": "autoregressive", "prompt": example.prompt, "prediction": ar_text, "target": example.answer, "correct": ar_ok, "inference_settings": autoregressive_settings}
                    if example.kind != "code":
                        record.update(extracted_prediction=extract_answer(ar_text, example.kind, example.answer), extracted_target=extract_answer(example.answer, example.kind))
                    reporter.save_result(record)
                    ar_progress.set_postfix(correct=f"{ar_correct}/{index}", accuracy=f"{ar_correct / index:.3f}")
                completed_autoregressive[autoregressive_key] = ar_model_name
            messages = []
            if run_diffusion:
                summary = {"model": model_label, "corruption_mode": mode, "task": task, "method": "diffusion", "inference_settings": task_settings, "accuracy": correct / max(len(examples), 1), "correct": correct, "total": len(examples)}
                reporter.save_summary(summary)
                messages.append(f"{model_label} | {task} | diffusion accuracy={summary['accuracy']:.4f} ({correct}/{len(examples)})")
            if include_autoregressive:
                if not supports_autoregressive:
                    messages.append("autoregressive comparison skipped")
                    print(" | ".join(messages))
                    continue
                if autoregressive_already_run:
                    messages.append(f"duplicate autoregressive baseline skipped (already evaluated as {completed_autoregressive[autoregressive_key]})")
                    print(" | ".join(messages))
                    continue
                ar_summary = {"model": ar_model_name, "evaluation_model": ar_model_name, "model_variant": "original_base", "corruption_mode": mode, "task": task, "method": "autoregressive", "inference_settings": autoregressive_settings, "accuracy": ar_correct / max(len(examples), 1), "correct": ar_correct, "total": len(examples)}
                reporter.save_summary(ar_summary)
                messages.append(f"{ar_model_name} | {task} | autoregressive accuracy={ar_summary['accuracy']:.4f} ({ar_correct}/{len(examples)})")
            print(" | ".join(messages))
        print(f"Released generation model for {model_label}; clearing GPU memory before the next model.", flush=True)
        release_session(session)
        del session
    # Generation models are no longer needed; release the last session before
    # loading the shared reference model to keep peak GPU memory manageable.
    if open_ended_pending:
        judge_result = None
        judge_settings = dict(config.get("open_ended_judge", {}))
        if judge_settings.get("enabled", False):
            print(
                f"\nBlind-ranking open-ended responses with {judge_settings.get('model', 'gpt-5')}...",
                flush=True,
            )
            judge_result = judge_open_ended_groups(
                open_ended_pending,
                judge_settings,
                on_progress=lambda complete, total: print(
                    f"Judge comparisons: {complete}/{total}", flush=True
                ),
            )
            if judge_result.get("skipped_reason"):
                print(f"Open-ended judge skipped: {judge_result['skipped_reason']}", flush=True)
            else:
                reporter.save_run_records("judge_rankings.jsonl", judge_result["comparisons"])
                reporter.save_run_json("judge_leaderboard.json", {
                    "judge_model": judge_result["judge_model"],
                    "methods": judge_result["methods"],
                    "candidate_count": judge_result["candidate_count"],
                    "successful_comparisons": len(judge_result["comparisons"]),
                    "failed_comparisons": len(judge_result["errors"]),
                    "errors": judge_result["errors"],
                    "leaderboard": judge_result["leaderboard"],
                })
                print("Open-ended judge leaderboard:", flush=True)
                for row in judge_result["leaderboard"]:
                    print(
                        f"  {row['judge_leaderboard_position']}. {row['model']} | {row['method']} "
                        f"| points={row['judge_total_score']} | wins={row['judge_first_place_count']}",
                        flush=True,
                    )
        print("\nLoading shared perplexity reference model...", flush=True)
        reference_model, reference_tokenizer, reference_device, reference_info = _load_perplexity_reference(config)
        print(f"Scoring {len(open_ended_pending)} open-ended result groups with {reference_info['model_name_or_path']}...", flush=True)
        judge_summaries = {
            row["group_index"]: row for row in (judge_result or {}).get("leaderboard", [])
        }
        judge_annotations = (judge_result or {}).get("per_group", {})
        for group_index, group in enumerate(open_ended_pending):
            scores = score_texts_with_model(reference_model, reference_tokenizer, reference_device, group["texts"])
            annotations = judge_annotations.get(group_index, [None] * len(group["examples"]))
            for example, text, per_text, diversity, annotation in zip(group["examples"], group["texts"], scores["per_text"], group["distinct"], annotations):
                record = {
                    "model": group["model"], "corruption_mode": group["corruption_mode"], "task": group["task"],
                    "example_id": example.example_id, "method": group["method"],
                    "prompt": example.prompt, "prediction": text,
                    "inference_settings": group["inference_settings"],
                    "perplexity_reference": reference_info,
                    **per_text,
                    **diversity,
                    **({"evaluation_model": group["evaluation_model"], "model_variant": group["model_variant"]} if "evaluation_model" in group else {}),
                }
                if annotation is not None:
                    record.update(annotation)
                reporter.save_result(record)
            judge_summary = dict(judge_summaries.get(group_index, {}))
            judge_summary.pop("group_index", None)
            summary = {
                "model": group["model"], "corruption_mode": group["corruption_mode"], "task": group["task"],
                "method": group["method"], "total": len(group["examples"]),
                "inference_settings": group["inference_settings"],
                "perplexity_reference": reference_info,
                "perplexity": scores["perplexity"], "mean_perplexity": scores["mean_perplexity"], "median_perplexity": scores["median_perplexity"], "mean_nll": scores["mean_nll"], "tokens": scores["tokens"],
                "mean_distinct_1": sum(item["distinct_1"] for item in group["distinct"]) / max(len(group["distinct"]), 1),
                "mean_distinct_2": sum(item["distinct_2"] for item in group["distinct"]) / max(len(group["distinct"]), 1),
                "mean_distinct_3": sum(item["distinct_3"] for item in group["distinct"]) / max(len(group["distinct"]), 1),
                **judge_summary,
                **({"evaluation_model": group["evaluation_model"], "model_variant": group["model_variant"]} if "evaluation_model" in group else {}),
            }
            reporter.save_summary(summary)
            print(f"{group['model']} | {group['method']} | median perplexity={summary['median_perplexity']:.4f} | mean per-response perplexity={summary['mean_perplexity']:.4f} | pooled perplexity={summary['perplexity']:.4f} | Distinct-1={summary['mean_distinct_1']:.4f}", flush=True)
        del reference_model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    run_path = reporter.complete()
    print(f"\nStructured benchmark report written to {run_path}", flush=True)


if __name__ == "__main__":
    main()
