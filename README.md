# BYOD: Build Your Own DLM

Convert a pretrained instruction model into a bidirectional masked-diffusion
language model on one GPU. The same training entry point supports:

- Llama 3.1 8B Instruct
- Gemma 2 9B IT
- Qwen2.5 7B Instruct
- Ministral 8B Instruct 2410

[![Open the training notebook in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RuurdKuiper/BYOD/blob/main/notebooks/train_diffusion.ipynb)

The notebook is the recommended starting point: select a model, choose a run
mode, and watch the training and validation losses update while training runs.
Its source is also available at
[`notebooks/train_diffusion.ipynb`](notebooks/train_diffusion.ipynb).

Try the converted models directly in the full-precision demos:
[Gemma](https://huggingface.co/spaces/Dovelove/byod-gemma-2-9b),
[Llama](https://huggingface.co/spaces/Dovelove/byod-llama-3.1-8b),
[Qwen](https://huggingface.co/spaces/Dovelove/byod-qwen2.5-7b), and
[Ministral](https://huggingface.co/spaces/Dovelove/byod-ministral-8b).

## Two run modes

| Mode | Purpose | Important settings |
|---|---|---|
| `quick` | Train the full recipe with a smaller backbone | 25,000 updates, rank-128 LoRA, smaller base model |
| `paper` | Run the reported conversion recipe | 25,000 updates, rank-1024 LoRA, batch 16, no quantization |

Quick mode uses Llama 3.2 1B, Gemma 3 1B, or Qwen2.5 1.5B in full FP16.
There is no comparable official text-only ~1B Ministral checkpoint, so
Ministral is intentionally not offered in quick mode. The original Ministral
8B checkpoint remains available for paper reproduction.

| Family | Quick model | Paper reproduction |
|---|---|---|
| Llama | Llama 3.2 1B Instruct | Llama 3.1 8B Instruct |
| Gemma | Gemma 3 1B IT | Gemma 2 9B IT |
| Qwen | Qwen2.5 1.5B Instruct | Qwen2.5 7B Instruct |
| Ministral | — | Ministral 8B Instruct 2410 |

`paper` implements the reported recipe, including the author-confirmed 0.9/0.1
answer/padding loss. Model-specific presets cover all four families. The saved
Llama reference does not record that weighting; see
[configuration provenance](REPRODUCIBILITY.md) for the documented correction. It requires a
high-memory NVIDIA GPU; the original runs used one 96 GB GPU. `quick` is an
accessible full-length conversion, but it is not expected to reproduce the
paper scores because it changes both the backbone and adapter rank. All other
objective, data, validation, and optimization settings are retained; it uses
FP16 instead of BF16/FP8 for compatibility with common Colab GPUs.
Here, “quick” refers to the smaller memory requirement, not a shortened run.

## Notebook setup

The default notebook install uses `.[colab]`: the core training dependencies and
notebook widgets. It reuses compatible installed packages and shows progress and
elapsed time for each setup stage. It does not request Transformer Engine,
bitsandbytes, or TorchAO. Quick mode uses FP16 with unquantized weights and AdamW,
so none of those optional acceleration packages are needed.

After choosing the run mode, the notebook's separate FP8 setup cell installs
Transformer Engine only for paper mode on a supported native-FP8 GPU. Its PyTorch
extension may need a lengthy CUDA compilation. The paper configuration retains
the documented BF16 fallback on other GPUs; it is not silently changed to disable
FP8 on capable hardware. For a local paper run requiring FP8, install it with:

```bash
python -m pip install --no-build-isolation 'transformer-engine[pytorch]==2.13.0'
```

The `fp8` extra declares the same dependency. The separate `cuda` extra supplies
bitsandbytes for optional 4-bit inference or 8-bit optimizers, neither of which
is used by the shipped training presets.

Old preinstalled TorchAO versions below 0.16 can cause PEFT imports to fail even
without quantization. Setup removes only those known-incompatible versions and
leaves absent or newer installations alone. The checkout's root and `src` paths
are added to the running kernel explicitly, so editable installation does not
require restarting the kernel just to discover `diffusion_lm`.

Faster installation does not remove model/data download time or shorten the
25,000-update training schedule. For a brief CLI smoke run, use
`python train.py --model llama --mode quick --max-updates 2`.

## Local use

```bash
git clone https://github.com/RuurdKuiper/BYOD.git
cd BYOD
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[colab]'
cp .env.example .env  # then add a read-only Hugging Face token
```

Run the smaller-model configuration:

```bash
python train.py --model llama --mode quick
```

Run the paper configuration:

```bash
python train.py --model llama --mode paper
```

Inspect the configuration without loading a model or starting training:

```bash
python train.py --model llama --mode paper --print-config
```

For quick mode, replace `llama` with `gemma` or `qwen`. Paper mode additionally
accepts `ministral`. Outputs contain the resolved configuration, metrics,
inference-ready checkpoints, and the adapter with the best validation loss
under `best/`.

To fold that adapter into a standalone model of the original base-model size:

```bash
python merge_adapter.py outputs/llama-3.1-8b-mask/best merged/llama-3.1-8b-mask
```

## What the paper configuration does

The source prompt remains visible and only response/EOS-padding positions are
eligible for masking. At every presentation of an example, an IID mask ratio
is sampled and a fresh corrupted response is constructed. Loss is calculated
only at corrupted answer and eligible padding positions, with separately
normalized region losses weighted 0.9/0.1. A rank-1024 LoRA is
trained on the query and value projections; base weights and normalization
layers remain frozen.

The exact common recipe is in [`configs/paper.yaml`](configs/paper.yaml), and
the four pinned model revisions are in
[`configs/models.yaml`](configs/models.yaml). The training dataset is pinned to
revision `6041513b684d6ca627879ffdb547b8629bb8b8de` of
[`Ruurd/LAD-training-1m-256`](https://huggingface.co/datasets/Ruurd/LAD-training-1m-256).
The saved configuration from the original Llama run is retained in
[`reference/`](reference/) for auditing.

Llama, Gemma, and Ministral are gated on Hugging Face. Accept their model
licenses and use a read-only token. Never commit that token.

## Reproducibility notes

- Dataset, tokenizer, and base-model revisions are pinned.
- Training data is shuffled deterministically with seed 42.
- Training corruption is resampled online; held-out corruption is deterministic.
- Every run writes `resolved_config.json`, `metrics.jsonl`, and test metrics.
- CUDA kernels and hardware can still introduce small numerical differences.

Run the lightweight repository checks with:

```bash
pip install -e '.[test]'
pytest -q
```


## Benchmark evaluation

```bash
pip install -e '.[evaluation]'
python evaluate_benchmarks.py --config configs/evaluation.yaml
python evaluate_benchmarks.py --config configs/open_ended.yaml
```

Edit `models` in each configuration to select trained adapters and optionally
hosted LLaDA. The nine-task profile uses 125 examples per task and evaluates the
unchanged AR parent. The open-ended profile uses 100 bundled prompts, 128-token
answers, and 64 denoising steps, then scores with Phi-4. Its 4-bit scorer requires
CUDA and `bitsandbytes` (install with `pip install -e '.[cuda]'`). GPQA and gated parent
models require the corresponding Hugging Face access.

HumanEval and MBPP scoring executes generated Python in subprocesses. Run those
benchmarks in a disposable environment without credentials or sensitive files.

To construct a new mixture locally:

```bash
python build_training_dataset.py --no-upload --output-dir data/new-mixture
```

The builder defaults to a new 500k-example, 512-token mixture; it does not recreate
the pinned one-million-example, 256-token dataset byte for byte. Use the pinned
dataset in `configs/paper.yaml` for the reported training profile.

## Release scope

This repository includes training, denoising inference, adapter merging, dataset
construction, and evaluation code. It does not bundle model weights, datasets,
or complete original experiment logs. The public demos above provide immediate
access to the reported conversions. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for provenance and validation limits.

Tests use synthetic examples and tiny local models without downloading the
7B–9B checkpoints. They do not demonstrate a complete GPU training reproduction.

Run `python prepare_release.py` to export a clean checkout and source ZIP under
`dist/`. Publishing instructions are in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).

## License and citation

Original software is released under the [MIT License](LICENSE). Parent models,
datasets, and dependencies retain their own licenses and access conditions.
See [CITATION.cff](CITATION.cff) for citation metadata.
