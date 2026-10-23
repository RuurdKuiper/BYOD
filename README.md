# BYOD: Build Your Own DLM

Convert an instruction model into a bidirectional masked-diffusion language
model on one GPU.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RuurdKuiper/BYOD/blob/main/notebooks/train_diffusion.ipynb)

Start with the notebook to select a model, train, and watch the loss curves.
Step 6 opens an interactive inference page with generation settings and live
denoising updates, including from a saved interrupted run. Setup reuses Colab’s
PyTorch/CUDA packages and installs Gradio only when you run step 6. For a local
inference web app, install `pip install -e ".[demo]"`.
Try the trained models in the hosted demos:
[Llama](https://huggingface.co/spaces/Dovelove/byod-llama-3.1-8b),
[Gemma](https://huggingface.co/spaces/Dovelove/byod-gemma-2-9b),
[Qwen](https://huggingface.co/spaces/Dovelove/byod-qwen2.5-7b),
[Ministral](https://huggingface.co/spaces/Dovelove/byod-ministral-8b).

## Train locally

```bash
pip install -e .
cp .env.example .env  # add your Hugging Face read token
python train.py --model llama --mode quick
```

Accept the parent model's Hugging Face terms before downloading gated models.
Training requires an NVIDIA GPU. Use `--max-updates 2` for a short smoke run.

| Model | `quick` | `paper` |
| --- | --- | --- |
| `llama` | Llama 3.2 1B Instruct | Llama 3.1 8B Instruct |
| `gemma` | Gemma 3 1B IT | Gemma 2 9B IT |
| `qwen` | Qwen2.5 1.5B Instruct | Qwen2.5 7B Instruct |
| `ministral` | — | Ministral 8B Instruct 2410 |

Both modes default to 25,000 updates. Quick mode uses smaller models and
rank-128 LoRA; paper mode uses rank-1024 LoRA and needs a high-memory GPU
(the original runs used 96 GB). Quick mode does not reproduce the paper scores.
This version uses FP16/BF16 training without FP8.

Training downloads the matching [pretokenized dataset](https://huggingface.co/datasets/Ruurd/BYOD-tokenized-256)
automatically. Each profile covers the default 400,000-example training budget.
Random masking is sampled online, with answer/padding losses weighted 0.9/0.1.
See [reproducibility notes](REPRODUCIBILITY.md) for provenance and validation limits.

Runs evaluate at each checkpoint and keep only `best/` under `outputs/`.
Stopping training manually also saves `interrupted/`.
Merge an adapter into its base model with:

```bash
python merge_adapter.py outputs/llama-3.1-8b-mask/best merged/llama
```

## Change settings or data

Edit [`configs/paper.yaml`](configs/paper.yaml) for shared hyperparameters,
[`configs/quick.yaml`](configs/quick.yaml) for quick-mode overrides, and
[`configs/models.yaml`](configs/models.yaml) for model choices. To override a
preset without editing it, copy and edit [`configs/custom.example.yaml`](configs/custom.example.yaml):

```bash
python train.py --model llama --mode quick --config configs/my_run.yaml
```

In Colab, enter the file path in **Config YAML**. Overrides apply after the
selected preset; CLI flags take precedence. Set `dataset_name` to your Hugging
Face dataset; it needs text columns `instruction` and `output` (`input` is optional).
Use `dataset_columns: {instruction: question, output: answer}` for other names.
Missing validation/test splits are reserved from train using seed 42. Custom
data is tokenized once using the selected model and cached, including on Drive
when enabled. Random training masking is still sampled online.

## Evaluate

Set your adapter paths in the evaluation configs, then run:

```bash
pip install -e '.[evaluation]'
python evaluate_benchmarks.py --config configs/evaluation.yaml
python evaluate_benchmarks.py --config configs/open_ended.yaml
```

The open-ended profile's 4-bit Phi-4 scorer also needs `pip install -e '.[cuda]'`.
Code benchmarks execute generated Python; use a disposable environment.

## Development

Core code is in [`src/diffusion_lm/`](src/diffusion_lm/): `training.py` runs
optimization, `data.py` and `corruption.py` handle batches and masking, `loss.py`
defines the objective, and `inference.py` implements denoising generation.

```bash
pip install -e '.[test]'
pytest -q
```

[MIT license](LICENSE) for the software; upstream model and dataset terms apply.
Citation metadata is in [CITATION.cff](CITATION.cff).
