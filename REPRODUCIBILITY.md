# Reproducibility notes

The paper preset uses separately normalized answer and EOS-padding losses,
weighted 0.9/0.1, as confirmed by the authors. The retained historical Llama
configuration in `reference/` predates the explicit weighting setting and is
kept unchanged as provenance. Complete configurations and logs for every
reported checkpoint are not available.

Dataset, tokenizer, and model revisions are pinned in `configs/`. Published
token profiles preserve the seed-42 training order and original example indices.
Training corruption is resampled online; held-out corruption is deterministic.
Each run saves its resolved configuration and metrics. Upstream dataset terms,
including non-commercial restrictions on Alpaca-derived rows, still apply.

`configs/evaluation.yaml` uses nine tasks with 125 examples each and paired
AR-parent evaluation. `configs/open_ended.yaml` uses 100 bundled prompts,
128-token answers, and 64 denoising steps. Training diagnostics use the original
20-prompt, 64-token, 32-step settings.

Tests cover synthetic data and tiny local models. Full GPU conversion and exact
regeneration of published scores have not been verified by these tests.
Dependency ranges are compatibility requirements, not the original environment's
lock file. Quick mode changes the backbone, adapter rank, and precision.
