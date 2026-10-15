# Recipe provenance and release scope

The `paper` preset implements the manuscript's separately normalized answer and
EOS-padding losses, combined with weights 0.9/0.1. The author confirmed this
objective for the reported checkpoints during preprint preparation on 2026-10-15.

## Historical configuration

`reference/llama-3.1-8b-mask.resolved_config.json` is the existing saved Llama
configuration. It does not record `answer_padding_loss`; the previous release
preset explicitly disabled separate weighting. That preset has been corrected
to the author-confirmed objective. The historical file is retained as evidence,
not rewritten to imply it contains that confirmation. Its absolute Colab output
path has been replaced with portable `outputs/llama-3.1-8b-mask_1`.

All other available historical settings are retained. The Llama preset uses
padding multiple 8 and the other families use 16, as recorded in the model
presets; the preprint's training table now states that distinction. Model and
dataset revisions are pinned. Complete checkpoint-specific configurations for
every reported run are not bundled in this release.

## Data and evaluation

Training uses `Ruurd/LAD-training-1m-256` at revision
`6041513b684d6ca627879ffdb547b8629bb8b8de`. The manuscript also links the public
`Dovelove/BYOD-training-1m-256` mirror. Public LFS SHA-256 hashes were checked on 2026-10-15: all five train,
validation, and test Parquet files match the pinned original byte for byte. The
100 published open-ended questions also match the bundled prompts in order.
Revisions and hashes are recorded in `reference/public_artifacts.json`. Upstream
dataset licenses still apply.

`configs/evaluation.yaml` provides the nine-task, 125-example paired evaluation;
`configs/open_ended.yaml` provides the 100-question, 128-token, 64-step profile.
The fixed prompts are bundled as package data. The harness writes predictions
and summaries under `outputs/benchmark_runs`. Training-time diagnostics retain
the historical 20-prompt, 64-token, 32-step settings, which differ from the final
evaluation profile.

The builder constructs a new mixture under its documented defaults, rather than
reconstructing the historical dataset byte for byte. The release does not
include the original adapters, complete experiment logs, or all checkpoint
configurations. Fresh training and evaluation are supported; exact regeneration
of every published number has not been demonstrated. Dependency ranges specify
compatibility, not a recovered lock file from the original GPU environment.

## Validation limits

Repository tests use synthetic data and tiny local models. Full conversion, FP8
execution, gated-model access, and GPU memory requirements still require checks
on suitable CUDA hardware. Tests cannot verify the author-confirmed provenance
of the original reported checkpoints.
