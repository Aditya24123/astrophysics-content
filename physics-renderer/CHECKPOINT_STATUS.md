> Historical code-only checkpoint from before followup/review/repair. Superseded by [current results](progress/README.md).

# Physics renderer historical checkpoint - October 23, 2026

The H II synthetic simulator and dataset preparation stage is preserved.
The current candidate contains 524 source groups and 1,048 noisy observations:
397/67/48 train/validation/test sources, plus 12 separate challenge sources.
All 3,712 original files passed a fresh integrity recheck; all five readiness
gates passed. The old exposed dataset is not used for training.

The new ML stage uses a small full covariance logit-normal conditional neural
posterior, train-only PCA, source-grouped splits, validation early stopping,
and a reserved final test. The written protocol is `ml/plan_v1.json`.
Numerical gradient, covariance-sampling, support, PCA and forward-equivalence
checks passed. The selected neural checkpoint is epoch 22; validation stopping
ended the fit at epoch 52. All 96 test observations were evaluated once after
freezing the selected model. This test is now an evaluated archive, not an
untouched model-selection resource. Scientific accuracy is not established. Two early launch
failures concerned an overbroad image guard and Windows resource logging;
both were corrected before any test evaluation.

A conventional Bayesian posterior comparator uses the same radio data, priors,
forward physics and nuisance assumptions on two predeclared validation sources.
Both pilots failed the predeclared convergence criteria; their summaries are
provisional, and no method has demonstrated superiority. The last forward-check
console milestone was 44/48; later read-only inspection found 46 saved cases.
Two cases and the overall completion receipt remain incomplete. Scientific
execution remains paused; only documentation and local checkpoint preparation
have resumed.
Primary image reconstruction remains separate from inferred-parameter renders.

The prepared checkpoint includes project source, tests, configs, dependency
specifications and workflow documentation. It excludes generated data, private
labels/truth, checkpoints, caches, local environments, credentials, machine paths
and the machine-specific CASA setup log. Data/report paths in documentation are
references to local artifacts, which are intentionally absent from GitHub.

No real-data training or download occurred in this historical stage. See the current results for later completed scientific work.