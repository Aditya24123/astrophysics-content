# Physics renderer checkpoint — October 23, 2026

The H II synthetic simulator and dataset preparation stage is preserved.
The current candidate contains 524 source groups and 1,048 noisy observations:
397/67/48 train/validation/test sources, plus 12 separate challenge sources.
All 3,712 original files passed a fresh integrity recheck; all five readiness
gates passed. The old exposed dataset is not used for training.

The new ML stage uses a small full covariance logit-normal conditional neural
posterior, train-only PCA, source-grouped splits, validation early stopping,
and a reserved final test. The written protocol is `ml/plan_v1.json`.
Numerical gradient, covariance-sampling, support, PCA and forward-equivalence
checks passed. A bounded CPU training run has been launched; this checkpoint
does not yet claim a completed model or scientific accuracy. Two early launch
failures concerned an overbroad image guard and Windows resource logging;
both were corrected before any test evaluation.

A conventional Bayesian posterior comparator uses the same radio data, priors,
forward physics and nuisance assumptions on two predeclared validation sources.
Its convergence diagnostics will be reported; no method is assumed to win.
Primary image reconstruction remains separate from inferred-parameter renders.

This public checkpoint includes project source, tests, configs, dependency
specifications and workflow documentation. It excludes generated data, private
labels/truth, checkpoints, caches, local environments, credentials, machine paths
and the machine-specific CASA setup log. Data/report paths in documentation are
references to local artifacts, which are intentionally absent from GitHub.

No real-data training, download, cloud compute, or Library upload is included.
This is a code/workflow save, not a complete backup of the generated datasets.
