# H II physics renderer and inference experiments

The simulator creates numerical radio observations of a simple ionized-hydrogen source family. A separate synthetic inference stage compares a neural posterior estimator with conventional Bayesian inference.

- [Editable project README](physics-renderer/README.md)
- [Outputs and intermediate progress](physics-renderer/progress/README.md)
- [Latest measured before/after report](physics-renderer/progress/ml_initial_20261023/repair_02/BEFORE_AFTER.md)
- [Publication scope and provenance](physics-renderer/PUBLICATION_SCOPE.md)
- [Physics specification](physics-renderer/docs/MODEL.md)

The original epoch-22 model is retained. The PCA128 remedy was rejected; posterior fidelity, broader calibration and temperature information remain unresolved. These are synthetic experiments, not evidence of real-observation readiness.

Code, tests, scientific protocols, aggregate diagnostics, five figures and intermediate failed/successful phases are included. Datasets, private labels/seeds, raw chains, model checkpoints, credentials and task prompts are excluded. Run commands from `physics-renderer/`; reproducing experiments requires the separately retained local data and model artifacts.
