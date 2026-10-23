# H II physics renderer and initial synthetic inference

This repository generates numerical sky brightness and synthetic radio
measurements for a deliberately simple ionized-hydrogen source family. A separate
initial neural posterior experiment now lives in `ml/`; real-object inference
has not been established.
Read [the physics specification](docs/MODEL.md) and [data contract](docs/DATA_CONTRACT.md)
before using the outputs as training data.

## Current editable project record — October 23, 2026

This is an ordinary editable Markdown file. Update this record as verified
progress is made, and retain the dated evidence rather than replacing it.
**Training, Bayesian sampling and forward checks are currently paused.**
Only documentation and a local Git checkpoint are being prepared.

The pipeline is:

1. Sample a source's EM0, electron temperature, major-axis EM FWHM, axis ratio,
   position angle and positional offsets from the declared synthetic priors.
2. Render the spatial EM profile, free-free optical depth and radio brightness,
   with pixel/channel integration. This is H II ionized gas, not H2.
3. Apply the primary beam and sampled Fourier observing operator, shared
   calibration draws and correlated thermal noise to obtain measured radio data.
4. Reconstruct the primary image independently from those measurements. The
   separate CASA path supplies higher-fidelity development fixtures; it does not
   certify this six-band Fourier proxy as a real telescope.
5. Feed only measured Fourier values to the neural posterior estimator. True
   parameters are separate supervised answers used in the loss; source IDs,
   seeds, true sky and realized gains are excluded from predictor features.
6. Compare against conventional explicit-physics Bayesian posterior inference
   using the same measurement vector, source priors and nuisance assumptions.
   Forward renders from inferred parameters are additional validation products.

The four current prediction targets, in order, are `em0_pc_cm6` [pc cm^-6],
`te_k` [K], `major_fwhm_arcsec` [arcsec, full major-axis **EM** FWHM], and
`axis_ratio` [dimensionless]. Other physical quantities remain future work.

### Data and readiness

The immutable local candidate is
`output/training_readiness_20261021/candidate_gl4_512_01`.
The verified preparation is `output/training_ready_20261022/initial_02`.
The fresh October 23 check is `output/ml_initial_20261023/readiness_01`:
all five readiness gates passed and all 3,712 files passed integrity checks.
These certify loading/numerical readiness within the declared proxy, not
parameter recovery or transfer to real nebulae.

| Split | Independent sources | Observations |
|---|---:|---:|
| Train | 397 | 794 |
| Validation | 67 | 134 |
| Test | 48 | 96 |
| Separate challenge | 12 | 24 |

Keep each source's two noisy realizations in the same split. Use the existing
loader and train-only normalization; do not substitute the older exposed
`output/production` bank. Challenges remain separate and have mixed layouts.
Generated banks, protected labels and truth are local artifacts excluded from
the Git checkpoint; a repository clone alone does not contain this dataset.

### Saved initial ML experiment

Protocol: [ml/plan_v1.json](ml/plan_v1.json). Results:
`output/ml_initial_20261023/run_03/`. Simulator and dataset files are separate
from the ML implementation and were preserved.

- CPU-only, 1,518-parameter NumPy neural density estimator: train-only 32-component
  PCA, one 32-unit tanh layer, and a full covariance logit-normal conditional
  posterior over the four bounded parameters. This approximates the posterior;
  it is not a Bayesian distribution over neural-network weights. Compression
  and a single Gaussian in transformed coordinates can miss useful information,
  curved degeneracies and multiple modes.
- Best validation checkpoint: **epoch 22**. Validation patience stopped training
  at **epoch 52**. The half-data diagnostic had a similar validation loss; one
  such comparison does not establish a learning curve or data sufficiency.
- The selected checkpoint was frozen before evaluating **all 96 test
  observations from 48 sources once**. This test is now an evaluated archive,
  **not an untouched test for selecting subsequent models**. No test-based tuning
  followed. Earlier dataset audits had also inspected its original targets.
- Saved test median absolute relative errors were approximately 29% EM0,
  20% temperature, 22% size and 16% axis ratio. These describe this small
  experiment only. Temperature MAE was worse than the prior-midpoint diagnostic;
  several posterior intervals were broad or overcovered. Scientific accuracy
  and calibration have not been established.
- Conventional Bayesian inference ran four chains on each of two predeclared
  validation sources, with explicit PA, position and calibration nuisances.
  **Both pilots failed the declared convergence criteria.** Their summaries are
  provisional; neither method has demonstrated superiority. No MCMC-derived
  training labels were created.
- Forward checks were interrupted: the last console milestone was **44/48**.
  A later read-only inspection found **46 saved cases** in
  `test_forward_checks.json`; the remaining two are incomplete and
  `completion.json` is absent. These checks fit nuisances at the neural marginal
  medians and are optimistic profile-residual diagnostics, not full neural
  posterior predictive checks. Bayesian posterior forward samples are recorded
  separately in `bayesian_comparison.json`.
- **43 regression/contract/numerical tests passed**. Two earlier launch attempts
  are retained: an overbroad image guard and a Windows memory-monitor handle
  error were corrected before the frozen test evaluation.

| Local artifact within `run_03/` | Purpose |
|---|---|
| `full/best.npz`, `full/pca.npz` | Selected weights and fitted input projection |
| `full/training.json`, `full/epochs.jsonl` | Selection, stopping and resource logs |
| `FROZEN_BEFORE_TEST.json` | Timestamp and hashes recorded before test unsealing |
| `run_binding.json` | Plan, code, dataset, normalization and environment hashes |
| `test_evaluation.json`, `test_predictions.npz` | Test metrics and private per-observation outputs |
| `bayesian_comparison.json`, `bayes_01_chains.npz`, `bayes_02_chains.npz` | Pilot posterior comparison and diagnostics |
| `test_forward_checks.json` | Partial forward-profile evidence |

The selected checkpoint SHA256 is
`e07db567c60f826280f47fd063f7068cebc4df49fedb4e1377cfe1222925ec0e`.
No real-data training/download, paid/cloud compute or Library upload occurred.

### Read-only orientation and safe restart

From the project root, these commands inspect existing results without starting
training or changing the saved scientific artifacts:

```powershell
$env:PYTHONPATH = 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
Get-Content ml/plan_v1.json
Get-Content output/ml_initial_20261023/run_03/full/training.json
Get-Content output/ml_initial_20261023/run_03/FROZEN_BEFORE_TEST.json
Get-Content output/ml_initial_20261023/run_03/test_evaluation.json
Get-Content output/ml_initial_20261023/run_03/bayesian_comparison.json
Get-FileHash output/ml_initial_20261023/run_03/full/best.npz -Algorithm SHA256
```

Do not restart `ml/run_initial.py` to finish the interrupted checks: it is an
end-to-end training/evaluation program and has **no resume-only CLI**. A later
agent should first inspect the partial outputs, implement an explicit
evaluation-only continuation if authorized, and load the frozen checkpoint
without retraining or reopening test-based model selection. Changed models
must use train/validation for development and a newly reserved independent
evaluation set before new blind-generalization claims.

For reproducibility, the original successful launch was the command below,
with Python 3.13.14, NumPy 2.2.6 and SciPy 1.18.0 in the existing environment.
It is recorded as history, **not a command to run while paused**. The existing
output directory is deliberately protected against overwrite; any authorized
exact replay requires a new directory and must be labelled a replay of already
evaluated data, not a new blind test.

```powershell
$env:PYTHONPATH = 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.venv/Scripts/python.exe -B ml/run_initial.py --out output/ml_initial_20261023/run_03
```

The completed verification command was:

```powershell
.venv/Scripts/python.exe -B -m pytest -q ml/test_ml.py tests/test_training_data.py tests/test_npe_contract.py --basetemp output/ml_initial_20261023/regression_01
```

Any later verification must choose a fresh temporary output directory. No new
scientific calculation or test run is authorized by this documentation update.
The [next-agent prompt](ml/NEXT_AGENT_PROMPT.md) is editable text for the user to
review and give to their own agent.

### Git checkpoint scope

Requested destination: `https://github.com/Aditya24123/astrophysics-content.git`
(public). The checkpoint uses branch `checkpoint/physics-renderer-20261023` and
subdirectory `physics-renderer/`, preserving the repository's existing files
and history. The first local checkpoint was `b71fe0e13963ab0d9b62f442639ea52d6f548b30`.
Automatic approval review rejected the first public push before execution.
The user subsequently gave specific consent to publish the scanned code and
documentation, including this editable README, to this public destination.
Verify the remote branch/commit when checking backup status. Scientific work
remains paused; only documentation and this repository save have resumed.

The allowlisted upload contains source, tests, configs, dependency declarations
and documentation. It excludes generated banks, private labels/truth,
per-observation outputs, model binaries, credentials, caches, logs, local
environments and machine-specific setup notes. It is a code/progress save,
not a complete backup of the locally generated data or trained model.

## Historical simulator-stage notes

The following dated notes and existing setup instructions are retained for
context. Their no-training statements describe those earlier stages.

**October 22 initial data entry point:** the fresh 524-parent/1,048-observation
candidate has a [training batch starter](docs/TRAINING_STARTER.md) with train-only
normalization, split-safe original targets and public-only test loading. The
versioned preparation report is `output/training_ready_20261022/initial_02/readiness_report.json`.
This is an initial declared-proxy experiment, not real-data or trained-model
certification. No training is started by the preparation or example commands.

**Audit revision 0.2.0:** the original production bank has confirmed label
leakage and scientific limitations. It is preserved, but the predictor loader
rejects its schema by default. Read [the audit](docs/AUDIT_V2.md),
[instrument scope](docs/INSTRUMENT_AUDIT.md), and
[measurement-only reconstruction](docs/RECONSTRUCTION.md). No training or
new bulk generation was performed for this repair.

The separate CASA path supports [shared coarse measurement-only reconstruction](docs/CASA_RECONSTRUCTION_PATH.md)
for existing Measurement Sets and explicitly selected future simulation runs.
Its 3x MAD multiplier is selected among the bounded development comparisons;
one actual saved-case execution has exact parity. See the
[milestone assessment](docs/CASA_TERMINAL_MILESTONE.md) for the remaining
acceptance step and scientific limits. Earlier outputs are preserved.

## Install and run

Use Python 3.13 with the pinned dependencies in `requirements.txt`:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pip install -r requirements-dev.txt
$env:PYTHONPATH = 'src'
.venv\Scripts\python -m pytest -q --basetemp output/test_rerun_01
.venv\Scripts\python -m hii_sim.audit --config configs/compact_proxy_v2.json --out output/audit_rerun_01 --widefield
```

`hii-sim generate`, `hii-sim plan`, `hii-sim validate`, and `hii_sim.audit`
default to the guarded `configs/compact_proxy_v2.json` scope. Pass
`--config configs/base.json` explicitly only when reproducing the legacy
instrument behavior; that configuration has no short-spacing guard and uses
128 MHz windows.

`generate` creates independent base sources plus the configured challenge
sources. `--limit` bounds each invocation; rerun the exact command to resume.
The output directory records the configuration and source plan, refuses a
changed configuration/code/runtime on resume, commits one physical source at
a time, and rebuilds the public manifest from committed records. Keep the
`protected/` directory private. The `public/` subtree is the only predictor
input surface.

The first `generate` invocation for a dataset must have room for its output.
The bounded 16,384-source production dataset was generated before this audit.
Its old code/provenance cannot be resumed using the revised code; the strict
guard is retained. Use a new destination for any later authorized generation.
[Historical validation](docs/VALIDATION.md) retains the old sizing measurements,
not a certification of the old dataset.

## Two instrument paths

The default is a fast *gridded Fourier development approximation*: a synthetic
Hermitian uv mask, Gaussian taper, primary-beam envelope, correlated complex
noise, shared flux-scale/spectral calibration errors, and a unit-peak dirty
image. It is useful for bounded experiments and is explicitly not CASA, a
VLA observation simulator, or a complete interferometer model.

For an actual measurement-set path, run `prepare-casa` on a protected truth
FITS cube, then run `src/hii_sim/casa_simulate.py` inside CASA 6.7.4. The bridge
uses the named built-in `vla.c.cfg`, 30 min on source, 10 s integrations, a
transit pointing, one 128 MHz continuum window per modeled frequency,
`simobserve` with `tsys-atm`, and `tclean` natural-weighted dirty imaging
(`niter=0`, no primary-beam correction). The Measurement Sets, dirty images,
PSFs, PB maps, and residuals are numerical products. The bridge has not been
run in the current Windows environment: CASA is absent. See the exact
integration command and limitations in [the CASA notes](docs/CASA.md).

## Deliverables currently in the repository

- Physics, source geometry, fast measurement, reconstruction, data I/O,
  predictor/training loaders, bounded dataset generation, and CLI modules in
  `src/hii_sim/`.
- The guarded compact scope in `configs/compact_proxy_v2.json`; the preserved
  `configs/base.json` is for explicit legacy replay.
- Versioned equations, assumptions, and survey evidence in `docs/MODEL.md`.
- Independent numerical validation, reproducible pilot and production
  distribution plots, and pilot/production measurements in `output/validation/`.
- Completed 16,396-source, 32,792-observation dataset in `output/production/`;
  its public subset order exposed generating parameters. Legacy shared
  instrument products had no original checksums. See the audit before reuse.
- Original `output/workflow.md` retained as user-provided context; its generic
  density/magnetic/motion targets should be read as later work, not claims of
  this simulator.
