> Public project snapshot. See [published outputs and intermediate progress](progress/README.md) and [publication scope](PUBLICATION_SCOPE.md). Generated datasets and checkpoints are intentionally local-only.

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
The local synthetic repair campaign is complete. The canonical checkpoint and existing test evaluation remain preserved. This public snapshot includes sanitized code, reports and intermediate progress; scientific data and model binaries remain local.

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

<!-- repair02:start -->
### Latest local repair campaign — October 23, 2026

The measured continuation is [repair_02/BEFORE_AFTER.md](progress/ml_initial_20261023/repair_02/BEFORE_AFTER.md), with [SUMMARY.json](progress/ml_initial_20261023/repair_02/SUMMARY.json). The earlier review below remains the preserved before record. **The single thermal-whitened PCA128 remedy failed validation and was not promoted.** Original epoch 22/stop 52 model, 96-observation test archive, simulator, data, preparation and earlier runs/review remain unchanged. No new test scoring, bank generation, dependency change or publication occurred.

| Target | Unit | Original validation MAE | Thermal128 validation MAE | Prior median MAE |
|---|---|---:|---:|---:|
| em0_pc_cm6 | pc cm^-6 | 33718 | 35336 | 87035.2 |
| te_k | K | 1832.04 | 1829.81 | 1788.08 |
| major_fwhm_arcsec | arcsec (full major EM FWHM) | 7.53589 | 8.44283 | 11.4158 |
| axis_ratio | 1 | 0.130725 | 0.13329 | 0.128671 |

The new fit selected epoch 18 and stopped at 48. Metrics use 134 reused validation observations /67 sources; source-paired uncertainty, coverage/widths, prior-coordinate density scores and measured local likelihood-information retention are in the report. The proposed representation change did not yield a better learned posterior.

Canonical matched PCA32 cases passed: 2/2; full-data cases with complete diagnostics: 2/2. Two optional PCA128 Bayesian routes were deferred after the remedy failed validation. Failed phases and costs remain saved; only passing chains support comparisons. The matched cases use the same ideal compression, priors, exact transformed covariance and explicit nuisance parameters; historical float32 rounding and full-data information differences are documented. Two cases support no population ranking.

The canonical neural posterior still differs substantially from matched Bayes in these two cases: EM95% intervals are about 8–10 times wider and axis-ratio intervals about 1.7–1.8 times wider. Fast inference and reasonable aggregate point errors do not establish posterior fidelity or real-nebula readiness.

Temperature remains weakly identified in the demonstrated cases; 129/134 new validation outputs receive an explicit prior-dominance warning. All four targets and full intervals remain; no fixed/dropped Te or precise-temperature claim. A new measured-data CLI exposes this warning. The corrected half-data run, predictive checks, convergence history, scoped timings, regressions and limitations are linked in the report.

A [protected evaluation proposal](progress/ml_initial_20261023/repair_02/PROTECTED_EVALUATION_PROPOSAL.md) is saved, with no generation authorized or started. Further model/data/physics validation and explicit approval for any new protected bank remain pending.
<!-- repair02:end -->


### Previous independent audit and repair — October 23, 2026 (preserved baseline)

The earlier audit comparison is
[review_01/final_01/BEFORE_AFTER.md](progress/ml_initial_20261023/review_01/final_01/BEFORE_AFTER.md),
with machine-readable `final_01/SUMMARY.json` and `DELIVERY.json`.
`review_01/LATEST.json` points to this version; `STATUS.json` records execution.
**All scientific limitations have NOT been eliminated.** The simulator,
original candidate/preparation, `run_03` and every `followup_01` file remain
unchanged. The original checkpoint remains canonical; the new probe is not
automatically promoted. No existing test prediction was rescored or used for
selection, and no new scientific/protected evaluation bank was generated.

**Prominent unresolved result:** temperature and axis-ratio validation MAEs
remain worse than their matched prior-median diagnostics. Their original
source-paired disadvantage intervals include zero, so these small point-error
disadvantages alone do not prove a bad posterior. The following numbers use
**134 validation observations /67 source groups**, not 134 independent skies:

| Target | Units | Frozen model MAE | Single probit probe MAE | Prior median MAE |
|---|---|---:|---:|---:|
| `em0_pc_cm6` | pc cm^-6 | 33,718.01 | 32,218.78 | 87,035.24 |
| `te_k` | K | 1,832.04 | 1,831.85 | 1,788.08 |
| `major_fwhm_arcsec` | arcsec, full major EM FWHM | 7.53589 | 7.57307 | 11.41581 |
| `axis_ratio` | 1 | 0.130725 | 0.129264 | 0.128671 |

The original Te MAE excess is +43.97 K (descriptive source-resampling 95%
interval [-59.67,146.74] K); q excess is +0.002054 ([-0.005336,0.009427]).
Point estimates are marginal posterior medians. Log-uniform EM/size baselines
use geometric midpoints; Te/q use linear midpoints. The same source grouping
is used for uncertainty calculations. Validation was reused for stopping and
development; these are not fresh blind or selection-adjusted population claims.

The model-focused diagnosis is in `model_diagnosis_01/DIAGNOSIS.md`:

- Full-data Bayesian Te marginals in the two preselected cases are nearly
  prior-like (descriptive prior-relative histogram KL about 0.005–0.007 nats).
  This supports weak identifiability there, not universal irreducibility.
- Full-data Bayesian q marginals are informative (KL about 0.89/1.31 nats),
  while neural q marginals remain nearly prior-like. Local Fisher calculations
  with the exact transformed correlated covariance retain median q information
  fractions 0.195/0.578 after PCA, at three posterior states per case. These
  local quantities, unlike variance retention alone, measure parameter
  information; they still do not prove global sufficiency.
- Marginal prior-relative log scores expose weak Te/q despite strong EM/size
  gains. Joint density loss and realized sample MAE need not rank models the
  same way; an exact posterior median is nevertheless optimal for expected
  absolute error. No median/unit/pairing/gradient/sampling defect was found in
  the frozen predictor.
- Limited training sources, one seed and restricted density families remain
  plausible contributors. The old half-data fit reused normalization from all
  794 training observations, although its PCA/weights used the half subset.
  This is not heldout leakage, but that comparison is not a clean end-to-end
  learning curve and does not establish data sufficiency.

One **predeclared validation-only sensitivity probe** changed only the target
map from scaled logit(prior-unit) to probit(prior-unit), keeping data, PCA32,
hidden32, seed and optimizer/stopping settings fixed. Its protocol was saved
before training; it selected epoch 27 and stopped at 57. A Gaussian in probit
coordinates can represent the exact uninformative prior, whereas the old
Gaussian in logit coordinates cannot. This changes representability, not the
physical prior or available measurement information.

The probe's results were mixed, as the MAE table shows. Joint NLL in common
prior-unit coordinates improved from -0.69441 to -0.92699 nats (exact prior=0;
both Jacobians included); change -0.23258 has a descriptive source-resampling
interval [-0.36924,-0.08437]. This is development evidence only. Central 95%
validation coverage is 91.79% EM, 95.52% Te, 93.28% size and 91.79% q, with
sample-size/selection limits; no broad calibration improvement is claimed.
See `probit_probe_01/RESULTS.md`, `comparison.json`, saved checkpoint and curves.

Recoverable verification defects were repaired in a versioned route: actual
raw-chain acceptance, complete-variable checks, JSON-safe nonfinite/stuck
failures, odd-length FFT inversion and fail-closed prior/unit/layout contracts.
Historical code/artifacts are preserved. Direct comparisons independently
confirm the exact replay, the expected 48 profile records (original 46 intact),
and both full Bayesian passes. All 128 Bayesian predictive receipts per case
were regenerated from the raw chains and original RNG stream. Sixteen new
validation optimizer starts returned to known regions; finite searches and
diagnostics still cannot prove global mode completeness.

The neural nuisance extension now uses
`q_NN(theta|C(y)) p(nuisance|theta,y)`, with no true nuisance values or optimized
points substituted for distributions. All four original outer target draws
were retained: the initial bounded conditional pilot passed 2/4; exact saved
state/proposal/RNG continuation raised this to 3/4; one predeclared full-period
PA proposal passed the last case in a separate homogeneous phase. **4/4
conditional passes are not full neural predictive calibration**: there are only
two outer theta draws per validation observation. Earlier failures remain saved,
and tail ESS margins are reported. Band/radius checks using the same accepted
draws show some theta draws have structured full-data discrepancies even after
nuisance integration; their frequency across the neural posterior is unknown.

The NPE sees PCA-compressed measurements while the conventional reference uses
the full vector. The hybrid nuisance extension uses full data for its conditional
solve. A matched compressed-likelihood Bayesian posterior, using `C Sigma C^T`,
is still needed to separate compression from estimator error completely. The
current full-data comparison is a reference for the desired raw-data posterior,
not a controlled estimator-only comparison. Physical priors/noise/forward laws
match; float precision, compression, family and finite-design differences remain
documented. No test or challenge data were used to remedy them.

Testing verified 69 unique checks across the combined and targeted runs. The
combined 63-test run had 62 passes and a Windows temporary-fixture replacement
failure; a sandbox TEMP retry had a setup error; the unchanged test then passed
in a fresh shorter workspace path. This is an encountered environment incident,
not a simulator software defect claimed fixed. The new guard/checksum
integration failure is also preserved, with its source snapshot and corrected
attempt. `INCIDENTS.json` and test logs retain every result. Tiny regression
fixtures are distinct from any scientific training or protected evaluation bank.

New nuisance integration, continuation and PA-phase costs are reported separately
from the ~0.00077 s amortized neural target inference. The probit probe cost
56.19 s including loading/evaluation, with 1.79 s in its optimizer loop; its
scope differs from the older 83.89 s normalization-including training replay.
There is no matched-accuracy speedup claim. No dependencies, simulator, original
datasets, real observations, credentials/security settings or GitHub publication
were changed.

**Completed:** bounded local audit, scoped numerical/contract fixes, independent
saved-state verification, all four accepted nuisance conditionals, model-focused
diagnosis, one frozen validation-only probe, before/after metrics and failure
accounting. **Pending:** Te/q model adequacy, cause partition, broader calibrated
posteriors, finite-mode/tail confidence and real-data transfer. A future blind
claim needs a separately approved protected independent source set; none was
created here. The original canonical model and evaluated test archive remain
preserved.

### Earlier local continuation — October 23, 2026 (preserved `followup_01` record)

The earlier evaluation-only artifacts are
`output/ml_initial_20261023/followup_01/`. The original `run_03/` files remain
unchanged. An elevated read-only process inventory found no Python/trainer
before launch; one continuation was started, with an exclusive process lock.
No new dependency or replacement neural architecture was needed.

- Rechecked all **3,712 original dataset files** and the frozen model/code/data
  bindings. Reproduced the normalization exactly from training observations and
  unique training parents; reproduced PCA from training rows only.
- Evaluated the frozen model on **134 validation observations /67 sources** for
  documentation and debugging. Existing test predictions were checked for
  archive consistency and were **not regenerated**. Validation temperature MAE
  is 1,832 K versus 1,788 K for the prior-midpoint diagnostic.
- Finished the missing **two** forward profiles: **48/48** are now saved in the
  continuation directory, retaining the original 46 rows. All optimizer statuses
  were successful. These remain optimistic nuisance-profile residual checks,
  not full neural posterior predictive checks.
- The conventional follow-up uses the **same two preselected validation
  observations**. It retains the original physical/noise/nuisance posterior and
  improves only the sampler proposal. Four independent chains, dispersed
  measurement-only initialization, rank/folded Rhat, bulk/tail ESS, all nuisance
  coordinates and periodic PA diagnostics are recorded. **Both follow-up cases
  pass**, each with 4,000 retained draws per chain after 1,000 warmup steps.
  The original two pilots remain failed historical results; their summaries
  are not substituted for the validated follow-up.

| Validation case | Maximum rank/folded Rhat | Minimum bulk ESS | Minimum tail ESS | Bayesian predictive tail fraction (128 selections) |
|---|---:|---:|---:|---:|
| First preselected source | 1.00464 | 902 | 986 | 0.508 |
| Second preselected source | 1.00710 | 943 | 854 | 0.469 |

The gate requires Rhat <1.01 and both ESS >=400 for every target, nuisance,
and PA sine/cosine. Reference-render discrepancies were below 6.5e-14 in
thermal-whitened L2 units. These diagnostics do not prove global mode discovery.
Predictive tail fractions are descriptive conditional thermal residual checks,
not evidence of population calibration. Two validation sources cannot establish
which method is generally more accurate. Full matched posterior intervals,
widths and original-unit errors are in the report/case JSONs.

The existing test evaluation has 96 observations but only **48 independent
sources**. It is an evaluated archive, not a fresh blind test:

| Target | Units | MAE | RMSE | Central 95% coverage | Mean 95% interval width |
|---|---|---:|---:|---:|---:|
| `em0_pc_cm6` | pc cm^-6 | 38,041 | 56,108 | 98.96% | 266,234 |
| `te_k` | K | 1,698 | 1,992 | 96.88% | 6,459 |
| `major_fwhm_arcsec` | arcsec, full major-axis EM FWHM | 6.978 | 9.590 | 97.92% | 38.25 |
| `axis_ratio` | dimensionless | 0.1254 | 0.1435 | 94.79% | 0.4613 |

EM0 and size are log10-transformed in the loader; Te/q remain linear, with
standardization fitted from unique training parents. The frozen neural model
uses the separate fixed transform logit(prior-unit coordinate)/(pi/sqrt(3))
on original labels in its loss. Inputs contain only 3,906 normalized measured
Fourier components. Labels, IDs, seeds, true images and realized nuisances do
not enter the predictor. Source-grouped splits remain 794/134/96 observations;
the 24 different-layout challenge observations remain separate.

The neural approximation compresses each raw observation to 32 training-only
PCA coordinates (39.22% of standardized training variance) and uses one joint
Gaussian in transformed target coordinates. Conventional inference uses the
full raw vector and samples target/nuisance correlations explicitly. The
loader rounds normalized neural inputs to float32; conventional likelihood
arithmetic promotes the same stored measurements to float64. The
continuous priors and noise/forward law match; the neural nuisance marginal
is learned approximately from a finite scrambled-Sobol design. Those are
documented approximation/design differences, not identical algorithms.
Both methods use uniform PA, position priors, shared lognormal common gain,
Gaussian spectral slope and the same correlated conditional thermal likelihood.

The [full local report](progress/ml_initial_20261023/followup_01/REPORT.md) includes all 50/80/95% interval widths/coverage,
source-bootstrap uncertainty, original-unit errors, convergence diagnostics,
training curves and runtime definitions. Broad/overcovering intervals, the
small number of test groups and weak temperature recovery limit conclusions.
The training timer originally omitted PCA and data loading. A separate,
instrumented **training/validation-only replay** reproduced every original
weight/PCA array exactly and the same epoch 22/52 decisions. It did not replace
the frozen model or open test/challenge labels. Measured cost was **83.89 wall s,
69.81 CPU s**: normalization fit 35.60 s, verified train/validation loading
43.14 s, and PCA/optimizer/checkpoint writing 1.90 s. The remaining time includes
normalization verification/serialization and bookkeeping. Startup imports,
generation/readiness and the half-data diagnostic are outside that timer.

Neural median inference took about 0.00019 s, or 0.00077 s including 2,048
posterior draws, on the same validation observations (30 warmed repetitions;
normalization/PCA included, disk loading excluded). Conventional follow-up
sampling took 485.91/630.83 s per source including initialization and warmup.
Because its proposal uses the original pilot draws, include those historical
76.37/80.85 s too: **562.28/711.68 s of recorded sampler work**. Sessions/cache
conditions differ; setup/reference-render/predictive costs are saved separately.
Approximate neural draws and MCMC effective samples have different accuracy;
these are cost measurements, not a matched-accuracy speedup claim.

The software environment stayed Python 3.13.14, NumPy 2.2.6, SciPy 1.18.0,
Astropy 8.0.2 and Matplotlib 3.11.0. Original/replay training seed: 231026;
follow-up seed: 231027 (case and independent-chain offsets in the saved code).
Regression evidence contains 46 unique passing checks across the initial
45-pass/one-failure run and corrected three-test rerun. The sole failure was a
new test's comparison of equivalent periodic PA coordinates, corrected in the
fixture; the sampler was unchanged. Receipts retain both results.

No method superiority, predetermined accuracy threshold or real-nebula
readiness is claimed.

| Artifact | Path |
|---|---|
| Complete local report and inspected figures | `output/ml_initial_20261023/followup_01/REPORT.md`, `training_curves.png`, `archived_test_coverage.png`, `bayes_01_traces.png`, `bayes_02_traces.png` |
| Final completed/pending/blocker record | `output/ml_initial_20261023/followup_01/summary.json` |
| Inventory, source/code/environment bindings and final verification | `output/ml_initial_20261023/followup_01/audit.json`, `binding.json`, `final_verification.json`, `verification_receipt.json` |
| Same-observation Bayesian comparison, convergence and joint predictive draws | `output/ml_initial_20261023/followup_01/bayesian_comparison.json`, `bayes_01_comparison.json`, `bayes_02_comparison.json` |
| Retained chains, seeds, end RNG states and initialization records | `output/ml_initial_20261023/followup_01/bayes_*_chains.npz`, `bayes_*_sampling.json`, `bayes_*_initialization.json` |
| Completed forward profiles | `output/ml_initial_20261023/followup_01/test_forward_checks.json` |
| Validation metrics and separate targets/predictions | `output/ml_initial_20261023/followup_01/validation/` |
| Training replay, identical weights/PCA, config and full timings | `output/ml_initial_20261023/followup_01/training_runtime_replay/` |
| Complete runtime interpretation | `output/ml_initial_20261023/followup_01/runtime_accounting.json` |
| Original comparison model and PCA (unchanged) | `output/ml_initial_20261023/run_03/full/best.npz`, `full/pca.npz` |

Runnable continuation/staged commands are in [ml/README.md](ml/README.md).
The fixed sampler protocol is [ml/followup_plan_v1.json](ml/followup_plan_v1.json).
Further model development must use training/validation only. Before any new
blind claim, propose a protected independently generated source bank, freeze
model/metrics/prior/noise choices, and reserve its labels for one evaluation.
No new bank was generated or authorized here.

**Completed:** frozen training/test-state verification, train-only preprocessing
verification, all 48 forward profiles, two converged matched Bayesian validation
comparisons with joint predictive checks, exact training replay, saved curves,
metrics, configs/seeds/versions and runtime evidence. **Pending:** full neural
predictive nuisance correlations, broader source-level Bayesian/calibration
evidence, and a protected independent evaluation set if future models require a
new blind test (proposal only). **Blockers:** none for this bounded local
follow-up. Nothing is running in the background after the final checks.

### Saved initial ML experiment (original `run_03` archive)

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

### Read-only orientation and evaluation-only continuation

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
end-to-end training/evaluation program and has **no resume-only CLI**.
The separate `ml/continue_initial.py` now provides explicit audit, forward,
Bayesian and report stages. It loads the frozen checkpoint without retraining
or reopening test-based model selection. See [continuation commands](ml/README.md).
Changed models
must use train/validation for development and a newly reserved independent
evaluation set before new blind-generalization claims.

For reproducibility, the original successful launch was the command below,
with Python 3.13.14, NumPy 2.2.6 and SciPy 1.18.0 in the existing environment.
It is recorded for reproducibility; it was not rerun for this continuation. The existing
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

Any later verification must choose a fresh temporary output directory. The
current user request authorizes local scientific verification and continuation.
review and give to their own agent.

### Git checkpoint scope

Requested destination: `https://github.com/Aditya24123/astrophysics-content.git`
(public). The checkpoint uses branch `checkpoint/physics-renderer-20261023` and
subdirectory `physics-renderer/`, preserving the repository's existing files
and history. The first local checkpoint was `b71fe0e13963ab0d9b62f442639ea52d6f548b30`.
Automatic approval review rejected the first public push before execution.
The user subsequently gave specific consent to publish the scanned code and
documentation, including this editable README, to this public destination.
Verify the remote branch/commit when checking backup status. That historical
save did not resume scientific work. The present local continuation is separate;
no new GitHub publication was requested or performed.

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
