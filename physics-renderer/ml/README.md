# Initial synthetic machine-learning stage

## Independent audit/repair route (October 23, 2026)

The current versioned audit is `output/ml_initial_20261023/review_01/`.
Its `BEFORE_METRICS.json` and `ISSUES_BEFORE.md` precede the repairs; the final
comparison is `final_01/BEFORE_AFTER.md` (see `LATEST.json`). `STATUS.json` distinguishes active execution,
completed stages and failures. The preserved `run_03` and `followup_01` files and
the simulator/dataset are never overwritten by this route.

Use `review_inference_v2.py` for new independent checks, rather than treating
the historical finalizer's saved summary as fresh evidence. It recomputes all
chain diagnostics, source/case counts, physical quantiles and every stored
Bayesian predictive receipt from raw chains and the bound RNG stream. Strict
`diagnostics_v2.py` requires every variable to be named, handles nonfinite/stuck
failures in JSON, and fixes odd-length FFT inversion. The original diagnostic
code is kept unchanged to preserve old bindings. The existing 4000-draw chain
passes are unaffected by the odd-padding fix.

```powershell
$env:PYTHONPATH='src;ml'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OPENBLAS_NUM_THREADS='2'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
.venv/Scripts/python.exe -u -B ml/audit_before_v2.py --out output/ml_initial_20261023/review_NEW
.venv/Scripts/python.exe -u -B ml/review_inference_v2.py --review output/ml_initial_20261023/review_NEW --stage verify
.venv/Scripts/python.exe -u -B ml/review_inference_v2.py --review output/ml_initial_20261023/review_NEW --stage modes
.venv/Scripts/python.exe -u -B ml/review_inference_v2.py --review output/ml_initial_20261023/review_NEW --stage conditional
.venv/Scripts/python.exe -u -B ml/review_inference_v2.py --review output/ml_initial_20261023/review_NEW --stage report
```

The `conditional` stage is a bounded, costly validation pilot. It draws frozen
neural targets first, then samples the physical nuisance posterior conditional
on each inferred target draw and the observation. Its approximation is
`q_NN(theta|y) p(nuisance|theta,y)`, preserving conditional correlations without
using true nuisances or replacing distributions with optimized points. All
conditional chains must pass the same all-nuisance gate; failed outer draws
are never filtered/replaced. Two outer draws per observation cannot establish
full predictive calibration, even when conditional solves converge. The extra
Bayesian work is timed separately from fast neural inference.

The first integration attempt is retained under `review_01/v2/`, with its exact
source snapshot and traceback: the holdout guard blocked checksum-only reads.
The corrected `v2_02/` attempt permits only the frozen raw-byte digest function
for these reads, still forbids label decoding, and records failures explicitly.
No test prediction, new source bank or real-observation inference is part of
these commands. See `review_plan_v2.json` for seeds, bounded budgets and limits.

The completed review also records source-paired model-error uncertainty,
proper marginal/joint prior-relative scores, and exact-covariance local PCA
information in `model_diagnosis_01/`. The historical half-data diagnostic reused
full-training normalization; it is not a clean end-to-end learning curve.
One predeclared probit target-transform fit, `probit_probe_01/`, improves some
development metrics but leaves Te/q MAEs above their prior medians and was
not promoted. Its protocol was frozen before training and includes common
prior-unit Jacobians for score comparisons; no test evaluation occurred.

All four original predictive outer targets remain saved. After the initial
2/4 conditional passes, exact saved-state/proposal/RNG continuation reached
3/4. A separately predeclared full-period PA proposal passed the remaining
case in a new homogeneous phase, retaining all failures. Four accepted
conditionals still do not establish full neural predictive calibration.

After the earlier stages finish, these are the additional historical commands
used in this review. For reproduction supply `--review` with a NEW review
directory from the BEFORE stage; existing outputs deliberately refuse overwrite.

```powershell
.venv/Scripts/python.exe -u -B ml/resume_conditionals.py
.venv/Scripts/python.exe -u -B ml/diagnose_model.py
.venv/Scripts/python.exe -u -B ml/probit_probe.py
.venv/Scripts/python.exe -u -B ml/angle_refresh.py
.venv/Scripts/python.exe -u -B ml/predictive_details_final.py
.venv/Scripts/python.exe -u -B ml/assemble_review.py
```

The model-diagnosis, probit-probe and PA-repair hypotheses/budgets were frozen
in their named JSON plans before execution. There was one model fit and one
targeted proposal repair, with no broad sweep or score-driven seed retries.
Original and new numerical/sampling/contract tests passed across split runs;
the combined suite's Windows temporary-fixture I/O failure and TEMP setup error
are retained. The unchanged test passed on a fresh shorter workspace-path retry;
this is not called a simulator defect fixed. No new scientific evaluation bank
was generated; tiny disposable regression fixtures are separate.

The full-data Bayesian reference and compressed-input NPE have an information
mismatch. A matched compressed-likelihood Bayesian comparator remains future
work; the conditional extension is a costly hybrid using full data for nuisance
conditioning. See the final report for actual errors, priors, coverage/widths,
all diagnostic margins, failures, runtime scopes and remaining scientific limits.

---

This directory is separate from the simulator and the immutable dataset. The
pre-result protocol is `plan_v1.json`. No existing dataset or preparation report
is overwritten. This is an initial experiment in H II ionized gas, not H2.

The neural estimator learns a joint conditional probability distribution for
EM0, electron temperature, full major-axis EM FWHM, and axis ratio. It receives
only the existing loader's radio measurement vector. Labels enter the loss.
PCA and all fitted input scaling use training rows only; source splits are
inherited unchanged. Validation selects the checkpoint, then the protocol and
checkpoint are frozen before test labels are opened once for final evaluation.

A full covariance logit-normal is a deliberately small posterior approximation,
not a claim of exact Bayesian inference or Bayesian neural-network weights.
It respects prior bounds but cannot represent every degeneracy. The conventional
Bayesian comparator uses explicit physics and nuisance priors. It is evaluated
on a predeclared small validation subset, with sampler convergence reported.
Neither algorithm is ground truth. The primary image reconstruction remains
independent of these parameter estimates.

The initial experiment ran. Its selected checkpoint is epoch
22, with validation stopping at epoch 52. All 96 test observations have already
been evaluated once; keep them out of further model selection. Both conventional
Bayesian pilots failed convergence checks. Forward checks were interrupted at
the last console milestone 44/48; 46 cases were subsequently found saved.

Run commands, evidence and limitations are documented in the editable project
[README](../README.md). Local artifacts are under
`output/ml_initial_20261023/run_03/`. `run_initial.py` is not a resume-only command.
No real-data work or upload has occurred. The editable

## Authorized local continuation, October 23, 2026

`continue_initial.py` adds evaluation-only stages without changing the original
model, PCA, simulator, preparation or dataset. `followup_plan_v1.json` fixes the
sampler settings and seeds. The original `run_initial.py` is preserved byte for
byte to retain its recorded experiment binding. No replacement neural model or
new dependency is needed to finish the interrupted checks.

```powershell
$env:PYTHONPATH = 'src;ml'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
# Choose a NEW output directory for a complete continuation attempt.
.venv/Scripts/python.exe -u -B ml/continue_initial.py --out output/ml_initial_20261023/followup_NEW --stage all
# Alternatively audit first, then continue stages in the SAME bound output:
.venv/Scripts/python.exe -u -B ml/continue_initial.py --out output/ml_initial_20261023/followup_STAGED --stage audit
.venv/Scripts/python.exe -u -B ml/continue_initial.py --out output/ml_initial_20261023/followup_STAGED --resume --stage forward
.venv/Scripts/python.exe -u -B ml/continue_initial.py --out output/ml_initial_20261023/followup_STAGED --resume --stage bayes
.venv/Scripts/python.exe -u -B ml/continue_initial.py --out output/ml_initial_20261023/followup_STAGED --resume --stage report
```

The audit verifies all 3,712 dataset files and frozen ML hashes, reproduces
normalization and PCA from training only, checks the saved test archive, and
evaluates validation. Forward continuation copies the saved 46 profiles and
computes only the missing two, using archived medians and public measurements.
It never reruns test predictions. Completed Bayesian cases are skipped on
resume. Interrupted sampling arrays and RNG end states are retained but are
not silently restarted or claimed resumable; an incomplete case requires an
explicit new attempt in a fresh output directory. An exclusive lock blocks
concurrent continuation jobs; inspect its PID before any manual stale-lock repair.

The new sampler changes proposals only: four independent chains use a fixed
Student-t independence proposal fitted from the existing validation pilot,
with the exact Hastings correction, mixed with symmetric local moves. Its
convergence gate includes all four targets and all nuisances, rank/folded
Rhat <1.01, bulk/tail ESS >=400, and periodic PA sine/cosine diagnostics.
Unconverged cases cannot support Bayesian error or predictive conclusions.
The likelihood, continuous physical/nuisance priors, correlated thermal law and
forward renderer are the same; vectorized thermal whitening is tested against
the original. No true nuisance values are read by inference.

The original per-source pilot is required to construct this proposal, so a
from-scratch runtime comparison must include that pilot cost. The final
`runtime_accounting.json` includes both stages and distinguishes historical
session timings, setup/predictive work, neural training and neural inference.

After the continuation exits, the following commands measure the training
cost and assemble final evidence. The replay fits only training/validation,
reproduces the frozen arrays/epoch decisions, and never replaces the original
checkpoint or evaluates test/challenge targets. Run it separately from sampling
to avoid CPU timing interference; choose a new replay directory if it exists.

```powershell
.venv/Scripts/python.exe -u -B ml/benchmark_training.py --out output/ml_initial_20261023/followup_01/training_runtime_replay
.venv/Scripts/python.exe -u -B ml/finalize_initial.py --out output/ml_initial_20261023/followup_01
```

Training remains runnable through the original command in the root README,
using a new destination and explicitly labelled as a replay of already
evaluated data. The continuation does not retrain. `report_initial.py` renders
archived training curves, source-bootstrap coverage, all-coordinate Bayesian
traces, and a detailed local `REPORT.md` with runtime and scientific limitations.

Meaningful continuation tests:

```powershell
.venv/Scripts/python.exe -B -m pytest -q ml/test_followup.py ml/test_ml.py tests/test_training_data.py tests/test_npe_contract.py --basetemp output/ml_initial_20261023/regression_NEW
```
