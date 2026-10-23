> Sanitized publication copy. Private identities, labels, machine paths and task prompts are omitted. Original hashes below identify local originals, not these edited copies. Some linked raw artifacts are intentionally local-only.

# H II inference: final BEFORE / AFTER audit

**Execution: COMPLETE for this bounded local audit and repair. All scientific limitations have NOT been eliminated.**

**Prominent unresolved result:** temperature and axis-ratio validation MAEs remain worse than their matched prior-median diagnostics, both for the frozen model and after one predeclared target-transform probe. The original disadvantages have source-paired intervals that include zero, so they do not alone prove a bad posterior. The probe gives mixed development improvements, not a validated general accuracy gain or a new canonical model.

The simulator, all original candidate/preparation files, `run_03`, and EVERY `followup_01` artifact are unchanged by SHA256. The original checkpoint remains canonical. All four original neural predictive outer draws, every failed conditional attempt, both continuations and the new angle-mixing phase remain saved. No existing test prediction was recomputed or used for selection. No real-data work, publication, dependency change or new scientific/protected evaluation bank was performed. Regression tests generated only their documented tiny throwaway fixtures, which were not training/evaluation data.

## Actual model errors and uncertainty

MAE uses the marginal posterior MEDIAN, not the posterior mean or a fitted physical optimum. The matched prior median is the geometric midpoint for log-uniform EM0/size and arithmetic midpoint for linear Te/q. Units below are original physical units. The comparison uses 134 validation observations from **67 source groups**; source-paired resampling keeps each source’s two noise realizations together.

| Target | Units | Frozen model MAE | One probit probe MAE | Prior median MAE | Probe minus frozen: resampling 95% interval |
|---|---|---:|---:|---:|---|
| em0_pc_cm6 | pc cm^-6 | 33718 | 32218.8 | 87035.2 | -1499.23 [-2593.13, -402.514] |
| te_k | K | 1832.04 | 1831.85 | 1788.08 | -0.19145 [-13.9749, 13.4094] |
| major_fwhm_arcsec | arcsec (full major EM FWHM) | 7.53589 | 7.57307 | 11.4158 | 0.0371801 [-0.115252, 0.179775] |
| axis_ratio | 1 | 0.130725 | 0.129264 | 0.128671 | -0.00146137 [-0.00286297, -0.000172245] |

Temperature’s original MAE excess is 43.966 K with source-resampling interval [-59.673793779541185, 146.74390251261408]; q’s excess is 0.002054 with interval [-0.005336430155002786, 0.009426812684251867]. These are descriptive intervals on reused validation/Sobol sources, not selection-adjusted iid population confidence guarantees.

The probe was frozen before training, made ONE fit, reused only training-fitted normalization/PCA and changed only scaled-logit to probit(prior-unit) target coordinates. It selected epoch 27 and stopped at 57. It makes the exact uninformative prior representable, but cannot restore PCA information or express arbitrary multimodal/curved posteriors. It improved EM0 and q MAEs in this development sample, barely changed Te, and slightly worsened size MAE. The original model was NOT replaced and the probe was NOT scored on the 96 test observations.

## Marginal and joint density scores: matched support/Jacobians

Scores are NLL relative to the exact prior in common prior-unit coordinates u, so the uniform prior score is 0. Negative is better. Both transforms’ exact label Jacobians are included. Raw Gaussian losses in logit and probit coordinates are not directly compared. A proper density objective can disagree with realized sample MAE; an exact posterior median nevertheless minimizes expected absolute error.

| Target | Frozen marginal prior-relative NLL [nats] | Probit probe marginal prior-relative NLL [nats] |
|---|---:|---:|
| em0_pc_cm6 | -0.544409 | -0.691923 |
| te_k | 0.023476 | 0.0315081 |
| major_fwhm_arcsec | -0.295433 | -0.38021 |
| axis_ratio | 0.0134858 | 0.0131094 |

Joint prior-relative NLL: frozen **-0.694412**, probe **-0.926993**; change -0.232581 nats with descriptive source-resampling interval [-0.36923732576900214, -0.08436833334464373]. The joint gain must not mask weak Te/q marginals, and validation reuse prevents calling this a fresh blind gain.

## What the model-focused diagnosis supports

- **Prior dominance / identifiability:** the two full-data Bayesian temperature marginals have histogram KL from prior about 0.005–0.007 nats and 95% widths about 94–95% of the full prior-unit range. Te is weakly identified in THESE observations; two cases do not prove a universal irreducible limit.
- **Axis ratio is different:** the two full-data Bayesian q marginals have KL about 0.89/1.31 nats, while the frozen neural q marginals are nearly prior-like. Thus universal physical non-identifiability is not a sufficient explanation for the neural q result.
- **Compression:** exact-covariance local Fisher calculations at three saved posterior states per case retain median nuisance-projected q information fractions 0.195/0.578 after PCA32. This measures local parameter information, not merely 39.22% training-variance retention. It is not a global posterior sufficiency proof.
- **Learning / family / data:** only 397 training sources, one original seed, a restricted Gaussian conditional family and reused validation constrain conclusions. One half-data comparison does not establish data sufficiency. That historical half-data fit also reused feature normalization fitted on all 794 training observations (while its PCA/weights used the half subset), so it is not a clean end-to-end sample-size learning curve. This is not validation/test leakage, but its preprocessing benefit must not be ignored. The probit probe shows prior representability alone is insufficient. The separate causal contributions of finite training size, optimizer/generalization and density-family restriction are not identified by this audit.
- **Implementation:** no point-median, units, target-pairing, gradient, precision-factor sampling or train/validation normalization/PCA defect was found in the frozen predictor. Numerical verification defects were repaired separately; they are not advertised as predictive-accuracy improvements.

## Coverage and sharpness: development evidence only

| Target | Nominal level | Frozen coverage | Probe coverage | Frozen mean width | Probe mean width |
|---|---:|---:|---:|---:|---:|
| em0_pc_cm6 | 50% | 66.42% | 55.22% | 76524.6 | 59463.9 |
| em0_pc_cm6 | 80% | 91.79% | 87.31% | 145836 | 113849 |
| em0_pc_cm6 | 95% | 92.54% | 91.79% | 227248 | 176134 |
| te_k | 50% | 53.73% | 43.28% | 3728.05 | 3404.87 |
| te_k | 80% | 82.84% | 80.60% | 5659.14 | 5476.59 |
| te_k | 95% | 95.52% | 95.52% | 6551.03 | 6561.08 |
| major_fwhm_arcsec | 50% | 63.43% | 58.96% | 18.1465 | 15.3698 |
| major_fwhm_arcsec | 80% | 86.57% | 84.33% | 30.5855 | 27.0036 |
| major_fwhm_arcsec | 95% | 94.78% | 93.28% | 39.1425 | 36.17 |
| axis_ratio | 50% | 46.27% | 44.78% | 0.249532 | 0.227329 |
| axis_ratio | 80% | 74.63% | 73.88% | 0.3883 | 0.373046 |
| axis_ratio | 95% | 91.79% | 91.79% | 0.458724 | 0.45705 |

Widths use the physical units in the error table. There are 67 validation sources, not 134 independent skies; 2048 neural draws/observation add Monte Carlo uncertainty. Existing source-resampling intervals are saved in each metrics JSON. Coverage is not broadly established, especially after stopping/model development on validation. The evaluated 96-observation/48-source test archive is unchanged; 24 mixed-layout challenges from 12 sources remain separate. No new blind evaluation was performed.

## Raw verification and numerical fixes

| Item | Before | After |
|---|---|---|
| Exact replay claim | Saved assertion | Direct NPZ comparisons: every original/replay weight and PCA array identical |
| Forward 48/48 claim | Saved profile records | All 48 expected source/first-realization IDs verified; original 46 identical; no new test rerender |
| 2/2 Bayesian passes | Saved diagnostic summaries | Recomputed from raw chains, all targets/nuisances/PA sin/cos pass; quantiles and 128 predictive receipts per case replay exactly |
| Omitted/invalid diagnostic variable | Could omit a bad dimension; stuck report not strict-JSON serializable | Every variable required; nonfinite/degenerate failures explicit and JSON-safe |
| Odd FFT inverse length | Example requested 63, inverted 62 | Explicit inverse length, direct autocovariance regression |
| Assumed nuisance/config priors | Hardcoded PA/offset and target transform silently assumed | New route rejects unsupported configs/units/layouts instead of silently changing priors |
| Predictive nuisance treatment | Median target + optimized nuisance point | Coherent q_NN(theta|C(y)) p(nuisance|theta,y) pilot, validated conditional chains, original failures retained |

The original 4000-draw Bayesian passes survive unchanged: their FFT padding was even. Historical code is preserved for its bindings; new checks live in the strict versioned route. Extra mode searches used eight starts per validation case; all 16 optimization statuses succeeded and returned to known regions. These optimizers are initialization/mode checks, NEVER posterior substitutes. Finite Rhat/ESS and searches cannot prove global mode completeness.

## Conditional predictive pilot: failures and actual final acceptance

All four neural target draws were frozen BEFORE conditional fitting and none was filtered, redrawn or reweighted. The first bounded pilot passed 2/4 cases. Exact same-kernel continuation preserved all old draws and RNG paths, passed one more case, and left one failure at the 300 s budget. A separately predeclared full-period PA proposal then passed that last case in a new homogeneous phase. Its priors, likelihood and target draw were unchanged; old failed chains are not counted as passing in the new-phase diagnostics.

| Original outer target | Accepted phase | Retained /chain | Max rank/folded Rhat | Min bulk ESS | Min tail ESS |
|---|---|---:|---:|---:|---:|
| conditional_01_01 | same-kernel exact continuation | 4000 | 1.008233 | 863.6 | 413.2 |
| conditional_01_02 | original conditional pilot | 3000 | 1.007420 | 845.5 | 454.2 |
| conditional_02_01 | new PA-refresh homogeneous phase | 1445 | 1.007846 | 581.0 | 572.7 |
| conditional_02_02 | original conditional pilot | 3000 | 1.005150 | 935.5 | 596.8 |

The gate remains Rhat <1.01 and bulk/tail ESS >=400 for every nuisance and PA sine/cosine. Margins are finite and sometimes narrow; a pass does not certify tail quantiles or global modes perfectly. **4/4 conditional passes do NOT mean a validated full neural predictive marginal:** there are only two theta draws per source and two conventional comparison observations. These are conditional/hybrid implementation checks.

Band/radius residual checks retain all four accepted theta cases and show some fixed-theta draws have larger structured residuals than the joint Bayesian samples. This is evidence about those draws and information mismatch, not an estimate of how often the neural posterior fails across sources.

![Validation residual checks](../predictive_details_01/band_residuals.png)

## Fairness and remaining information mismatch

The raw observations, four physical definitions/supports/continuous priors, PA/position priors, calibration priors and full forward/thermal law match. Common amplitude is mean-one lognormal, slope is Gaussian; thermal covariance retains cross-frequency correlation and self-mode variance. Imaging taper is not inverse variance. True parameters remain separate supervised labels; seeds/IDs/true sky/realized nuisances are excluded from predictor tensors. All preprocessing is fitted on training sources/observations only.

However, the NPE sees PCA-compressed C(y), whereas the conventional reference uses full y. The new hybrid conditional extension uses p(nuisance|theta,full y) but retains q_NN(theta|C(y)). It is a normalized approximate joint, not the exact full-data posterior or a fully amortized joint neural model. A raw-data discrepancy may reflect discarded information even if a compressed-data estimator were ideal. A validated conventional posterior on the matched compressed likelihood (with C Sigma C^T) is still needed to fully separate compression from density-estimation error. Local Fisher diagnostics do not replace that posterior comparison. Float32 neural input rounding vs float64 conventional arithmetic and the finite Sobol training design also remain explicit.

## Runtime, tests and encountered issues

No duplicate trainer was launched; jobs were serialized with the same lock. Original training replay cost 83.89 s included normalization and verified reads; its 0.56 s optimizer-only record is not end-to-end cost. The single probit probe used 56.19 s including loading/evaluation, with 1.79 s in the optimizer loop; its preprocessing scope differs from the old full replay. Archived neural target inference (~0.00077 s including 2048 approximate draws) excludes all conditional solves. Pilot-inclusive conventional recorded work remains 562.28/711.68 s per source across different sessions. New nuisance integration/continuation/PA-phase costs are saved separately and must not be hidden in a fast-neural headline. There is no matched-accuracy speedup claim.

Testing: 69 unique checks passed across the combined run and targeted runs. The combined 63-test run had **62 passes and one Windows temporary-fixture replacement failure**; a sandbox TEMP retry had a setup error; the identical lineage test then passed unchanged in a fresh shorter workspace path. This is NOT described as a simulator software defect fixed. Original source/test logic was preserved. Two probit/Jacobian and two PA-proposal tests passed separately, as did two correlated-covariance Fisher-information regressions. Prior known-distribution/RNG segmentation tests passed in the combined run.

- **I10:** The first v2 integration attempt completed raw-chain and 128-draw predictive verification for both validation cases but its final dataset fingerprint failed: the new holdout guard rejected raw checksum reads. STATUS remained actively_executing despite exception. Handling: Allow only the frozen hii_sim.io.digest function's raw-byte open for sealed files; keep all label decoding blocked. Add an isolated subprocess regression proving SHA256 succeeds while read_text raises. Record failed stage status on exceptions and retry in v2_02 without overwriting v2. Logs: verify_console.log retains the PermissionError; v2/source_snapshot retains the failed attempt's exact source files; v2/binding.json and independent_verification.json retain its partial evidence. No protected label was decoded and no preserved data was changed.
- **I11:** Combined regression: 62 passed and one unchanged lineage test failed during Windows atomic replacement of a temporary fixture manifest. A sandbox TEMP retry failed during temporary-directory setup. Handling: No simulator or test logic changed. The identical lineage test passed when retried in a fresh shorter workspace fixture directory (output/review_retry01). Root cause of Windows I/O denial was not established; this is not claimed as a software defect fixed. Logs: tests_final.log, test_file_lock_retry.log, test_short_path_retry.log retain 62-pass/one-I/O-failure, one setup error, and one unchanged passing retry.

Scientific failed attempts are also preserved: the two historical Bayesian pilots, two initial conditional failures and one bounded exact-continuation failure. Failed intermediate milestones remain in console logs. Their later accepted phases do not erase those failures.

## Reproducible commands and saved state

All commands run from project root with the existing pinned environment. Use new versioned outputs for a new attempt; never overwrite these artifacts. `ml/README.md` documents the BEFORE/verify/mode/conditional stages. The subsequent commands used here were:

```powershell
$env:PYTHONPATH='src;ml'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OPENBLAS_NUM_THREADS='2'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
.venv/Scripts/python.exe -u -B ml/resume_conditionals.py
.venv/Scripts/python.exe -u -B ml/diagnose_model.py
.venv/Scripts/python.exe -u -B ml/probit_probe.py
.venv/Scripts/python.exe -u -B ml/angle_refresh.py
.venv/Scripts/python.exe -u -B ml/predictive_details_final.py
.venv/Scripts/python.exe -u -B ml/assemble_review.py
```

Frozen protocols are `model_diagnosis_plan.json`, `probit_probe_plan.json`, `angle_refresh_plan.json` and `review_plan_v2.json` in `ml/`. Software versions, source hashes, seeds, saved RNG states, original/probe checkpoints, raw epoch curves, coverage/sharpness, conditional chains and test logs are retained locally. The probe’s FROZEN_PROTOCOL timestamp precedes its training-start log. Its checkpoint is `../probit_probe_01/full/best.npz`; the unchanged canonical checkpoint remains under `run_03/full/best.npz`.

## Exact remaining limits and next steps

1. **Te/q performance remains unresolved.** Te is prior-dominated in the two inspected full-data cases, not proven globally unidentifiable; q has full-data information that the NPE does not fully exploit. Point MAE disadvantages alone are not a posterior-validity test.
2. **Cause partition remains incomplete.** Compression, family/optimization and finite training data contribute plausibly; a matched compressed-likelihood Bayesian comparison, more predeclared validation cases and training-seed/learning-curve studies are needed to separate them.
3. **No broad calibration claim.** Four outer theta draws, two conventional cases, reused 67-source validation, finite MCMC tails and single-seed fits do not establish calibrated full neural uncertainty. The NPE is not a Bayesian distribution over network weights; model/training uncertainty is not separately characterized.
4. **No real-object transfer.** The smooth pure-H isothermal Gaussian family, synthetic Fourier operator and known synthetic uncertainty laws do not certify real telescope/morphology/noise modeling.
5. **Future protected evaluation needs approval.** After a future model/metric freeze, propose a separate independent source set with private seed, unchanged intended physical priors/noise/operator and sealed evaluator-held labels. Prefer independently sampled prior sources if formal iid coverage statements are required; group all realizations. No such scientific bank was generated or approved here.

**Completed:** independent archive verification, scoped diagnostic/contract repairs, honest failure accounting, nuisance-correlated predictive implementation and all four accepted conditional cases, source-aware model diagnosis, one frozen validation-only sensitivity probe, and this measured before/after record. **Pending:** the scientific limits above. **Blocker to claiming every issue eliminated:** the available information/model family and lack of independent final validation; more prose or regression tests cannot remove those limits.

Primary diagnostic references: [Stan posterior analysis](https://mc-stan.org/docs/reference-manual/analysis.html) and [rank/folding/localization](https://arxiv.org/abs/1903.08008).
