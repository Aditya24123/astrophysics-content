> Sanitized publication copy. Private identities, labels, machine paths and task prompts are omitted. Original hashes below identify local originals, not these edited copies. Some linked raw artifacts are intentionally local-only.

# Frozen initial H II posterior experiment — continuation

The neural model was retained after verifying its saved checkpoint, selected epoch 22 and stopping epoch 52. The original test evaluation is an archive of 96 observations from 48 source groups. No new blind evaluation, test-based selection, real-observation training or data generation occurred. A separate instrumented train/validation-only replay reproduced the frozen fit without replacing it; see runtime accounting below.

## Data, targets and fairness

The complete immutable 3,712-file inventory passed again. Train-only normalization was reproduced exactly, and PCA was reproduced from the 794 training observations. Splits remain 397/67/48 independent training/validation/test sources (794/134/96 observations); 24 challenge observations remain separate. Predictor inputs contain only 3,906 normalized measured Fourier components. True targets stay separate in the supervised loss.

| Target | Units | Prior | Transformation |
|---|---|---|---|
| em0_pc_cm6 | pc cm^-6 | Log-uniform [10000,500000] | log10 |
| te_k | K | Uniform [5000,12000] | Linear |
| major_fwhm_arcsec | arcsec, full major-axis EM FWHM | Log-uniform [12,60] | log10 |
| axis_ratio | 1 | Uniform [0.5,1] | Linear |

The verified loader standardizes these transformed labels using unique training parents. The frozen neural loss instead uses original labels mapped into prior-unit coordinates and then logit/(pi/sqrt(3)). This is a fixed bounded transform, not a holdout-fitted normalization. The original labels and simulator were preserved.

Both methods use the same raw observations, target definitions and continuous prior bounds, uniform PA [0,180), uniform east/north offsets [-10,10] arcsec, mean-one 5% common lognormal calibration and Gaussian spectral slope sigma 0.01, and the same correlated thermal covariance in Jy^2. Conventional inference samples all nine coordinates. Neural inference approximates the four-target nuisance marginal through the simulator training distribution; its train-only PCA retains only 39.22% of standardized training variance and can discard information. Its single Gaussian in logit coordinates restricts posterior shape. The training bank uses a finite scrambled Sobol design rather than iid continuous prior draws. These computational approximations prevent calling the two methods mathematically identical.

## Existing test archive (not a new blind evaluation)

| Target | MAE | RMSE | Median abs relative error | MAE source-bootstrap 95% CI |
|---|---:|---:|---:|---|
| em0_pc_cm6 | 38040.8 | 56108.2 | 29.10% | [27199.6, 49830.8] |
| te_k | 1698.3 | 1991.57 | 19.50% | [1409, 1989.24] |
| major_fwhm_arcsec | 6.9776 | 9.58989 | 21.61% | [5.31061, 8.89294] |
| axis_ratio | 0.125388 | 0.143495 | 15.55% | [0.105909, 0.143742] |

MAE/RMSE and interval widths use the units above. Temperature MAE is worse than the prior-midpoint diagnostic (1486.9 K). This is descriptive evidence, not a basis for subsequent test tuning.

| Target | Nominal level | Coverage | Source-bootstrap 95% CI | Mean interval width |
|---|---:|---:|---|---:|
| em0_pc_cm6 | 50% | 77.08% | [65.62%, 87.50%] | 99524.1 |
| em0_pc_cm6 | 80% | 93.75% | [87.50%, 100.00%] | 183073 |
| em0_pc_cm6 | 95% | 98.96% | [96.88%, 100.00%] | 266234 |
| te_k | 50% | 54.17% | [40.60%, 66.67%] | 3593.69 |
| te_k | 80% | 80.21% | [68.75%, 90.62%] | 5518.7 |
| te_k | 95% | 96.88% | [92.68%, 100.00%] | 6458.57 |
| major_fwhm_arcsec | 50% | 61.46% | [48.96%, 75.00%] | 17.0967 |
| major_fwhm_arcsec | 80% | 91.67% | [83.33%, 97.92%] | 29.3079 |
| major_fwhm_arcsec | 95% | 97.92% | [93.75%, 100.00%] | 38.2497 |
| axis_ratio | 50% | 44.79% | [31.25%, 59.38%] | 0.254612 |
| axis_ratio | 80% | 81.25% | [68.75%, 91.67%] | 0.393523 |
| axis_ratio | 95% | 94.79% | [87.50%, 100.00%] | 0.4613 |

The 1,000 bootstrap resamples use source groups; the two noise realizations are not independent skies. Coverage has only 48 independent test groups, and bootstrap intervals at 0 or 1 are unreliable near boundaries. 2,048 neural draws per observation add finite Monte Carlo uncertainty. Broad/overcovering intervals are not proof of calibration. PIT histograms, joint coverage, log10 errors (EM/size), and full metric definitions remain in the original test JSON. Validation metrics are in `validation/metrics.json`; they are for development, not blind claims.

!Training curves (local-only artifact: `training_curves.png`)

!Archived coverage (local-only artifact: `archived_test_coverage.png`)

## Bayesian follow-up and predictive checks

Both original 500-warmup/1,000-retained-draw pilots failed their original target gate. Slow mixing also affects nuisances, and shared MAP-region initialization gave weak mode exploration. Their posterior errors and predictive summaries remain provisional. The follow-up changes the sampler proposal only, using the pilot covariance for a fixed Student-t independence proposal with an exact Hastings correction plus symmetric local moves. Four independent chains start at measurement-only optimizations from dispersed physical/PA starts. No dependency was added. Batched thermal whitening is algebraically equivalent and regression-tested. Forward renders are checked against the original renderer at prior extrema/random points and saved posterior states.

The new gate requires rank/folded split Rhat <1.01 and bulk/tail ESS >=400 for all targets, nuisances and PA sine/cosine. This is stricter than the historical pilot gate and is recorded before new sampling. [Diagnostic method](https://arxiv.org/abs/1903.08008); [Stan guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html). Finite diagnostics cannot prove global mode discovery.

### Validation case 1: `SOURCE_ID_REDACTED`

Convergence gate: **PASS**. 4000 retained draws per chain; wall 485.91 s; CPU 402.45 s. Maximum checked reference-render discrepancy 6.48e-14 in whitened L2 units.

| Variable | Rank/folded Rhat | Bulk ESS | Tail ESS | Mean MCSE |
|---|---:|---:|---:|---:|
| em0_pc_cm6 | 1.00442 | 1300.2 | 1256.3 | 108.5 |
| te_k | 1.00251 | 1102.0 | 1058.0 | 57.07 |
| major_fwhm_arcsec | 1.00360 | 1151.3 | 1048.2 | 0.02332 |
| axis_ratio | 1.00464 | 902.4 | 986.1 | 0.001266 |
| pa_branch | 1.00346 | 1185.0 | 1072.8 | 0.005697 |
| east_prior_unit | 1.00377 | 1452.1 | 2133.1 | 0.00044 |
| north_prior_unit | 1.00363 | 1352.7 | 1375.2 | 0.0004513 |
| common_gain_z | 1.00201 | 1289.8 | 1279.2 | 0.02743 |
| slope_z | 1.00142 | 1496.5 | 1307.0 | 0.02603 |
| pa_sin | 1.00288 | 1306.3 | 1069.1 | 0.01832 |
| pa_cos | 1.00262 | 1479.1 | 1508.8 | 0.01651 |

Neural median inference: 0.00019255 s for the marginal median; 0.00077385 s including 2048 draws. Both include normalization and PCA, exclude observation disk loading, use the same CPU/two-thread environment, and repeat 30 times after warmup. Neural draw count is not MCMC ESS; no matched-precision speedup is asserted.

| Target | Bayesian median [95% interval] | Neural median [95% interval] | Bayes abs error | Neural abs error |
|---|---|---|---:|---:|
| em0_pc_cm6 | 36858.5 [29735.1, 44631] | 24293.8 [11808.6, 145070] | 5451.84 | 7112.91 |
| te_k | 8467.44 [5141.01, 11823.6] | 8864.91 [5285.24, 11817.4] | 1855.02 | 2252.49 |
| major_fwhm_arcsec | 24.4431 [22.899, 26.0335] | 22.9002 [12.8978, 51.7662] | 0.340711 | 1.88365 |
| axis_ratio | 0.952091 [0.862778, 0.997153] | 0.768799 [0.518901, 0.983092] | 0.0554695 | 0.127822 |

Full-joint Bayesian posterior predictive discrepancy tail fraction: 0.5078 from 128 selections. This checks replicated thermal residual energy relative to the same sampled expected signal; it is not the old sqrt(2)-scale replicate-minus-data RMS presented as a unit-noise test. Detailed draws and interval inclusion/widths are in the case JSON. No coverage estimate is justified by two validation sources.

!All-coordinate chain traces (local-only artifact: `bayes_01_traces.png`)

### Validation case 2: `SOURCE_ID_REDACTED`

Convergence gate: **PASS**. 4000 retained draws per chain; wall 630.83 s; CPU 528.25 s. Maximum checked reference-render discrepancy 6.31e-14 in whitened L2 units.

| Variable | Rank/folded Rhat | Bulk ESS | Tail ESS | Mean MCSE |
|---|---:|---:|---:|---:|
| em0_pc_cm6 | 1.00263 | 1024.2 | 897.3 | 119.9 |
| te_k | 1.00295 | 943.2 | 853.9 | 62.98 |
| major_fwhm_arcsec | 1.00343 | 1385.1 | 1621.2 | 0.02355 |
| axis_ratio | 1.00530 | 1194.7 | 1347.4 | 0.001689 |
| pa_branch | 1.00486 | 1210.6 | 989.0 | 0.006678 |
| east_prior_unit | 1.00157 | 1305.4 | 976.8 | 0.0005811 |
| north_prior_unit | 1.00710 | 1539.0 | 2233.4 | 0.0005473 |
| common_gain_z | 1.00539 | 1938.2 | 2767.2 | 0.0228 |
| slope_z | 1.00134 | 1895.1 | 2557.0 | 0.02274 |
| pa_sin | 1.00328 | 1715.7 | 1348.3 | 0.01626 |
| pa_cos | 1.00321 | 1405.4 | 1697.6 | 0.0184 |

Neural median inference: 0.0001906 s for the marginal median; 0.00076935 s including 2048 draws. Both include normalization and PCA, exclude observation disk loading, use the same CPU/two-thread environment, and repeat 30 times after warmup. Neural draw count is not MCMC ESS; no matched-precision speedup is asserted.

| Target | Bayesian median [95% interval] | Neural median [95% interval] | Bayes abs error | Neural abs error |
|---|---|---|---:|---:|
| em0_pc_cm6 | 31100.6 [24471.4, 39479.3] | 22481.8 [11305, 163799] | 4509.94 | 13128.7 |
| te_k | 8422.63 [5219.44, 11791.3] | 8352.82 [5185.23, 11806.6] | 1930.47 | 2000.28 |
| major_fwhm_arcsec | 15.9615 [14.3853, 17.8779] | 25.8998 [12.924, 55.1693] | 0.266157 | 10.2044 |
| axis_ratio | 0.927698 [0.782861, 0.996278] | 0.740369 [0.514937, 0.984656] | 0.0771894 | 0.11014 |

Full-joint Bayesian posterior predictive discrepancy tail fraction: 0.4688 from 128 selections. This checks replicated thermal residual energy relative to the same sampled expected signal; it is not the old sqrt(2)-scale replicate-minus-data RMS presented as a unit-noise test. Detailed draws and interval inclusion/widths are in the case JSON. No coverage estimate is justified by two validation sources.

!All-coordinate chain traces (local-only artifact: `bayes_02_traces.png`)

Completed forward profiles: **48/48**, retaining the 46 original records and computing only the two missing cases. 0 optimizer statuses were unsuccessful; all residual values are retained. Median/max profiled whitened RMS: 1.04/3.092. Targets stay at saved neural marginal medians while nuisances are fitted to the same data. These are optimistic profile residuals, not neural posterior predictive draws. A full neural nuisance-correlated predictive check remains unimplemented.

## Runtime and limits

Original full neural optimizer loop: 0.563541 wall s, 0.515625 CPU s. That timer EXCLUDED PCA and data loading; it is not the total training cost. The present training-only PCA verification refit took 0.316863 wall s. The original end-to-end elapsed time was never saved because the run was interrupted, so it cannot be reconstructed exactly. The half-data diagnostic is a separate development cost. The follow-up sampler uses its saved pilot draws to build a proposal, so the pilot cost must also be included in its from-scratch inference accounting; see the measured accounting below. Per-case follow-up timings include initialization, warmup and retained sampling; full setup/reference-check/predictive timing is saved separately. Inference comparisons use exactly the same validation measurements; two sources do not establish general runtime or accuracy superiority.

## Pending scientific work and protected next evaluation

A full neural posterior predictive treatment of nuisance correlations, broader validated conventional comparison, and stronger source-level calibration evidence remain pending. No real-nebula readiness or predetermined accuracy threshold is claimed. The small smooth pure-hydrogen/isothermal Gaussian source family and Fourier instrument proxy do not establish telescope or morphology transfer.

If a later model is developed using train/validation, propose a separately generated final evaluation bank with an independent private design seed, independent source IDs and all realizations grouped; freeze sample size, prior/instrument/noise law, metrics, code and model hashes first. An evaluator should protect its labels until one final unsealing. Choose sample size from a source-level coverage precision analysis (e.g. roughly 300–500 independent sources for useful few-percent marginal coverage precision, with multiple-target and design caveats). This is a proposal only: no such data generation is authorized or performed.

## Reproduction and artifact provenance

`binding.json` records original archive hashes, continuation source/plan hashes, environment and thread settings. `audit.json` records source-group, normalization and inventory verification. Original `run_03/run_binding.json` records Python/NumPy/SciPy/Astropy/Matplotlib versions, seed 231026 and original code hashes. Follow-up seed 231027 and chain offsets, fixed sampler settings, RNG end states, arrays and diagnostics are saved locally. See the root README and `ml/README.md` for executable commands.

<!-- final-runtime-accounting -->
## Additional precision and predictive scope notes

Both algorithms start from the same stored observations. The verified neural loader emits normalized float32 inputs before float64 network/PCA arithmetic; the conventional likelihood promotes the stored measurement vector directly to float64. This finite-precision difference is additional to the explicitly documented PCA/posterior-family and finite-design approximations. No extra truth metadata enters either method. Neural forward profiles are not full neural posterior predictive draws; that check remains pending.

The nuisance-profile optimizers and MAP initializations use calibration-coordinate bounds [-7,7] for numerical stability. Actual Bayesian sampling uses the unbounded standard-normal calibration priors. These bounded optimizations are diagnostics/initialization only and are never substituted for posterior inference.

## Measured training replay and complete inference accounting

An explicitly instrumented train/validation-only replay reproduced the frozen fit; it did not replace that model or reopen the test evaluation. The replay timing includes train-only normalization fitting, verified repeated data reads, PCA, optimization, validation stopping and checkpoint I/O. Startup imports and initial dataset generation/readiness are excluded. Cache and system load affect the I/O time.

- End-to-end replay: 83.892 wall s; 69.812 CPU s.
- Training-only normalization fit: 35.600 s.
- Training/validation verified loading: 43.143 s.
- PCA + optimizer + checkpoint I/O: 1.896 s.
- Selected/stopped epochs: 22/52; maximum array difference from original: 0.

| Validation case | Required historical pilot sampler [s] | Follow-up sampler [s] | Sum of recorded sampler work [s] | Neural median + 2048 draws [s] |
|---|---:|---:|---:|---:|
| 1 | 76.369 | 485.913 | 562.282 | 0.00077385 |
| 2 | 80.849 | 630.832 | 711.681 | 0.00076935 |

Follow-up proposal construction depends on the saved pilot, so its historical per-source cost belongs in a from-scratch comparison. Historical and follow-up times were measured in different sessions and system/cache conditions; sums describe recorded work, not a controlled repeated from-scratch benchmark. Neural training is an offline cost; per-observation inference is separate. Neural approximate independent draws and MCMC effective samples have different accuracy, so no numerical speedup or superiority claim is justified.

Full timing scopes and per-case setup/reference/predictive costs are in `runtime_accounting.json`. The training replay arrays, epoch log, configuration/code hashes and timing are in `training_runtime_replay/`. These measurements provide cost accounting, not a matched-accuracy speed claim.
