> Sanitized publication copy. Private identities, labels, machine paths and task prompts are omitted. Original hashes below identify local originals, not these edited copies. Some linked raw artifacts are intentionally local-only.

# Local repair campaign — measured before / after

Final record: 2026-10-23T13:22:29.603343+00:00. Deadline 2026-10-23T14:00:00+00:00; science stop 2026-10-23T13:30:00+00:00.

**The campaign completed; the scientific limitations did not. The canonical epoch-22 model and original test archive remain unchanged. The single thermal PCA128 remedy failed validation and is not promoted.**

The baseline is the preserved `review_01/final_01` delivery. Original simulator, 3,712 dataset files, prepared normalization, `run_03`, `followup_01` and all of `review_01` were checked by hashes. No dependency was installed or replaced, no real data was trained, and no bank or publication was created. Exclusive process lock and process inventory were used; scientific jobs ran serially.

## Contract and fairness

| Target | Physical unit | Prior / forward transformation |
|---|---|---|
| `em0_pc_cm6` | pc cm^-6 | Log-uniform [10000,500000]; log10 into prior-unit coordinate |
| `te_k` | K | Uniform [5000,12000]; linear prior-unit coordinate |
| `major_fwhm_arcsec` | arcsec, full major EM FWHM | Log-uniform [12,60]; log10 into prior-unit coordinate |
| `axis_ratio` | 1 | Uniform [0.5,1]; linear prior-unit coordinate |

Training is 794 observations /397 sources; validation 134/67; the previously evaluated test archive is 96/48. The 24 observations /12 sources of different-layout challenges remain separate and were not scored. All preprocessing was fitted to training measurements only. Targets enter the supervised loss separately; source IDs, true parameters, seeds, images and realized gains never enter the feature matrix. Fixed thermal whitening uses declared observing metadata. The post-fit fixed-operator audit verifies identical covariance/layout/beam/taper/bands over development inputs.

The starter target API supplies train-parent-standardized log10 EM/log10 major FWHM and linear Te/q, with statistics fitted on 397 unique training parents. Those prepared statistics remain unchanged. The original network instead uses the equivalent fixed prior-unit scaled-logit coordinates; this repair uses fixed prior-unit probit coordinates. These are reversible output-coordinate changes, not changes to target definitions or priors. The new input is thermal-whitened PCA128 with training mean, basis and component scaling. Its 91.09% variance retention is not a sufficiency claim. The posterior remains a hidden32 full-covariance Gaussian in probit(prior-unit) coordinates (4590 parameters), one fit at seed 231026. Original and earlier probit32 models remain intact. Common prior-unit density comparisons include correct label Jacobians.

For each old32/new128 compression, the Bayesian likelihood is exactly Gaussian in Cᵀy with covariance CᵀΣthermalC, the same forward physics and the same target/nuisance priors. PA is uniform [0,180), east/north uniform ±10arcsec, common amplitude is mean-one lognormal with 5% fractional standard deviation, and log-slope is normal with standard deviation0.01. Calibration remains explicit, never an independent added noise floor. Seven uniform-prior coordinates use u=Φ(z); together with the two normal calibration coordinates this has exact N(0,I9) density, including Jacobians. The four-target neural posterior learns a nuisance-marginal conditional from simulations drawn under those intended population priors; Bayesian inference samples all nine coordinates and marginalizes nuisances in its reported four targets. Finite Sobol training design and the restricted learned family are approximation limits, not an exact finite empirical-prior match. Full-y and compressed comparisons are separate. Historical float32 feature rounding is quantified in `rounding_and_runtime.json`; compressed Bayes targets the ideal float64 affine summary, not the exact likelihood of a quantizer.

## Single neural remedy on reused validation

| Target | Unit | Original MAE | Earlier probit32 MAE | Thermal128 MAE | Prior-median MAE | Thermal128 RMSE |
|---|---|---:|---:|---:|---:|---:|
| em0_pc_cm6 | pc cm^-6 | 33718 | 32218.8 | 35336 | 87035.2 | 66395.5 |
| te_k | K | 1832.04 | 1831.85 | 1829.81 | 1788.08 | 2066.22 |
| major_fwhm_arcsec | arcsec (full major EM FWHM) | 7.53589 | 7.57307 | 8.44283 | 11.4158 | 11.4207 |
| axis_ratio | 1 | 0.130725 | 0.129264 | 0.13329 | 0.128671 | 0.153846 |

Selected epoch 18; stopped 48. Joint prior-unit NLL (prior=0): original -0.694412; earlier probit32 -0.926993; thermal128 -0.502224. Thermal128 minus original change 0.192188 with descriptive source-resampling95 interval [0.003910532585910079, 0.36731871844701036]. Higher NLL is worse. No second architecture/seed/hyperparameter fit followed this failure.

| Target | Thermal128 marginal prior-unit NLL | MAE change vs original | Source-resampling95 interval for change |
|---|---:|---:|---|
| em0_pc_cm6 | -0.523122 | 1618.03 | [-2999.35,6838.39] |
| te_k | 0.0356555 | -2.23446 | [-111.246,105.686] |
| major_fwhm_arcsec | -0.225429 | 0.906943 | [0.0771147,1.80929] |
| axis_ratio | 0.0619815 | 0.00256505 | [-0.00478236,0.0105567] |

![Training and validation curves](training_curves.png)

| Target | Nominal | Validation coverage | Source-resampling95 interval | Mean width, stated physical unit |
|---|---:|---:|---|---:|
| em0_pc_cm6 | 50% | 70.15% | [0.5970149253731343, 0.7985074626865671] | 81929.7 |
| em0_pc_cm6 | 80% | 88.81% | [0.8134328358208955, 0.9552238805970149] | 154997 |
| em0_pc_cm6 | 95% | 94.03% | [0.8805970149253731, 0.9850746268656716] | 241921 |
| te_k | 50% | 40.30% | [0.291044776119403, 0.5149253731343284] | 3303 |
| te_k | 80% | 79.10% | [0.6940298507462687, 0.8805970149253731] | 5364.1 |
| te_k | 95% | 94.78% | [0.8880597014925373, 0.9925373134328358] | 6493.29 |
| major_fwhm_arcsec | 50% | 59.70% | [0.48507462686567165, 0.7089552238805971] | 17.9436 |
| major_fwhm_arcsec | 80% | 82.84% | [0.7388059701492538, 0.9104477611940298] | 31.1305 |
| major_fwhm_arcsec | 95% | 94.78% | [0.8955223880597015, 0.9925373134328358] | 40.7463 |
| axis_ratio | 50% | 43.28% | [0.3208955223880597, 0.5447761194029851] | 0.227236 |
| axis_ratio | 80% | 73.88% | [0.6343283582089553, 0.835820895522388] | 0.373398 |
| axis_ratio | 95% | 88.06% | [0.7985074626865671, 0.9477611940298507] | 0.45809 |

All validation values reuse sources used for stopping/debugging and are development evidence. Resampling treats the two variants as one source group. The intervals do not correct model selection or establish iid population uncertainty for the prior Sobol design. `validation/source_stress.json` reports fixed prior-quartile errors/coverage with actual source counts; these small subgroups support no reliable population claim. PIT and joint transformed-region coverage remain in saved metrics.

## Matched Bayesian posterior checks

Every accepted route requires four chains; rank/folded split Rhat<1.01 and bulk/tail ESS≥400 for all targets, all nuisance coordinates, PA sine/cosine and log likelihood. The full-y log likelihood was additionally replayed over saved unique posterior states for the same complete diagnostic set. Initial failed phases and any fixed refresh phases are separate; no heterogeneous production chains were pooled, no failed posterior is substituted, and no deterministic optimizer is called Bayesian inference. Finite diagnostics and multistart optimization do not prove global mode completeness.

| Case | Data route | Phase | Gate | Draws/chain | Max Rhat | Min bulk ESS | Min tail ESS |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | old32 | refresh_01 | PASS | 3000 | 1.00683 | 891.417 | 656.976 |
| 1 | noise128 | not admitted | DEFERRED, rejected-model route | — | — | — | — |
| 1 | full_y | historical_full_y | PASS | 4000 | 1.00464 | 902.446 | 986.107 |
| 2 | old32 | refresh_01 homogeneous continuation_01 | PASS | 3901 | 1.00496 | 1134.73 | 550.01 |
| 2 | noise128 | not admitted | DEFERRED, rejected-model route | — | — | — | — |
| 2 | full_y | historical_full_y | PASS | 4000 | 1.0071 | 943.176 | 853.868 |

`matched_comparison.json` contains complete gates, posterior quantiles, marginal empirical-CDF distances and width ratios for each passing route. Original/probit32 NPE vs old32 Bayes uses the same ideal affine summary, up to the explicitly quantified historical float32 rounding. Both optional thermal128 Bayesian routes were deferred, not launched: the neural remedy was already rejected, and admission priorities reserved the concrete half-data correction, stress checks, regressions and final verification. They are not convergence failures or validated comparisons. Full-y comparisons explicitly include an information mismatch. Only two preselected validation sources support these case studies; no test-wide winner or posterior calibration is established. Failed phases and their timings remain saved; prefix gates were replayed.

### Canonical matched case errors and posterior shape

These are absolute errors on only two preselected validation sources, in stated units. Full-y Bayes has additional information and is shown separately, not as a matched-data winner. Neural point estimates and95% intervals below are analytic marginal medians/quantiles of the frozen density; Bayesian quantiles are Monte Carlo estimates from accepted chains.

Individual-case parameter errors withheld from public copy; aggregate errors remain above.

| Case | Target | Neural / matched Bayes95% width | Empirical marginal CDF distance | Full-y / matched Bayes95% width |
|---:|---|---:|---:|---:|
| 1 | em0_pc_cm6 | 7.89585 | 0.594682 | 0.900861 |
| 1 | te_k | 0.979916 | 0.0457738 | 0.999014 |
| 1 | major_fwhm_arcsec | 5.18059 | 0.435369 | 0.429837 |
| 1 | axis_ratio | 1.67597 | 0.3942 | 0.486857 |
| 2 | em0_pc_cm6 | 9.63563 | 0.51677 | 1.02469 |
| 2 | te_k | 0.997385 | 0.0369681 | 0.989386 |
| 2 | major_fwhm_arcsec | 8.76314 | 0.682193 | 0.71735 |
| 2 | axis_ratio | 1.77232 | 0.49084 | 0.802248 |

Empirical CDF distances use saved 16,384 neural draws and correlated accepted Bayesian samples; there is no IID p-value. Two sources cannot estimate calibrated Bayesian interval coverage. Raw quantiles and physical truths/errors are in `matched_case_errors.json`; no unconverged posterior or optimizer appears in this table.

| Case | Local nuisance-profiled retained information, old32: EM/Te/size/q | Thermal128: EM/Te/size/q |
|---:|---|---|
| 1 | 0.567365, 0.564423, 0.197838, 0.226224 | 0.999077, 0.998783, 0.963457, 0.943547 |
| 2 | 0.739885, 0.738194, 0.453072, 0.577877 | 0.995851, 0.995485, 0.942752, 0.926786 |

These local derivatives use fixed 20/50/80% positions in each saved flattened full-y chain, sharing identical states between old32 and thermal128. The earlier review used 12.5/50/87.5% positions, so its separately reported median is not a numerical replay of this state set. Exact compressed covariance and calibration-prior nuisance projection separate local information retention from learning error; they do not prove global sufficient statistics. A more informative representation can still be difficult to learn from 397 sources with this restricted conditional family.

## Forward prediction and temperature

| Case | Bayesian data route | Fitted discrepancy PP tail fraction | Full-y signal residual RMS median [whitened sigma] |
|---:|---|---:|---:|
| 1 | old32 | 0.703125 | 1.0016 |
| 1 | full_y | 0.5625 | 0.999142 |
| 2 | old32 | 0.453125 | 1.00184 |
| 2 | full_y | 0.40625 | 1.00139 |

These use128 joint posterior draws of all nine parameters per accepted route, with replicated thermal noise and saved band residuals. For compressed fits, the fitted discrepancy uses compressed data; the full-y residual reveals discarded information and is labelled separately. Correlated posterior draws are not independent observed sources. Original48/48 nuisance-profile forward checks and the four fixed original-neural nuisance-conditional pilot draws remain in the previous audit: profile residuals are not posterior predictive calibration, and four neural outer draws remain too few to estimate calibrated tails. The failed thermal128 model has no validated large neural predictive experiment.

![Matched and full-data Bayesian residual checks](bayesian_residual_checks.png)

The shaded bands are10–90% ranges across128 posterior draws, not source-level confidence intervals. The horizontal unit-noise RMS reference is not an acceptance threshold. Two cases and near-unit aggregate residuals do not establish predictive calibration.

Temperature operational warning: 129/134 validation observations meet KL(q_u||uniform)<0.1nats AND central95 width/prior95 width>0.8. The exact Gaussian marginal KL is invariant under the prior CDF transformation. These are explicit heuristic informativeness flags, not accuracy thresholds. All four posterior outputs remain; Te is neither fixed at 8,500 K nor dropped. `repair_cli.py predict` exposes the full posterior, flags and an abstention from constrained-Te precision; even unflagged neural temperature precision is unvalidated.

The preserved transfer equation gives optically thin Iν∝EM·Te^(-1/2)·gff(Te,ν); flux, morphology and calibration can trade against its weak temperature dependence. `temperature_identifiability.json` quantifies prior-corner optical depths and states additional-observation hypotheses: resolved optically thick brightness/frequencies bracketing turnover, or an independently temperature-sensitive diagnostic with a new jointly specified model. These are equation-based hypotheses, not authorization to generate or analyze new observations.


After profiling all other targets and nuisances, local unbounded Gaussian Te scales at these six saved states range 68,166–144,959K, wider than the 7,000 K prior range. These are local likelihood-curvature diagnostics, **not posterior credible intervals**; they do not prove global non-identifiability. Accepted matched-old32 Te marginals have coarse20-bin prior KL about0.0033/0.0027nats and prior-unit95% widths0.956/0.949; full-y values remain near prior too.

## Learning-curve correction, runtime and checks

The corrected half run uses396 observations /198 sources with its own feature mean/std, PCA and unique-parent target statistics. The model itself uses the same fixed prior-logit target coordinates, not those target standardization statistics. Selected epoch 40, stopped 70; validation MAE [30094.529337689288, 1886.7220737391604, 7.8637547575135205, 0.1274826802847079]. Its NLL 5.12997 is directly comparable with the original fixed-target-coordinate full model. One half/full pair at one seed is not a data-sufficiency result. The original confounded half run is preserved.

![Full and correctly preprocessed half-data curves](half_corrected/training_curves.png)

`failure_timing_ledger.json` retains the phase-level gate, failed-variable list, draw count and incremental wall time, including original pilots, new failures and the single exact continuation. Deferred routes are distinguished from failed convergence.

New preparation (verified loading, train-only whitening/PCA) 31.4255s wall/30.9688s CPU. Neural optimization 0.418993s wall/0.375s CPU. Median2048-draw neural inference on an already projected observation 0.00093215s, excluding disk/preprocessing and nuisance inference. The CLI receipt additionally times raw affine transformation plus posterior sampling. These timer scopes differ from the historical83.89s verified end-to-end training replay; they must not be divided into a matched-accuracy speedup. CPU-only with two BLAS threads, same package environment.

| Case | Bayesian route | Production phase wall [s] | Total recorded sampler work incl. failed phase [s] |
|---:|---|---:|---:|
| 1 | old32 | 561.218 | 1294.26 |
| 1 | noise128 | pending | pending |
| 1 | full_y | 485.913 | 562.282 |
| 2 | old32 | 790.745 | 1521.16 |
| 2 | noise128 | pending | pending |
| 2 | full_y | 630.832 | 711.681 |

Offline likelihood diagnostic replay and posterior-predictive analysis are additional evaluation costs, saved separately. Sampler wall time includes initial optimizations/warmup in its phase; failed work is counted. Training amortization, different summaries, two cases, and unmatched approximation accuracy preclude a universal neural/Bayesian speedup claim.

Current regression receipt: `{'utc': '2026-10-23T13:06:27.852200+00:00', 'command': ['.\\.venv\\Scripts\\python.exe', '-B', '-m', 'pytest', '-q', 'tests/test_npe_contract.py', 'tests/test_training_data.py', 'ml/test_ml.py', 'ml/test_followup.py', 'ml/test_review_v2.py', 'ml/test_resume_conditionals.py', 'ml/test_probit_probe.py', 'ml/test_angle_refresh.py', 'ml/test_model_information.py', 'ml/test_repair_campaign.py', 'ml/test_repair_refresh.py', 'ml/test_repair_resume.py', '-p', 'no:cacheprovider', '--basetemp', 'LOCAL_PATH_REDACTED'], 'exit_code': 0, 'passed': True, 'wall_seconds': 79.53072320000138, 'fixture_root': 'LOCAL_PATH_REDACTED', 'fixture_policy': 'Fresh explicitly chosen path, no cleanup/deletion, no physics/source-I/O changes, no automatic failure suppression. A numerical failure is never retried as an environment failure.', 'log': 'regressions.log'}`. Earlier audit evidence remains69 unique passing checks across a combined run and targeted retries; its combined run had one WinError5 fixture-manifest replace failure before the unchanged short-path retry passed. Current tests use a fresh non-synced fixture path, without changing simulator I/O or machine security. Tiny numerical/test fixtures are not a generated scientific evaluation bank. The raw float32 preparation failure is retained under `repair_02_failed_preparation_01`, with its failed source reconstructed and hash-verified; its tolerance was not loosened.

Provenance distinction: current package-wide source-manifest6b03... differs from historical generation0a31... because of later added training/photometry modules and preparation tooling, plus a CASA validator change; all common original physics files match. These differences predate this campaign. `GENERATOR_ENTRYPOINT_AUDIT.json` enumerates them. The future proposal explicitly selects the preserved original source snapshot; its0a31... digest/config/parser were verified without generation. The failed preparation also consumed a separately recorded UTC-receipt interval in `INCIDENTS.json`; that failed development cost is additional to the successful preparation timer.

### Same-observation neural latency and one-off costs

| Model | Warm median raw-input-to2048-draw wall [s] | Warm p90 [s] |
|---|---:|---:|
| canonical_logit32 | 0.00073875 | 0.0007731 |
| rejected_thermal_probit128 | 0.0038123 | 0.00401017 |

One fixed public validation observation,21 repeats, same two-thread CPU process; raw feature transform/PCA, RNG setup, posterior sampling and physical decoding are included. Disk/model/likelihood loading, informativeness flags and conventional nuisance inference are excluded. The executable prediction receipt separately verifies the warning-bearing command. Canonical32 can be placed beside matched old32 Bayes accounting, but its posterior fidelity differs substantially; this is not a matched-accuracy speedup.

Independent analysis/replay/predictive validation cost 467.301s wall and 390.547s CPU, separate from inference/training. Corrected half job 9.30965s wall/6.875s CPU. The historical synthetic bank generation cost267.159675s whole-command wall for524 sources /1048 observations; it includes shared validation/test/challenge data, so it is not wholly attributable to neural training. No new simulation bank was generated or billed in this campaign. The successful neural preparation timer also excludes its prior preservation-hash/binding setup.

## Existing test archive and next work

The original model selected epoch 22 and stopped 52; all96 test observations /48 sources were already evaluated. Their original parameter errors, intervals, widths and source-resampling uncertainty remain in `run_03/test_evaluation.json`; the frozen prediction arrays and all hashes are unchanged. No new development model was scored on that archive. Repeated access is not a fresh blind evaluation. The two original unconverged Bayesian pilots remain preserved; later full-y continuations, and the newly matched compressed inference, are distinguished above.

### Frozen original test metrics, read from the existing receipt

**96 observations /48 sources, original epoch 22 model only. No rescoring or new model/test comparison.**

| Target | Unit | Archived MAE | Archived RMSE | Archived95% coverage | Mean95% width |
|---|---|---:|---:|---:|---:|
| em0_pc_cm6 | pc cm^-6 | 38040.8 | 56108.2 | 98.96% | 266234 |
| te_k | K | 1698.3 | 1991.57 | 96.88% | 6458.57 |
| major_fwhm_arcsec | arcsec (full major EM FWHM) | 6.9776 | 9.58989 | 97.92% | 38.2497 |
| axis_ratio | 1 | 0.125388 | 0.143495 | 94.79% | 0.4613 |

The source-resampling intervals, PIT and50/80/95% widths/coverage remain in the unchanged archived JSON. Forty-eight source groups and prior scientific exposure limit inference; these numbers are historical evidence and cannot validate the development remedies as a fresh blind evaluation.

Remaining work is scientific: improve the posterior family/learning strategy under a new prespecified validation experiment; establish whether additional independent training sources help; validate predictive/coverage behavior on a broader prespecified source subset; and investigate temperature leverage with a jointly specified richer observation model. None is established by this single failed remedy. `PROTECTED_EVALUATION_PROPOSAL.md` specifies a separate256-source/512-observation final blind evaluation, seed/label custody, measured historical cost, runtime/disk assumptions and exact proposed bounded generator command/config. The unchanged generator uses an independently seeded randomized Sobol bank, not IID source draws; the entire proposed bank is reserved regardless of internal tags. It is unnecessary to reject today's failed remedy and can remain a later approval-gated step. No generation was performed.

## Reproducible local commands

Run from the project root with the existing environment. Never point reproduction commands at a preserved run. Bayesian commands retain the saved campaign science deadline and decline new work after it; any later continuation requires a separately recorded budget/protocol.

```powershell
$env:PYTHONPATH='src;ml'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OPENBLAS_NUM_THREADS='2'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
.venv/Scripts/python.exe -u -B ml/repair_cli.py run prepare --out output/ml_initial_20261023/repair_reproduction_01
.venv/Scripts/python.exe -u -B ml/repair_cli.py run neural --out output/ml_initial_20261023/repair_reproduction_01
# Measured-data prediction only, no label access; failed remedy remains explicitly labelled:
.venv/Scripts/python.exe -u -B ml/repair_cli.py predict --validation-index 0 --out output/ml_initial_20261023/predict_review_02
# The actual matched comparison and correction commands used in this campaign:
.venv/Scripts/python.exe -u -B ml/repair_campaign.py old32
.venv/Scripts/python.exe -u -B ml/repair_sampler_refresh.py
.venv/Scripts/python.exe -u -B ml/repair_refresh_resume.py
.venv/Scripts/python.exe -u -B ml/repair_half.py
.venv/Scripts/python.exe -u -B ml/repair_analysis.py
.venv/Scripts/python.exe -u -B ml/repair_report.py
```

Frozen protocols/configuration, source snapshots, package versions, seeds, checkpoint, curves, original-unit metrics, raw chains, complete diagnostics, timing and predictive draw receipts are saved alongside this report. No model-weight uncertainty is included in the NPE. No predetermined accuracy threshold or real-nebula readiness is claimed.
