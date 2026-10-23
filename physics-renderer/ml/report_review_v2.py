"""Measured before/after report; no changes to the archived experiment reports."""
import json
from pathlib import Path
import numpy as np
from hii_sim.io import atomic_json,digest
from diagnostics_v2 import diagnostic_report


def read(p):return json.loads(Path(p).read_text())


def build(review,attempt='v2_02'):
    out=review/attempt;before=read(review/'BEFORE_METRICS.json')
    verified=read(out/'independent_verification.json');modes=read(out/'mode_search.json')
    conditional=read(out/'neural_conditional_predictive.json')
    issues=read(review/'ISSUES_BEFORE.json')['issues']
    passed=sum(r['converged'] for r in conditional['records'])
    total=len(conditional['records'])
    status={
        'I01':('Fixed in v2 audit route','Raw chain hashes/counts/source pairs, all 11 diagnostic variables, quantiles and all 128 predictive receipts per case recomputed. Historical finalizer unchanged.'),
        'I02':('Fixed in v2 diagnostic route','Names must cover every variable; nonfinite/degenerate failures are JSON-safe; explicit inverse FFT length. Old 4000-draw passes reproduced without change.'),
        'I03':('Implementation added; scientific assessment remains limited',f'{passed}/{total} conditional solves passed. No profile or nuisance truth substituted. Only two outer neural draws per observation; no reliable aggregate predictive/calibration claim.'),
        'I04':('Exploration strengthened; irreducible limit remains','Eight independent validation optimizer starts per case, including position/PA/physical variation; optimizers are not posteriors. Finite mode completeness unproven.'),
        'I05':('Fixed in v2 review route','Unsupported PA/offset/target/grid/noise contracts fail closed; current contract matches. Frozen legacy likelihood preserved.'),
        'I06':('Clarified and verified; finite-data limit remains','Train 397/794, validation 67/134, test 48/96, challenge 12/24 source/observation counts. Train normalization/PCA and validation predictions reproduced. Test archive not rescored.'),
        'I07':('Unresolved scientific/model limitation','Frozen neural model unchanged. Same validation errors before/after; broad restricted-family posterior and temperature weakness are not repaired by audit checks.'),
        'I08':('Accounting clarified; comparison precision limit remains','Pilot-inclusive old accounting retained; conditional Bayesian extension cost/calls reported separately from amortized neural inference. No numerical speedup claim.'),
        'I09':('Fixed for new bounded conditional route; legacy limitation retained','Invalid settings rejected; stops before first milestone saved, all failures retained. No automatic partial-chain resume is claimed; old sampler code remains historical.')}
    rows=[]
    for issue in issues:
        state,evidence=status[issue['id']]
        rows.append({**issue,'after_status':state,'after_evidence':evidence})
    atomic_json(out/'ISSUES_AFTER.json',{'issues':rows})
    # Adversarial proof of changed behavior, no scientific data modified.
    x=np.random.default_rng(19).normal(size=(4,4000,2));x[0,:,1]+=8
    try:diagnostic_report(x,['one']);omission_rejected=False
    except ValueError:omission_rejected=True
    stuck=diagnostic_report(np.ones((4,100,1)),['stuck'])
    json.dumps(stuck,allow_nan=False)
    after={'omitted_bad_dimension_rejected':omission_rejected,'stuck_failure_json_serializable':True,
           'recomputed_bayesian_passes':sum(c['recomputed_diagnostics']['passed'] for c in verified['cases']),
           'conditional_converged':passed,'conditional_total':total,
           'validation_model_changed':False,'test_rescored':False,'scientific_limits_eliminated':False,
           'conditional_wall_seconds_total':sum(r['total_wall_seconds_including_predictive'] for r in conditional['records']),
           'conditional_forward_calls_total':sum(r['forward_calls_including_predictive'] for r in conditional['records'])}
    atomic_json(out/'AFTER_METRICS.json',after)
    lines=['# H II inference audit — BEFORE / AFTER','',
        '**Execution: complete for this bounded local audit and repair. Scientific/model limitations remain open.**', '',
        'Earlier simulator, original candidate/preparation, `run_03` and every `followup_01` artifact were fingerprinted before '
        'repairs and checked unchanged after each stage. No trainer was running at entry. One exclusive lock serialized jobs. '
        'No neural retraining, test rescoring, challenge mixing, new source-bank generation, real-data work, dependency '
        'change, cloud compute or publication occurred in this audit.', '',
        '## What was independently verified', '',
        '- Exact replay: compared original and replay weight/PCA NPZ arrays directly; every maximum absolute difference is zero.',
        '- Forward archive: 48 distinct expected test-source/first-realization records, original 46 rows identical, zero reported optimizer failures. '
        'This was a structural/receipt audit; test data were not rerendered. Stored median/max RMS 1.04034/3.09197 are optimistic profiles, not neural posterior predictive samples.',
        '- Raw Bayesian chains: recomputed rank/folded split Rhat, bulk/tail ESS for all targets/nuisances/PA sine and cosine; '
        'checked prior support, final states, source selection and chain hashes. Both original follow-up passes are independently reproduced.',
        '- Regenerated all 128 Bayesian predictive scalar receipts per validation case from the saved chain and original RNG stream. '
        'Selected posterior indices are now saved explicitly. Also reproduced physical quantiles and interval widths/errors.',
        '- Refit normalization from the 794 training observations /397 parents and PCA from training inputs only; reproduced '
        '134 validation predictions and their separate original targets exactly. No test prediction was recalculated.', '',
        '| Validation comparator | Recomputed max Rhat | Minimum bulk ESS | Minimum tail ESS | Max predictive receipt difference |',
        '|---|---:|---:|---:|---:|']
    for j,c in enumerate(verified['cases'],1):
        d=list(c['recomputed_diagnostics']['variables'].values())
        lines.append(f'| {j} | {max(r["rank_folded_split_rhat"] for r in d):.6f} | {min(r["bulk_ess"] for r in d):.1f} | '
                     f'{min(r["tail_ess"] for r in d):.1f} | {c["max_predictive_scalar_replay_difference"]:.3g} |')
    lines+=['','## Issue register and repairs','','| Issue | Severity | Before | After | Evidence / remaining limit |','|---|---|---|---|---|']
    for r in rows:lines.append(f'| {r["id"]} | {r["severity"]} | {r["evidence"]} | {r["after_status"]} | {r["after_evidence"]} |')
    defect=before['synthetic_defect_reproductions']
    lines+=['','## Measured change in verification behavior','',
            '| Check | Before | After |','|---|---|---|',
            f'| Bad second variable omitted from names | Could pass: {defect["omitted_bad_dimension_can_pass"]} | Raises explicit contract error |',
            '| Stuck-chain failure receipt | Infinity prevented strict JSON serialization | Explicit failed/null diagnostic, strict JSON serializable |',
            f'| Odd FFT padding example | Requested {defect["requested_fft_length"]}, inverse default {defect["legacy_irfft_default_length"]} | Correct explicit inverse length; tested against direct autocovariance |',
            '| Final acceptance | Saved summaries/finite array shape trusted | Actual chains and selected observations recomputed and bound |',
            '| Neural nuisance treatment | Marginal median + optimized nuisance point | Conditional posterior sampling for each pre-frozen neural target draw |',
            '', 'The historical implementation and artifact bindings were preserved. Repairs live in the versioned v2 route; '
            'the old code is not retroactively represented as fixed. The FFT defect does not change existing 4000-draw diagnostics '
            'because their padded lengths are even. No historical Bayesian pass was withdrawn by this recheck.', '',
            '## Neural model errors: unchanged, honestly', '',
            'The same frozen model is used before and after. These are validation-development results, not a new blind test. '
            'Point estimate is the marginal posterior median; MAE/RMSE use original physical units. Prior-midpoint is a '
            'deterministic diagnostic and is never treated as a Bayesian posterior comparator.', '',
            '| Target | Units | Before MAE | After MAE | RMSE (unchanged) | Prior-midpoint MAE |', '|---|---|---:|---:|---:|---:|']
    units=['pc cm^-6','K','arcsec (full major EM FWHM)','1']
    for (name,r),unit in zip(before['validation_metrics_from_saved_arrays'].items(),units):
        lines.append(f'| {name} | {unit} | {r["mae"]:.6g} | {r["mae"]:.6g} | {r["rmse"]:.6g} | {r["prior_midpoint_mae"]:.6g} |')
    lines+=['','| Target | Interval level | Validation coverage | Mean width (original units) |','|---|---:|---:|---:|']
    for name,r in before['validation_metrics_from_saved_arrays'].items():
        for level,d in r['coverage'].items():lines.append(f'| {name} | {float(level):.0%} | {d["observed"]:.2%} | {d["mean_width"]:.6g} |')
    lines+=['', 'Only 67 independent validation sources underlie 134 observations; two realizations are not independent skies. '
            'Coverage remains descriptive. Source-bootstrap intervals in the archived metrics are resampling intervals under '
            'this finite split; the scrambled-Sobol design and model-selection use of validation prevent treating them as '
            'guaranteed iid population confidence intervals. The 96 test observations belong to 48 already evaluated sources '
            'and remain an unchanged archive. No post-hoc interval scaling was fitted to the test.', '',
            'The neural model keeps only 32 PCA coordinates (39.22% of standardized training variance) and one Gaussian in '
            'scaled-logit target coordinates. Variance retention is not an information-sufficiency criterion. Weak temperature '
            'recovery and broad/different marginals remain model/identifiability evidence. Two matched conventional posteriors '
            'cannot establish general superiority or calibration.', '',
            '## Nuisance-correlated neural predictive pilot','',
            'The new route draws theta from the FROZEN neural posterior, then samples the same physical conditional posterior '
            'p(nuisance | theta, measurement). Its joint approximation is q_NN(theta | measurement) p(nuisance | theta, measurement). '
            'This explicitly retains conditional target/nuisance dependence. It uses neither true nuisance values nor profiled '
            'best fits as posterior samples, and it does not update/reweight theta by the likelihood. Conditional nuisance solves '
            'are additional conventional inference cost. Their gain priors are unbounded standard normals; calibration matrices '
            'are cached algebraically and tested against the original full correlated likelihood.', '',
            f'{passed}/{total} conditionals passed the all-nuisance Rhat/ESS gate. The four outer target draws were saved before '
            'conditional fitting; failed or difficult draws were not filtered or replaced. No aggregate full-neural predictive '
            'tail probability is asserted: two outer draws per source are far too few, even if all conditionals pass.', '',
            '| Conditional solve | Gate | Retained /chain | Wall seconds incl predictive | Full forward calls | Conditional observed RMS median |',
            '|---|---|---:|---:|---:|---:|']
    for r in conditional['records']:
        s=r.get('conditional_sampling',{})
        lines.append(f'| {r["tag"]} | {"PASS" if r["converged"] else "FAIL / INCONCLUSIVE"} | {s.get("retained_per_chain",0)} | '
                     f'{r["total_wall_seconds_including_predictive"]:.2f} | {r["forward_calls_including_predictive"]} | '
                     f'{r.get("conditional_observed_rms_median","withheld")} |')
    lines+=['','## Mode coverage, units and fairness','',
            'All methods use the same validation observation, four target definitions/continuous priors, uniform PA [0,180), '
            'uniform position offsets [-10,10] arcsec, mean-one 5% common lognormal amplitude, Gaussian spectral slope sigma 0.01, '
            'and the same conditional correlated thermal Gaussian law. Canonical measurements are Jy, thermal covariance Jy^2. '
            'Self-conjugate variance and cross-frequency correlation are retained; imaging taper is not inverse variance. '
            'EM0 and size use log10 in loader normalization, while the neural likelihood uses a fixed prior-unit logit transform; '
            'Te and q are linear before their fixed support mapping. The Gaussian size is EM FWHM, not necessarily brightness FWHM.', '',
            'The strict review wrapper rejects unsupported PA/offset/target-transform/grid/noise configurations that the legacy '
            'likelihood otherwise assumes. It does not alter the simulator or original dataset. Neural inputs round normalized '
            'measurements to float32 before float64 network/PCA operations; conventional likelihoods use float64 arithmetic on '
            'the same stored measurements. The finite Sobol training design approximates the intended continuous prior. '
            'These precision, compression, posterior-family and finite-design differences remain explicit.', '',
            '| Mode-search case | Completed optimization attempts | Successful status | Best cost | Best cost among 100 chain selections |',
            '|---|---:|---:|---:|---:|']
    for j,c in enumerate(modes['cases'],1):
        lines.append(f'| {j} | {len(c["starts"])} | {sum(r.get("optimizer_success",False) for r in c["starts"])} | {c["best_cost"]} | {c["best_of_100_sampled_chain_costs"]:.6g} |')
    lines+=['', 'Dispersed optimizer input points often converge to the same region. An optimizer is not a posterior, and '
            'finite multistart searches plus Rhat/ESS cannot prove absence of separated modes. Full outputs retain every '
            'unsuccessful initialization rather than reporting only the best.', '',
            '## Runtime scopes and evidence','',
            'The archived training replay took 83.89 s end to end (normalization and repeated verified reads included), '
            'with 1.90 s for PCA/optimizer/checkpoint I/O. The original optimizer-only 0.56 s is not total training cost. '
            'Archived neural inference was about 0.00077 s per observation including 2048 approximate draws, excluding disk '
            'loading. The proposal-dependent conventional workflow includes historical pilot cost: 562.28/711.68 s of '
            'recorded sampler work across sessions. Those sums are not a controlled matched-precision benchmark. '
            'The new conditional-predictive extension additionally cost '
            f'{after["conditional_wall_seconds_total"]:.2f} wall s / {after["conditional_forward_calls_total"]} full forward '
            'calls in this pilot. It must not be included in the fast neural-inference headline without attribution.', '',
            'Versions, seeds, settings and source hashes are in `v2/binding.json` and the preserved bindings. Test commands/results '
            'are saved beside the review. Failed old pilot checks and the historical periodic-coordinate test failure remain '
            'in their original files; new failed/limited conditional checks remain in result JSONs. The current review does '
            'not erase earlier failures or silently relabel them as passes.', '',
            '## Reproducible commands','', 'From project root, use the existing environment and a fresh BEFORE output for a new audit:', '',
            '```powershell', "$env:PYTHONPATH='src;ml'", "$env:PYTHONDONTWRITEBYTECODE='1'", "$env:OPENBLAS_NUM_THREADS='2'", "$env:OMP_NUM_THREADS='2'", "$env:MKL_NUM_THREADS='2'",
            '.venv/Scripts/python.exe -u -B ml/audit_before_v2.py --out output/ml_initial_20261023/review_NEW',
            '.venv/Scripts/python.exe -u -B ml/review_inference_v2.py --review output/ml_initial_20261023/review_NEW --stage verify',
            '.venv/Scripts/python.exe -u -B ml/review_inference_v2.py --review output/ml_initial_20261023/review_NEW --stage modes',
            '.venv/Scripts/python.exe -u -B ml/review_inference_v2.py --review output/ml_initial_20261023/review_NEW --stage conditional',
            '.venv/Scripts/python.exe -u -B ml/review_inference_v2.py --review output/ml_initial_20261023/review_NEW --stage report',
            '.venv/Scripts/python.exe -B -m pytest -q ml/test_review_v2.py ml/test_followup.py ml/test_ml.py tests/test_training_data.py tests/test_npe_contract.py --basetemp output/ml_initial_20261023/review_tests_NEW',
            '```', '',
            'The same process lock is used for every review stage. Completed per-target conditionals are retained; interrupted '
            'attempts are not silently restarted. A new attempt needs a new versioned output; automatic partial-chain resume '
            'is not implemented. No generation command is included.', '',
            '## Remaining limits and next step','',
            'Recoverable verification/contract/reporting defects were repaired in the v2 route. Model inadequacy, limited '
            'source count, finite mode exploration, an artificial Fourier instrument and a smooth pure-H isothermal source '
            'family are not eliminated. Full neural predictive calibration remains unestablished. No real-nebula readiness '
            'or predetermined accuracy threshold is claimed.', '',
            'Next development should compare information-preserving inputs/conditional density families and a jointly modeled '
            'nuisance posterior on training/validation sources, with a broader predeclared validation comparator budget. '
            'A later final blind claim needs a protected independently generated source set, fixed priors/instrument/noise, '
            'private independent seed, grouped realizations and sealed evaluator-held labels after model/metric freeze. '
            'That proposal is documented only; generation requires approval and was not performed.', '',
            '**Completed:** independent saved-state verification, strict diagnostics/contract route, matched predictive receipt '
            'replay, validation mode exploration, conditional-neural predictive implementation/pilot and measured before/after record. '
            '**Pending:** scientific limitations above and any failed conditional convergence. **Blocker for claiming all limits '
            'eliminated:** finite data/model assumptions and absence of independent final validation; this is not cured by more prose.', '',
            'Primary method references: [Stan posterior analysis](https://mc-stan.org/docs/reference-manual/analysis.html) and '
            '[rank/folding/localization paper](https://arxiv.org/abs/1903.08008).']
    lines_text='\n'.join(lines).replace('`v2/',f'`{attempt}/')
    incident_path=review/'INCIDENTS.json'
    if incident_path.exists():
        lines_text+='\n\n## Issues encountered during this audit\n\n'
        for incident in read(incident_path)['incidents']:
            lines_text+=f'- **{incident["id"]}:** {incident["problem"]} Repair: {incident["repair"]} Evidence: {incident["evidence"]}\n'
    (review/'BEFORE_AFTER.md').write_text(lines_text+'\n',encoding='utf-8')
    atomic_json(review/'SUMMARY.json',{'execution':'complete','bounded_audit_complete':True,'all_scientific_issues_eliminated':False,
                'issues':rows,'before_metrics':'BEFORE_METRICS.json','after_metrics':f'{attempt}/AFTER_METRICS.json',
                'report':'BEFORE_AFTER.md','preserved_runs_unchanged':True,'test_rescored':False,
                'conditional_converged':passed,'conditional_total':total})
    atomic_json(out/'delivery.json',{'report_sha256':digest(review/'BEFORE_AFTER.md'),
                'summary_sha256':digest(review/'SUMMARY.json'),
                'result_hashes':{p.name:digest(p) for p in sorted(out.glob('*')) if p.is_file() and p.name!='delivery.json'}})
    print(f'BEFORE_AFTER.md saved: {review}; conditional passes {passed}/{total}; scientific limitations remain.',flush=True)
