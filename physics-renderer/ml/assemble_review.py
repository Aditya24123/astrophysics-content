"""Assemble the FINAL measured audit report, preserving all earlier phases."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import numpy as np
from hii_sim.io import atomic_json,digest,versions
from review_inference_v2 import verify_preserved,read,arrays,inputs
from review_contract import full_diagnostics
from conditional_predictive_v2 import conditional_diagnostics
from probit_probe import ProbitSupport
from model import Support,project,sample
from hii_sim.training_data import load_transform


def execute(review):
    before_binding=verify_preserved(review)
    before=read(review/'BEFORE_METRICS.json');parent=review/'v2_02';ext=review/'extensions_01';angle=review/'angle_refresh_01'
    out=review/'final_01';out.mkdir(exist_ok=False)
    config=read('configs/compact_proxy_gl4_v3.json')
    independent=read(parent/'independent_verification.json')
    original=Path('output/ml_initial_20261023/run_03');follow=Path('output/ml_initial_20261023/followup_01')
    for n in (1,2):
        if not full_diagnostics(arrays(follow/f'bayes_{n:02d}_chains.npz')['coordinates'],config)['passed']:
            raise ValueError('Raw full Bayesian comparator acceptance failed')
    initial=read(parent/'neural_conditional_predictive.json')['records'];extended=read(ext/'results.json')
    expected_tags=['conditional_01_01','conditional_01_02','conditional_02_01','conditional_02_02']
    if [r['tag'] for r in initial]!=expected_tags or [r['tag'] for r in extended]!=expected_tags:
        raise ValueError('Missing, duplicate, reordered or extra outer conditional cases')
    plan=read('ml/review_plan_v2.json');_,_,base,root,_,selection=inputs(plan)
    transform=load_transform(base['prepared'],root/'public')
    pca,weights=arrays(original/'full/pca.npz'),arrays(original/'full/best.npz');original_support=Support(config)
    for number,inp in enumerate(selection,1):
        rng=np.random.default_rng(plan['seed']+number*10000)
        regenerated=original_support.decode(sample(project(transform.features(inp['features']['measurements'].values_jy)[None],pca),weights,2,rng))[0]
        np.testing.assert_array_equal(regenerated,np.array([initial[(number-1)*2+j]['neural_target_draw'] for j in range(2)]))
    repaired={r['tag']:r for r in read(angle/'completion.json')['new_phase_results']}
    conditional=[]
    for index,(record,continued) in enumerate(zip(initial,extended)):
        tag=record['tag']
        if continued['converged']:
            location=parent if continued['used_parent_pass'] else ext
            phase='original conditional pilot' if continued['used_parent_pass'] else 'same-kernel exact continuation'
        else:
            location=angle;phase='new PA-refresh homogeneous phase'
        raw=arrays(location/f'{tag}_chains.npz');chain=raw['coordinates'];diag=conditional_diagnostics(chain)
        if not diag['passed']:raise ValueError('Latest accepted conditional fails raw-chain recheck: '+tag)
        branch=read(location/f'{tag}_sampling.json')['branch_center']
        if (np.any((chain[:,:,1:3]<0)|(chain[:,:,1:3]>1)) or np.any(abs(chain[:,:,0]-branch)>.5)):
            raise ValueError('Conditional nuisance prior support violated')
        np.testing.assert_array_equal(raw['states'],chain[:,-1])
        np.testing.assert_array_equal(raw['fixed_prior_units'],arrays(parent/f'{tag}_chains.npz')['fixed_prior_units'])
        if location==ext:
            old=arrays(parent/f'{tag}_chains.npz')['coordinates']
            np.testing.assert_array_equal(chain[:,:old.shape[1]],old)
        conditional.append({'tag':tag,'phase':phase,'location':str(location),'retained_per_chain':chain.shape[1],
                            'max_rhat':max(r['rank_folded_split_rhat'] for r in diag['variables'].values()),
                            'min_bulk_ess':min(r['bulk_ess'] for r in diag['variables'].values()),
                            'min_tail_ess':min(r['tail_ess'] for r in diag['variables'].values()),
                            'chain_sha256':digest(location/f'{tag}_chains.npz'),'all_variables':diag})
    # Confirm exact-proposal RNG continuation preserved old raw draws for BOTH
    # continued cases, including the case that remained failed at the time limit.
    for row in extended:
        if row['used_parent_pass']:continue
        path=ext/f'{row["tag"]}_chains.npz'
        if path.exists():
            a,b=arrays(parent/f'{row["tag"]}_chains.npz'),arrays(path)
            np.testing.assert_array_equal(b['coordinates'][:,:a['coordinates'].shape[1]],a['coordinates'])
    probe=review/'probit_probe_01';comparison=read(probe/'comparison.json');diagnosis=read(review/'model_diagnosis_01/diagnosis.json')
    val=arrays(follow/'validation/predictions.npz');candidate=arrays(probe/'validation/predictions.npz')
    np.testing.assert_array_equal(val['truth'],candidate['truth']);np.testing.assert_array_equal(val['parent'],candidate['parent'])
    support=ProbitSupport(config);np.testing.assert_allclose(support.decode(candidate['transformed_mean']),candidate['median'],rtol=0,atol=0)
    for j,name in enumerate(config['reported_targets']):
        np.testing.assert_allclose(np.mean(abs(candidate['median'][:,j]-candidate['truth'][:,j])),comparison['parameters'][name]['probe_mae'],rtol=0,atol=0)
    frozen=read(probe/'FROZEN_PROTOCOL.json')
    if digest('ml/probit_probe_plan.json')!=frozen['protocol_sha256']:raise ValueError('Probe protocol changed')
    for path,sha in frozen['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('Probe source changed: '+path)
    training_started=next(json.loads(s)['utc'] for s in (review/'probit_probe_console.log').read_text(encoding='utf-8').splitlines()
                          if s.startswith('{') and json.loads(s).get('event')=='training_started')
    if datetime.fromisoformat(frozen['frozen_utc'])>=datetime.fromisoformat(training_started):raise ValueError('Protocol not frozen before training')
    issues=read(parent/'ISSUES_AFTER.json')['issues'];incidents=read(review/'INCIDENTS.json')['incidents']
    details=read(review/'predictive_details_01/residual_checks.json')
    bootstrap=diagnosis['parameters']
    lines=['# H II inference: final BEFORE / AFTER audit','',
           '**Execution: COMPLETE for this bounded local audit and repair. All scientific limitations have NOT been eliminated.**', '',
           '**Prominent unresolved result:** temperature and axis-ratio validation MAEs remain worse than their matched '
           'prior-median diagnostics, both for the frozen model and after one predeclared target-transform probe. '
           'The original disadvantages have source-paired intervals that include zero, so they do not alone prove a bad '
           'posterior. The probe gives mixed development improvements, not a validated general accuracy gain or a new canonical model.', '',
           'The simulator, all original candidate/preparation files, `run_03`, and EVERY `followup_01` artifact are unchanged '
           'by SHA256. The original checkpoint remains canonical. All four original neural predictive outer draws, every '
           'failed conditional attempt, both continuations and the new angle-mixing phase remain saved. No existing test '
           'prediction was recomputed or used for selection. No real-data work, publication, dependency change or new '
           'scientific/protected evaluation bank was performed. Regression tests generated only their documented tiny '
           'throwaway fixtures, which were not training/evaluation data.', '',
           '## Actual model errors and uncertainty','',
           'MAE uses the marginal posterior MEDIAN, not the posterior mean or a fitted physical optimum. The matched prior '
           'median is the geometric midpoint for log-uniform EM0/size and arithmetic midpoint for linear Te/q. Units below '
           'are original physical units. The comparison uses 134 validation observations from **67 source groups**; '
           'source-paired resampling keeps each source’s two noise realizations together.', '',
           '| Target | Units | Frozen model MAE | One probit probe MAE | Prior median MAE | Probe minus frozen: resampling 95% interval |',
           '|---|---|---:|---:|---:|---|']
    for name,r in comparison['parameters'].items():
        lo,hi=r['mae_change_vs_original_source_resampling95']
        lines.append(f'| {name} | {r["unit"]} | {r["original_mae"]:.6g} | {r["probe_mae"]:.6g} | {r["prior_median_mae"]:.6g} | {r["mae_change_vs_original"]:.6g} [{lo:.6g}, {hi:.6g}] |')
    lines+=['',f'Temperature’s original MAE excess is {bootstrap["te_k"]["mae_difference_neural_minus_prior"]:.3f} K '
            f'with source-resampling interval {bootstrap["te_k"]["paired_source_resampling_95_interval"]}; q’s excess is '
            f'{bootstrap["axis_ratio"]["mae_difference_neural_minus_prior"]:.6f} with interval '
            f'{bootstrap["axis_ratio"]["paired_source_resampling_95_interval"]}. These are descriptive intervals on reused '
            'validation/Sobol sources, not selection-adjusted iid population confidence guarantees.', '',
            'The probe was frozen before training, made ONE fit, reused only training-fitted normalization/PCA and changed '
            'only scaled-logit to probit(prior-unit) target coordinates. It selected epoch 27 and stopped at 57. It makes '
            'the exact uninformative prior representable, but cannot restore PCA information or express arbitrary '
            'multimodal/curved posteriors. It improved EM0 and q MAEs in this development sample, barely changed Te, and '
            'slightly worsened size MAE. The original model was NOT replaced and the probe was NOT scored on the 96 test observations.', '',
            '## Marginal and joint density scores: matched support/Jacobians','',
            'Scores are NLL relative to the exact prior in common prior-unit coordinates u, so the uniform prior score is 0. '
            'Negative is better. Both transforms’ exact label Jacobians are included. Raw Gaussian losses in logit and '
            'probit coordinates are not directly compared. A proper density objective can disagree with realized sample '
            'MAE; an exact posterior median nevertheless minimizes expected absolute error.', '',
            '| Target | Frozen marginal prior-relative NLL [nats] | Probit probe marginal prior-relative NLL [nats] |',
            '|---|---:|---:|']
    for name,r in comparison['parameters'].items():lines.append(f'| {name} | {r["original_marginal_prior_unit_nll"]:.6g} | {r["probe_marginal_prior_unit_nll"]:.6g} |')
    lines+=['',f'Joint prior-relative NLL: frozen **{comparison["original_joint_prior_unit_nll"]:.6f}**, probe '
            f'**{comparison["probe_joint_prior_unit_nll"]:.6f}**; change {comparison["joint_density_change_probe_minus_original"]:.6f} '
            f'nats with descriptive source-resampling interval {comparison["joint_density_change_source_resampling95"]}. '
            'The joint gain must not mask weak Te/q marginals, and validation reuse prevents calling this a fresh blind gain.', '',
            '## What the model-focused diagnosis supports','',
            '- **Prior dominance / identifiability:** the two full-data Bayesian temperature marginals have histogram KL '
            'from prior about 0.005–0.007 nats and 95% widths about 94–95% of the full prior-unit range. Te is weakly '
            'identified in THESE observations; two cases do not prove a universal irreducible limit.',
            '- **Axis ratio is different:** the two full-data Bayesian q marginals have KL about 0.89/1.31 nats, while '
            'the frozen neural q marginals are nearly prior-like. Thus universal physical non-identifiability is not a '
            'sufficient explanation for the neural q result.',
            '- **Compression:** exact-covariance local Fisher calculations at three saved posterior states per case '
            'retain median nuisance-projected q information fractions 0.195/0.578 after PCA32. This measures local '
            'parameter information, not merely 39.22% training-variance retention. It is not a global posterior sufficiency proof.',
            '- **Learning / family / data:** only 397 training sources, one original seed, a restricted Gaussian conditional '
            'family and reused validation constrain conclusions. One half-data comparison does not establish data sufficiency. '
            'That historical half-data fit also reused feature normalization fitted on all 794 training observations '
            '(while its PCA/weights used the half subset), so it is not a clean end-to-end sample-size learning curve. '
            'This is not validation/test leakage, but its preprocessing benefit must not be ignored. '
            'The probit probe shows prior representability alone is insufficient. The separate causal contributions of '
            'finite training size, optimizer/generalization and density-family restriction are not identified by this audit.',
            '- **Implementation:** no point-median, units, target-pairing, gradient, precision-factor sampling or '
            'train/validation normalization/PCA defect was found in the frozen predictor. Numerical verification defects '
            'were repaired separately; they are not advertised as predictive-accuracy improvements.', '',
            '## Coverage and sharpness: development evidence only','',
            '| Target | Nominal level | Frozen coverage | Probe coverage | Frozen mean width | Probe mean width |',
            '|---|---:|---:|---:|---:|---:|']
    for name,r in comparison['parameters'].items():
        old=before['validation_metrics_from_saved_arrays'][name]['coverage']
        for level,new in r['coverage_and_width'].items():
            lines.append(f'| {name} | {float(level):.0%} | {old[level]["observed"]:.2%} | {new["observed"]:.2%} | {old[level]["mean_width"]:.6g} | {new["mean_interval_width_original_units"]:.6g} |')
    lines+=['', 'Widths use the physical units in the error table. There are 67 validation sources, not 134 independent '
            'skies; 2048 neural draws/observation add Monte Carlo uncertainty. Existing source-resampling intervals are '
            'saved in each metrics JSON. Coverage is not broadly established, especially after stopping/model development '
            'on validation. The evaluated 96-observation/48-source test archive is unchanged; 24 mixed-layout challenges '
            'from 12 sources remain separate. No new blind evaluation was performed.', '',
            '## Raw verification and numerical fixes','',
            '| Item | Before | After |', '|---|---|---|',
            '| Exact replay claim | Saved assertion | Direct NPZ comparisons: every original/replay weight and PCA array identical |',
            '| Forward 48/48 claim | Saved profile records | All 48 expected source/first-realization IDs verified; original 46 identical; no new test rerender |',
            '| 2/2 Bayesian passes | Saved diagnostic summaries | Recomputed from raw chains, all targets/nuisances/PA sin/cos pass; quantiles and 128 predictive receipts per case replay exactly |',
            '| Omitted/invalid diagnostic variable | Could omit a bad dimension; stuck report not strict-JSON serializable | Every variable required; nonfinite/degenerate failures explicit and JSON-safe |',
            '| Odd FFT inverse length | Example requested 63, inverted 62 | Explicit inverse length, direct autocovariance regression |',
            '| Assumed nuisance/config priors | Hardcoded PA/offset and target transform silently assumed | New route rejects unsupported configs/units/layouts instead of silently changing priors |',
            '| Predictive nuisance treatment | Median target + optimized nuisance point | Coherent q_NN(theta|C(y)) p(nuisance|theta,y) pilot, validated conditional chains, original failures retained |', '',
            'The original 4000-draw Bayesian passes survive unchanged: their FFT padding was even. Historical code is '
            'preserved for its bindings; new checks live in the strict versioned route. Extra mode searches used eight '
            'starts per validation case; all 16 optimization statuses succeeded and returned to known regions. These '
            'optimizers are initialization/mode checks, NEVER posterior substitutes. Finite Rhat/ESS and searches cannot '
            'prove global mode completeness.', '',
            '## Conditional predictive pilot: failures and actual final acceptance','',
            'All four neural target draws were frozen BEFORE conditional fitting and none was filtered, redrawn or '
            'reweighted. The first bounded pilot passed 2/4 cases. Exact same-kernel continuation preserved all old draws '
            'and RNG paths, passed one more case, and left one failure at the 300 s budget. A separately predeclared '
            'full-period PA proposal then passed that last case in a new homogeneous phase. Its priors, likelihood and '
            'target draw were unchanged; old failed chains are not counted as passing in the new-phase diagnostics.', '',
            '| Original outer target | Accepted phase | Retained /chain | Max rank/folded Rhat | Min bulk ESS | Min tail ESS |',
            '|---|---|---:|---:|---:|---:|']
    for c in conditional:lines.append(f'| {c["tag"]} | {c["phase"]} | {c["retained_per_chain"]} | {c["max_rhat"]:.6f} | {c["min_bulk_ess"]:.1f} | {c["min_tail_ess"]:.1f} |')
    lines+=['', 'The gate remains Rhat <1.01 and bulk/tail ESS >=400 for every nuisance and PA sine/cosine. Margins are '
            'finite and sometimes narrow; a pass does not certify tail quantiles or global modes perfectly. **4/4 conditional '
            'passes do NOT mean a validated full neural predictive marginal:** there are only two theta draws per source '
            'and two conventional comparison observations. These are conditional/hybrid implementation checks.', '',
            'Band/radius residual checks retain all four accepted theta cases and show some fixed-theta draws have larger '
            'structured residuals than the joint Bayesian samples. This is evidence about those draws and information '
            'mismatch, not an estimate of how often the neural posterior fails across sources.', '',
            '![Validation residual checks](../predictive_details_01/band_residuals.png)', '',
            '## Fairness and remaining information mismatch','',
            'The raw observations, four physical definitions/supports/continuous priors, PA/position priors, calibration '
            'priors and full forward/thermal law match. Common amplitude is mean-one lognormal, slope is Gaussian; '
            'thermal covariance retains cross-frequency correlation and self-mode variance. Imaging taper is not inverse '
            'variance. True parameters remain separate supervised labels; seeds/IDs/true sky/realized nuisances are '
            'excluded from predictor tensors. All preprocessing is fitted on training sources/observations only.', '',
            'However, the NPE sees PCA-compressed C(y), whereas the conventional reference uses full y. The new hybrid '
            'conditional extension uses p(nuisance|theta,full y) but retains q_NN(theta|C(y)). It is a normalized approximate '
            'joint, not the exact full-data posterior or a fully amortized joint neural model. A raw-data discrepancy may '
            'reflect discarded information even if a compressed-data estimator were ideal. A validated conventional '
            'posterior on the matched compressed likelihood (with C Sigma C^T) is still needed to fully separate compression '
            'from density-estimation error. Local Fisher diagnostics do not replace that posterior comparison. Float32 '
            'neural input rounding vs float64 conventional arithmetic and the finite Sobol training design also remain explicit.', '',
            '## Runtime, tests and encountered issues','',
            'No duplicate trainer was launched; jobs were serialized with the same lock. Original training replay cost '
            '83.89 s included normalization and verified reads; its 0.56 s optimizer-only record is not end-to-end cost. '
            f'The single probit probe used {comparison["wall_seconds_with_loading_and_evaluation"]:.2f} s including loading/evaluation, '
            f'with {comparison["training_info"]["elapsed_seconds"]:.2f} s in the optimizer loop; its preprocessing scope differs '
            'from the old full replay. Archived neural target inference (~0.00077 s including 2048 approximate draws) excludes '
            'all conditional solves. Pilot-inclusive conventional recorded work remains 562.28/711.68 s per source across '
            'different sessions. New nuisance integration/continuation/PA-phase costs are saved separately and must not '
            'be hidden in a fast-neural headline. There is no matched-accuracy speedup claim.', '',
            'Testing: 69 unique checks passed across the combined run and targeted runs. The combined 63-test run had '
            '**62 passes and one Windows temporary-fixture replacement failure**; a sandbox TEMP retry had a setup error; '
            'the identical lineage test then passed unchanged in a fresh shorter workspace path. This is NOT described '
            'as a simulator software defect fixed. Original source/test logic was preserved. Two probit/Jacobian and two '
            'PA-proposal tests passed separately, as did two correlated-covariance Fisher-information regressions. '
            'Prior known-distribution/RNG segmentation tests passed in the combined run.', '']
    for incident in incidents:lines.append(f'- **{incident["id"]}:** {incident["problem"]} Handling: {incident["repair"]} Logs: {incident["evidence"]}')
    lines+=['', 'Scientific failed attempts are also preserved: the two historical Bayesian pilots, two initial conditional '
            'failures and one bounded exact-continuation failure. Failed intermediate milestones remain in console logs. '
            'Their later accepted phases do not erase those failures.', '',
            '## Reproducible commands and saved state','',
            'All commands run from project root with the existing pinned environment. Use new versioned outputs for a new '
            'attempt; never overwrite these artifacts. `ml/README.md` documents the BEFORE/verify/mode/conditional stages. '
            'The subsequent commands used here were:', '', '```powershell',
            "$env:PYTHONPATH='src;ml'", "$env:PYTHONDONTWRITEBYTECODE='1'", "$env:OPENBLAS_NUM_THREADS='2'", "$env:OMP_NUM_THREADS='2'", "$env:MKL_NUM_THREADS='2'",
            '.venv/Scripts/python.exe -u -B ml/resume_conditionals.py',
            '.venv/Scripts/python.exe -u -B ml/diagnose_model.py',
            '.venv/Scripts/python.exe -u -B ml/probit_probe.py',
            '.venv/Scripts/python.exe -u -B ml/angle_refresh.py',
            '.venv/Scripts/python.exe -u -B ml/predictive_details_final.py',
            '.venv/Scripts/python.exe -u -B ml/assemble_review.py',
            '```', '',
            'Frozen protocols are `model_diagnosis_plan.json`, `probit_probe_plan.json`, `angle_refresh_plan.json` and '
            '`review_plan_v2.json` in `ml/`. Software versions, source hashes, seeds, saved RNG states, original/probe '
            'checkpoints, raw epoch curves, coverage/sharpness, conditional chains and test logs are retained locally. '
            'The probe’s FROZEN_PROTOCOL timestamp precedes its training-start log. Its checkpoint is '
            '`../probit_probe_01/full/best.npz`; the unchanged canonical checkpoint remains under `run_03/full/best.npz`.', '',
            '## Exact remaining limits and next steps','',
            '1. **Te/q performance remains unresolved.** Te is prior-dominated in the two inspected full-data cases, not '
            'proven globally unidentifiable; q has full-data information that the NPE does not fully exploit. Point MAE '
            'disadvantages alone are not a posterior-validity test.',
            '2. **Cause partition remains incomplete.** Compression, family/optimization and finite training data contribute '
            'plausibly; a matched compressed-likelihood Bayesian comparison, more predeclared validation cases and '
            'training-seed/learning-curve studies are needed to separate them.',
            '3. **No broad calibration claim.** Four outer theta draws, two conventional cases, reused 67-source validation, '
            'finite MCMC tails and single-seed fits do not establish calibrated full neural uncertainty. The NPE is not '
            'a Bayesian distribution over network weights; model/training uncertainty is not separately characterized.',
            '4. **No real-object transfer.** The smooth pure-H isothermal Gaussian family, synthetic Fourier operator and '
            'known synthetic uncertainty laws do not certify real telescope/morphology/noise modeling.',
            '5. **Future protected evaluation needs approval.** After a future model/metric freeze, propose a separate '
            'independent source set with private seed, unchanged intended physical priors/noise/operator and sealed '
            'evaluator-held labels. Prefer independently sampled prior sources if formal iid coverage statements are '
            'required; group all realizations. No such scientific bank was generated or approved here.', '',
            '**Completed:** independent archive verification, scoped diagnostic/contract repairs, honest failure accounting, '
            'nuisance-correlated predictive implementation and all four accepted conditional cases, source-aware model '
            'diagnosis, one frozen validation-only sensitivity probe, and this measured before/after record. '
            '**Pending:** the scientific limits above. **Blocker to claiming every issue eliminated:** the available '
            'information/model family and lack of independent final validation; more prose or regression tests cannot '
            'remove those limits.', '',
            'Primary diagnostic references: [Stan posterior analysis](https://mc-stan.org/docs/reference-manual/analysis.html) '
            'and [rank/folding/localization](https://arxiv.org/abs/1903.08008).']
    (out/'BEFORE_AFTER.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summary={'execution':'complete','bounded_audit_and_repairs_complete':True,'all_issues_eliminated':False,
             'canonical_model_unchanged':True,'canonical_model_sha256':digest(original/'full/best.npz'),
             'probe_model_promoted':False,'probe_model_sha256':digest(probe/'full/best.npz'),
             'test_rescored':False,'original_runs_dataset_simulator_unchanged':True,
             'validated_full_bayesian_cases':2,'accepted_neural_conditionals':4,'outer_neural_draws_per_observation':2,
             'validation_observations':134,'validation_sources':67,'model_comparison':comparison['parameters'],
             'conditional_diagnostics':conditional,'remaining_limits':['Te/q still worse than prior-median point estimates; statistical disadvantages not decisive',
                 'Compression vs family/data contributions not fully isolated','Full neural predictive/population calibration unestablished',
                 'Finite mode/tail evidence','No independently generated final evaluation or real-data readiness'],
             'report':'BEFORE_AFTER.md','issues_before':'../ISSUES_BEFORE.json','issues_repaired_route':'../v2_02/ISSUES_AFTER.json',
             'incidents':incidents,'source_versions':versions()}
    atomic_json(out/'SUMMARY.json',summary)
    verify_preserved(review)
    atomic_json(out/'DELIVERY.json',{'utc':datetime.now(timezone.utc).isoformat(),'report_sha256':digest(out/'BEFORE_AFTER.md'),
                'summary_sha256':digest(out/'SUMMARY.json'),'assembler_sha256':digest(Path(__file__)),
                'before_binding_sha256':digest(review/'BEFORE_BINDING.json'),
                'code_and_plan_hashes':{str(p):digest(p) for p in sorted(Path('ml').glob('*')) if p.is_file() and p.suffix in ('.py','.json')},
                'figure_hashes':{str(p):digest(p) for p in [probe/'training_curves.png',review/'predictive_details_01/band_residuals.png']},
                'readme_sha256':digest('README.md'),'preserved_directories_match_before':True})
    atomic_json(review/'LATEST.json',{'execution':'complete','report':'final_01/BEFORE_AFTER.md','summary':'final_01/SUMMARY.json','scientific_limits_eliminated':False})
    atomic_json(review/'STATUS.json',{'execution':'complete','stage':'final_report','utc':datetime.now(timezone.utc).isoformat(),
                'scientific_limits_eliminated':False,'active_jobs':0})
    print('FINAL report saved: '+str(out/'BEFORE_AFTER.md'),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--review',type=Path,default=Path('output/ml_initial_20261023/review_01'));args=ap.parse_args()
    lock=Path('output/ml_initial_20261023/continuation.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,b'final_review_assembly');os.close(fd)
    try:execute(args.review)
    finally:lock.unlink()
