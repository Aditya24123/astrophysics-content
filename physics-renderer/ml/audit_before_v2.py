"""Read-only independent BEFORE inventory, before any v2 repair is applied."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
from scipy.fft import next_fast_len
from hii_sim.io import atomic_json, digest, versions
from bayes_followup import checked_diagnostics
from diagnostics import diagnostic_report


def read(p):
    return json.loads(Path(p).read_text())


def arrays(p):
    with np.load(p, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def fingerprint(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    old = Path('output/ml_initial_20261023/run_03')
    follow = Path('output/ml_initial_20261023/followup_01')
    plan = read(old/'run_binding.json')['plan']
    root = Path(plan['dataset'])
    sources = sorted(Path('ml').glob('*.py'))+[Path('README.md'), Path('ml/plan_v1.json'), Path('ml/followup_plan_v1.json')]
    binding = {'created_utc': datetime.now(timezone.utc).isoformat(), 'environment': versions(),
               'preserved_directories': {str(p): fingerprint(p) for p in
                    (old, follow, root, Path(plan['prepared']), Path('src/hii_sim'))},
               'pre_audit_sources': {str(p): digest(p) for p in sources},
               'test_policy': 'No new test prediction, test target unsealing, test rerender or test-based selection. Test archive hash/record audit only.'}
    atomic_json(out/'BEFORE_BINDING.json', binding)
    sys.path.insert(0, str(Path('tools').resolve()))
    from prepare_training_data import inventory
    inventory_result = inventory(root)
    # Direct array comparison, not the replay's asserted result.
    differences = {}
    for rel in ('full/best.npz', 'full/pca.npz'):
        original, replay = arrays(old/rel), arrays(follow/'training_runtime_replay'/rel)
        if original.keys() != replay.keys():
            raise ValueError('Replay array keys differ')
        for k in original:
            differences[rel+'/'+k] = {'max_absolute_difference': float(np.max(abs(original[k]-replay[k]))),
                                     'array_equal': bool(np.array_equal(original[k], replay[k]))}
    manifest = read(root/'public/manifest.json')['records']
    source_splits = {}
    for r in manifest:
        source_splits.setdefault(r['source_id'], set()).add(r['split'])
    if any(len(s) != 1 for s in source_splits.values()):
        raise ValueError('Source split leakage')
    expected = {}
    for r in manifest:
        if r['split'] == 'test':
            expected.setdefault(r['source_id'], r['realization_id'])
    profiles = read(follow/'test_forward_checks.json')['cases']
    observed = {r['source_id']: r['realization_id'] for r in profiles}
    old_profiles = read(old/'test_forward_checks.json')['cases']
    if len(profiles) != 48 or observed != expected or profiles[:len(old_profiles)] != old_profiles:
        raise ValueError('Forward-record completeness/preservation failed')
    config = read('configs/compact_proxy_gl4_v3.json')
    class Contract:
        pass
    like = Contract()
    like.config = config
    cases = []
    for n, case in enumerate(read(follow/'bayesian_comparison.json')['cases'], 1):
        raw = arrays(follow/f'bayes_{n:02d}_chains.npz')['coordinates']
        recomputed = checked_diagnostics(raw, like)
        delta = max(abs(recomputed['variables'][k][field]-case['sampling']['diagnostics']['variables'][k][field])
                    for k in recomputed['variables'] for field in ('rank_folded_split_rhat', 'bulk_ess', 'tail_ess'))
        physical = raw[:, :, :4].copy()
        for j, name in enumerate(config['reported_targets']):
            lo, hi, kind = config['sampling']['ranges'][name]
            physical[:, :, j] = lo*(hi/lo)**raw[:, :, j] if kind == 'log' else lo+(hi-lo)*raw[:, :, j]
        quantiles = np.quantile(physical.reshape(-1, 4), [.025, .5, .975], axis=0)
        q_delta = float(np.max(abs(quantiles-np.array(case['bayesian_quantiles_025_50_975']))))
        init = read(follow/f'bayes_{n:02d}_initialization.json')
        cases.append({'source_id': case['source_id'], 'realization_id': case['realization_id'],
                      'shape': list(raw.shape), 'chain_sha256': digest(follow/f'bayes_{n:02d}_chains.npz'),
                      'finite': bool(np.isfinite(raw).all()), 'recomputed_diagnostics': recomputed,
                      'max_saved_diagnostic_difference': delta, 'max_saved_quantile_difference': q_delta,
                      'initialization_costs': [m['cost'] for m in init['maps']],
                      'note': 'Broad optimizer INPUTS are not overdispersed final chain states; mode coverage still unproven'})
    val = arrays(follow/'validation/predictions.npz')
    metrics = {}
    for j, name in enumerate(config['reported_targets']):
        e = val['median'][:, j]-val['truth'][:, j]
        row = {'mae': float(abs(e).mean()), 'rmse': float(np.sqrt(np.mean(e*e))),
               'prior_midpoint_mae': read(follow/'validation/metrics.json')['parameters'][name]['prior_midpoint_mae'],
               'coverage': {}}
        for level, a, b in [(.5, 2, 4), (.8, 1, 5), (.95, 0, 6)]:
            lo, hi = val['quantiles'][:, a, j], val['quantiles'][:, b, j]
            row['coverage'][str(level)] = {'observed': float(np.mean((val['truth'][:, j] >= lo) & (val['truth'][:, j] <= hi))),
                                          'mean_width': float((hi-lo).mean())}
        metrics[name] = row
    # Reproduce defects on synthetic arrays without modifying any saved file.
    rng = np.random.default_rng(19)
    fake = rng.normal(size=(4, 4000, 2))
    fake[0, :, 1] += 8
    omission_pass = diagnostic_report(fake, ['only_first_variable'])['passed']
    stuck = diagnostic_report(np.ones((4, 100, 1)), ['stuck'])
    try:
        json.dumps(stuck, allow_nan=False)
        serializable = True
    except ValueError:
        serializable = False
    n = next(i for i in range(20, 500) if next_fast_len(2*i) % 2)
    result = {'created_utc': binding['created_utc'], 'state': 'BEFORE; no repair applied',
              'inventory': inventory_result, 'split_observations': dict(Counter(r['split'] for r in manifest)),
              'source_counts': dict(Counter(next(iter(s)) for s in source_splits.values())),
              'replay_array_comparisons': differences,
              'forward_record_check': {'cases': len(profiles), 'preserved_original_rows': len(old_profiles),
                  'all_selected_source_ids_match': True, 'optimizer_failures': sum(not r['optimizer_success'] for r in profiles),
                  'median_stored_rms': float(np.median([r['profiled_whitened_residual_rms_sigma'] for r in profiles])),
                  'max_stored_rms': max(r['profiled_whitened_residual_rms_sigma'] for r in profiles),
                  'limit': 'Structural/receipt audit only; no new test rerender. Profiles are not neural posterior predictive draws.'},
              'bayesian_raw_chain_audit': cases, 'validation_metrics_from_saved_arrays': metrics,
              'validation_observations': len(val['truth']), 'validation_sources': len(set(val['parent'])),
              'synthetic_defect_reproductions': {'omitted_bad_dimension_can_pass': omission_pass,
                  'stuck_failure_report_json_serializable': serializable,
                  'first_odd_fft_padding_example_chain_half_length': n,
                  'requested_fft_length': next_fast_len(2*n),
                  'legacy_irfft_default_length': 2*(len(np.fft.rfft(np.zeros(n), n=next_fast_len(2*n)))-1),
                  'nan_saved_diagnostic_passes_old_finalizer_predicate': not (float('nan') >= 1.01 or min(float('nan'), float('nan')) < 400)},
              'test_archive': {'sha256': digest(old/'test_predictions.npz'), 'status': 'Existing evaluated archive only; not rescored'},
              'original_run_checkpoint': digest(old/'full/best.npz')}
    atomic_json(out/'BEFORE_METRICS.json', result)
    issues = [
        ('I01','P1','Saved convergence/finalizer evidence not derived from raw chains at finalization; chain hashes not bound','Recompute raw-chain diagnostics, physical quantiles, source selection and predictive receipts; bind derived artifacts','Tampered chains, omitted/duplicate/missing cases, NaN diagnostics rejected','Actual raw-chain passes require independent recheck, not automatic withdrawal'),
        ('I02','P1','Diagnostic API silently accepts fewer names than variables; NaN/constant failure serialization and odd-FFT inverse-length handling are weak','Versioned strict finite diagnostics, all-variable dimension checks, explicit inverse FFT length, JSON-safe failures','Omitted variable, nonfinite/stuck chains, odd padding vs direct autocovariance','Existing 4000-draw chain FFT padding is even; numeric impact must be measured'),
        ('I03','P1','Neural forward records profile only marginal medians and fitted nuisances; no nuisance-correlated predictive integration','Implement conditional p(nuisance|theta,measurement) sampling for frozen neural theta draws; accept only converged conditionals','Known conditional Gaussian, cached likelihood vs original, all-nuisance convergence','Few outer neural draws and approximate neural theta posterior prevent global calibration claims'),
        ('I04','P2','Broad optimization starts collapse near modes; finite diagnostics cannot prove global mode coverage','Independent bounded validation multistart mode search and chain log-density/early-late checks','Displaced geometry modes and higher-density discoveries flagged','Global mode completeness is irreducible with finite exploration'),
        ('I05','P2','PhysicsLikelihood and Support assume PA 0..180, offsets +/-10 and log EM/size even if future config changes','Fail closed on unsupported config/observing units and prior definitions in new review route','Changed PA/offset/grid/target transforms rejected','Current documented config may already match exactly'),
        ('I06','P2','Validation/test/challenge roles and empirical coverage uncertainty can be confused','Explicit data-role ledger and source-based validation resampling with finite-design caveats; preserve test archive','Exact grouped counts, no test/challenge training, feature allowlist and train-only preprocessing checks','67 validation/48 test groups and Sobol design cannot certify frequentist calibration'),
        ('I07','P1','Frozen neural posterior is broad/weak for temperature and differs from validated Bayesian marginals','Quantify validation-only errors, widths and predictive mismatch; retain frozen model and propose later architecture/joint-nuisance work','Do not pick models or calibrate using test; no false fixed-accuracy claim','Insufficient design/posterior family cannot be eliminated by bookkeeping fixes'),
        ('I08','P2','Historical/follow-up timings and conditional-predictive work have different cost/precision scopes','Retain pilot-inclusive accounting; time new conditional work separately with call counts and CPU/thread environment','No matched-accuracy speedup from incomparable draw counts','Two cases/session effects remain'),
        ('I09','P2','Sampler final output requires draw count divisible by diagnostic interval; wall budget checked only at diagnostic milestones; interrupted conditional work needs honest status','New bounded conditional sampler validates settings and always records success/failure/time-stop with resumable state or explicit limitation','Invalid budgets, stop before first milestone, failed chain state serialization','Historical follow-up files/code preserved; new route does not imply old sampler has been retroactively repaired')]
    atomic_json(out/'ISSUES_BEFORE.json', {'issues': [dict(zip(('id','severity','evidence','intended_repair','regression_check','scientific_limit'), r)) for r in issues]})
    lines=['# BEFORE issue register','', 'Created before v2 fixes. See BEFORE_METRICS.json and BEFORE_BINDING.json for raw evidence.','',
           '| Issue | Severity | Evidence | Intended repair | Regression | Limit |','|---|---|---|---|---|---|']
    for row in issues:
        lines.append('| '+' | '.join(row)+' |')
    (out/'ISSUES_BEFORE.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'out': str(out), 'exact_replay_arrays': all(r['array_equal'] for r in differences.values()),
                      'forward_records': len(profiles), 'raw_chain_passes': [r['recomputed_diagnostics']['passed'] for r in cases],
                      'defects': result['synthetic_defect_reproductions']}), flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    main(ap.parse_args().out)
