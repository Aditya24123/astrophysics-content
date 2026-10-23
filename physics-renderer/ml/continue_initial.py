"""Audit and continue frozen run_03 WITHOUT training or test rescoring.

Use --stage all on a fresh output path. --resume permits later stages to load
only artifacts bound to the same frozen files, plan, and continuation code.
Interrupted sampling of a case is retained but not silently counted complete;
choose a new output path for an explicitly labelled new sampling attempt.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import least_squares
from hii_sim.io import atomic_json, digest, write_npz, versions
from hii_sim.loaders import NPEInputDataset
from hii_sim.training_data import load_transform
from model import Support, fit_pca, project, predict, sample
from bayes_followup import GroupedLikelihood, checked_diagnostics, sample_followup, NAMES
from run_initial import load_split, original_to_unit, evaluate


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def fingerprint(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def audit(old, out, plan, transform, pca, weights, support, config):
    sys.path.insert(0, str(Path('tools').resolve()))
    from prepare_training_data import inventory
    original = read(old/'run_binding.json')
    frozen = read(old/'FROZEN_BEFORE_TEST.json')
    root = Path(original['plan']['dataset'])
    inv = inventory(root)
    for name, path in [('checkpoint_sha256', old/'full/best.npz'), ('pca_sha256', old/'full/pca.npz')]:
        if frozen[name] != digest(path):
            raise ValueError(f'Frozen artifact changed: {path}')
    for path, sha in original['ml_source_sha256'].items():
        if digest(Path(path)) != sha:
            raise ValueError(f'Original experiment code changed: {path}')
    for path, key in [(root/'public/manifest.json', 'data_manifest_sha256'),
                      (root/'protected/provenance.json', 'data_provenance_sha256'),
                      (Path(original['plan']['prepared'])/'normalization.npz', 'normalization_sha256')]:
        if digest(path) != original[key]:
            raise ValueError(f'Dataset/preprocessing binding changed: {path}')
    counts = Counter(r['split'] for r in read(root/'public/manifest.json')['records'])
    if counts != Counter(train=794, validation=134, test=96, challenge=24):
        raise ValueError('Unexpected split counts')
    x, y, books = load_split(root, transform, 'train')
    vx, vy, vbooks = load_split(root, transform, 'validation')
    pca_started = time.perf_counter()
    refit = fit_pca(x, original['plan']['pca_components'])
    pca_seconds = time.perf_counter()-pca_started
    for k in pca:
        np.testing.assert_allclose(pca[k], refit[k], atol=1e-8, rtol=1e-8)
    # Prove normalization is fitted to training observations/unique parents only.
    from hii_sim.training_data import fit_training_transform
    norm = fit_training_transform(root)
    for k in ('feature_mean_jy', 'feature_scale_jy', 'target_mean', 'target_scale'):
        np.testing.assert_allclose(getattr(transform, k), getattr(norm, k), rtol=0, atol=0)
    history = [json.loads(s) for s in (old/'full/epochs.jsonl').read_text().splitlines()]
    info = read(old/'full/training.json')
    if info['best_epoch'] != 22 or info['epochs'] != 52 or len(history) != 52:
        raise ValueError('Saved selection/stopping reports inconsistent')
    saved = arrays(old/'test_predictions.npz')
    metrics = read(old/'test_evaluation.json')
    if saved['truth'].shape != (96, 4) or len(set(saved['parent'])) != 48:
        raise ValueError('Saved test archive incomplete')
    np.testing.assert_allclose(saved['median'], support.decode(saved['transformed_mean']))
    for j, name in enumerate(NAMES[:4]):
        np.testing.assert_allclose(np.mean(abs(saved['median'][:, j]-saved['truth'][:, j])),
                                   metrics['parameters'][name]['mae_original_units'])
    # Validation only. Reuse original metrics implementation, with explicit names.
    valdir = out/'validation'
    valdir.mkdir(exist_ok=False)
    evaluate(vx, vy, vbooks, pca, weights, support, original['plan'], valdir)
    (valdir/'test_predictions.npz').rename(valdir/'predictions.npz')
    val = read(valdir/'test_evaluation.json')
    val['split'] = 'validation'
    atomic_json(valdir/'metrics.json', val)
    (valdir/'test_evaluation.json').unlink()
    pilot = read(old/'bayesian_comparison.json')
    pilot_diagnostics = []
    public = NPEInputDataset(root/'public', split='validation')
    for n, case in enumerate(pilot['cases']):
        i = next(i for i, r in enumerate(public.records) if r['realization_id'] == case['realization_id'])
        like = GroupedLikelihood(public[i], config)
        chain = arrays(old/f'bayes_{n+1:02d}_chains.npz')['coordinates']
        pilot_diagnostics.append({'source_id': case['source_id'],
                                  'original_gate_pass': case['converged_targets_by_predeclared_gate'],
                                  'rank_diagnostics': checked_diagnostics(chain, like)})
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'inventory': inv,
              'normalization_exactly_reproduced_from_train_only': True,
              'pca_reproduced_from_train_only': True, 'split_observations': dict(counts),
              'pca_verification_refit_wall_seconds': pca_seconds,
              'checkpoint_sha256': digest(old/'full/best.npz'), 'selected_epoch': 22, 'stopped_epoch': 52,
              'saved_test_archive_consistent': True,
              'test_status': 'Previously evaluated 96 observations /48 sources. Archive consistency checked; no new test prediction or selection.',
              'saved_forward_cases': len(read(old/'test_forward_checks.json')['cases']),
              'original_completion_exists': (old/'completion.json').exists(),
              'pilot_diagnostics': pilot_diagnostics,
              'input_contract': '3906 normalized measured real Fourier components only; labels separate supervised loss arguments; IDs/seeds/true sky/realized nuisances excluded',
              'targets': [{'name': n, 'unit': u, 'prior': config['sampling']['ranges'][n]}
                          for n, u in zip(NAMES[:4], ['pc cm^-6', 'K', 'arcsec (full major EM FWHM)', '1'])],
              'transform': 'Loader log10 EM/size then train-parent standardization; original model instead encodes originals as prior-unit logit/(pi/sqrt(3)); no label transform fitted on holdout.',
              'new_dependencies': [], 'environment': versions()}
    atomic_json(out/'audit.json', result)
    print('Verified frozen archive, 3712-file inventory, train-only normalization/PCA, and validation metrics.', flush=True)


def finish_profiles(old, out, root, config):
    archive = arrays(old/'test_predictions.npz')
    public = NPEInputDataset(root/'public', split='test')
    expected = {}
    for r in public.records:
        expected.setdefault(r['source_id'], r['realization_id'])
    prior = read(old/'test_forward_checks.json')
    destination = out/'test_forward_checks.json'
    existing = read(destination) if destination.exists() else prior
    rows = existing['cases'][:]
    seen = set()
    for r in rows:
        if r['source_id'] in seen or expected.get(r['source_id']) != r['realization_id']:
            raise ValueError('Partial forward records duplicate or mismatch selection')
        seen.add(r['source_id'])
    medians = dict(zip(archive['realization'].tolist(), archive['median']))
    started = time.perf_counter()
    for i, record in enumerate(public.records):
        if record['source_id'] in seen or expected[record['source_id']] != record['realization_id']:
            continue
        inp = public[i]
        like = GroupedLikelihood(inp, config)
        fixed = original_to_unit(medians[record['realization_id']], config)
        fits = [least_squares(lambda n: like.residual(np.r_[fixed, n]), [pa, .5, .5, 0., 0.],
                            bounds=([0, 0, 0, -7, -7], [1, 1, 1, 7, 7]),
                            max_nfev=40, diff_step=1e-4, ftol=1e-5) for pa in (.25, .75)]
        best = min(fits, key=lambda f: f.cost)
        residual = like.residual(np.r_[fixed, best.x])[:-2]
        rows.append({**inp['bookkeeping'], 'profiled_whitened_residual_rms_sigma': float(np.sqrt(np.mean(residual**2))),
                     'optimizer_success': bool(best.success), 'nfev': int(best.nfev),
                     'nuisance_coordinates': best.x.tolist()})
        seen.add(record['source_id'])
        atomic_json(destination, {'cases': rows, 'definition': prior['definition'],
                    'original_saved_cases': 46, 'continued_cases': len(rows)-46,
                    'original_partial_elapsed_seconds': prior['elapsed_seconds'],
                    'continuation_elapsed_seconds': time.perf_counter()-started,
                    'test_policy': 'Uses saved frozen medians and public measurements only; no new test scoring or tuning.'})
        print(f'Forward profiles saved: {len(rows)}/48', flush=True)
    if len(rows) != 48:
        raise ValueError('Missing forward cases')


def neural_benchmark(inp, transform, pca, weights, support, n, seed):
    # Same observation and same CPU thread environment as conventional inference.
    rng = np.random.default_rng(seed)
    x = transform.features(inp['features']['measurements'].values_jy)[None]
    support.decode(sample(project(x, pca), weights, n, rng))  # warmup
    times, medtimes = [], []
    start = time.perf_counter()
    for _ in range(30):
        s = time.perf_counter()
        raw = transform.features(inp['features']['measurements'].values_jy)[None]
        z = project(raw, pca)
        mean, _, _ = predict(z, weights)
        support.decode(mean)
        medtimes.append(time.perf_counter()-s)
        s = time.perf_counter()
        support.decode(sample(z, weights, n, rng))
        times.append(time.perf_counter()-s+medtimes[-1])
    return {'posterior_draws': n, 'repetitions': 30,
            'median_point_inference_seconds': float(np.median(medtimes)),
            'median_posterior_inference_seconds': float(np.median(times)),
            'measurement_loading_included': False, 'normalization_and_pca_included': True,
            'note': 'Approximate neural draws, not MCMC effective samples; no scientific speedup claim from unmatched precision.'}


def compare(old, out, root, config, plan, transform, pca, weights, support):
    public = NPEInputDataset(root/'public', split='validation')
    original = read(old/'bayesian_comparison.json')
    selected = sorted({r['source_id'] for r in public.records})[:2]
    rows = []
    for number, sid in enumerate(selected):
        tag = f'bayes_{number+1:02d}'
        finished = out/f'{tag}_comparison.json'
        if finished.exists():
            rows.append(read(finished))
            continue
        if (out/f'{tag}_chains.npz').exists():
            raise ValueError('Incomplete sampler artifact retained. Use new output for a new attempt; never silently restart or count it completed.')
        i = next(i for i, r in enumerate(public.records) if r['source_id'] == sid)
        inp = public[i]
        if original['cases'][number]['realization_id'] != inp['bookkeeping']['realization_id']:
            raise ValueError('Comparator observation changed')
        started = time.perf_counter()
        like = GroupedLikelihood(inp, config)
        pilot = arrays(old/f'{tag}_chains.npz')['coordinates']
        # Verify optimized algebra against the original renderer at prior extremes,
        # random prior points and posterior states on this validation observation.
        rng = np.random.default_rng(plan['seed']+number)
        points = [np.r_[[1, 0, 0, 0, .9, 1, 0], 2, -2]]
        points += [np.r_[rng.uniform(size=7), rng.normal(size=2)] for _ in range(6)]
        points += list(pilot[:, -1])
        errors = [float(np.linalg.norm(like.whiten(like.vector(t)-like.vector(t, reference=True)))) for t in points]
        if max(errors) > 1e-5:
            raise ValueError('Accelerated/reference forward mismatch')
        benchmark = neural_benchmark(inp, transform, pca, weights, support,
                                     plan['neural_posterior_draws'], plan['seed']+number)
        chain, stats = sample_followup(like, pilot, plan, plan['seed']+number*10000, out, tag)
        flat = chain.reshape(-1, 9)
        physical = np.array([like.unpack(np.r_[t[:4], t[4] % 1, t[5:]])[0] for t in flat])
        neural = support.decode(sample(project(transform.features(like.y)[None], pca), weights,
                                       plan['neural_posterior_draws'], rng))[0]
        # Truth is copied from the already evaluated validation pilot only AFTER
        # sampling. It is never a proposal/likelihood/initialization input.
        truth = np.array(original['cases'][number]['truth'])
        case = {'source_id': sid, 'realization_id': inp['bookkeeping']['realization_id'],
                'converged': stats['diagnostics']['passed'], 'sampling': stats,
                'reference_check_whitened_l2_max': max(errors), 'neural_runtime': benchmark,
                'comparison_validity': 'Two validation observations only; no population error/coverage ranking. Posterior summaries usable only if every convergence gate passes.',
                'truth': truth.tolist(),
                'neural_quantiles_025_50_975': np.quantile(neural, [.025, .5, .975], axis=0).tolist()}
        case['bayesian_quantiles_025_50_975'] = np.quantile(physical, [.025, .5, .975], axis=0).tolist() if case['converged'] else None
        if case['converged']:
            for method, draws in [('bayesian', physical), ('neural', neural)]:
                med = np.median(draws, axis=0)
                lo, hi = np.quantile(draws, [.025, .975], axis=0)
                case[method+'_absolute_error_original_units'] = abs(med-truth).tolist()
                case[method+'_95_width_original_units'] = (hi-lo).tolist()
                case[method+'_truth_in_95_interval'] = ((truth >= lo) & (truth <= hi)).tolist()
            checks = []
            for k in rng.choice(len(flat), plan['posterior_predictive_draws'], replace=False):
                t = flat[k].copy()
                t[4] %= 1
                r = like.whiten(like.vector(t)-like.y)
                epsilon = rng.normal(size=len(r))
                # T(y,theta) vs T(y_rep,theta); whitened independent normal errors
                # are exactly equivalent to fresh correlated thermal replicates.
                checks.append({'observed_discrepancy_mean_square': float(np.mean(r*r)),
                               'replicated_discrepancy_mean_square': float(np.mean(epsilon**2)),
                               'replicate_minus_observed_rms': float(np.sqrt(np.mean((r+epsilon)**2)))})
            case['posterior_predictive'] = {'draws': checks,
                'tail_fraction': float(np.mean([r['replicated_discrepancy_mean_square'] >= r['observed_discrepancy_mean_square'] for r in checks])),
                'definition': 'Full joint target+nuisance posterior samples. Fresh conditional correlated thermal replicate via whitening; observed and replicate discrepancies measured relative to their SAME sampled expected signal. Gains are latent per-observation posterior draws, not newly independent gains.',
                'limits': '128 dependent posterior selections; Bayesian predictive p-value is descriptive and not uniform; two validation sources only.'}
        else:
            case['posterior_predictive'] = {'status': 'NOT VALIDATED: convergence failed; no posterior predictive conclusion'}
        case['total_wall_seconds_including_setup_checks_and_predictive'] = time.perf_counter()-started
        case['runtime_limits'] = 'Neural training separate; inference excludes disk load. Conventional sampler wall includes measurement-only initialization and warmup. Failed pilots are development cost, not inference speed evidence.'
        atomic_json(finished, case)
        rows.append(case)
        atomic_json(out/'bayesian_comparison.json', {'cases': rows,
                    'fairness': 'Same raw observation, bounds, continuous target prior, PA/offset priors, lognormal common gain, Gaussian spectral slope, conditional correlated thermal covariance, and original renderer. Neural PCA32 is information compression and posterior family approximation; simulator uses a finite scrambled-Sobol training design rather than iid continuous prior draws. Neural marginalizes nuisance variation approximately via training; conventional samples their joint posterior explicitly.',
                    'decision': 'No population winner or general speedup claim from two validation sources.'})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--plan', type=Path, default=Path('ml/followup_plan_v1.json'))
    ap.add_argument('--stage', choices=['all', 'audit', 'forward', 'bayes', 'report'], default='all')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    plan = read(args.plan)
    old = Path(plan['original_run'])
    original = read(old/'run_binding.json')
    frozen = read(old/'FROZEN_BEFORE_TEST.json')
    # Always verify critical bindings before any stage, including report-only.
    if digest(old/'full/best.npz') != frozen['checkpoint_sha256'] or digest(old/'full/pca.npz') != frozen['pca_sha256']:
        raise ValueError('Frozen neural files changed')
    lock = Path('output/ml_initial_20261023/continuation.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({'pid': os.getpid(), 'out': str(args.out), 'utc': datetime.now(timezone.utc).isoformat()}).encode())
    os.close(fd)
    try:
        args.out.mkdir(parents=True, exist_ok=args.resume)
        binding_path = args.out/'binding.json'
        source_paths = [Path('ml/continue_initial.py'), Path('ml/bayes_followup.py'), Path('ml/diagnostics.py'), Path('ml/report_initial.py'), args.plan]
        binding = {'original_artifact_hashes': fingerprint(old), 'settings': plan,
                   'code_hashes': {str(p): digest(p) for p in source_paths}, 'environment': versions(),
                   'thread_environment': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}
        if args.resume:
            if read(binding_path) != binding:
                raise ValueError('Continuation binding changed; preserve prior results and use a new output')
        else:
            atomic_json(binding_path, binding)
        root = Path(original['plan']['dataset'])
        transform = load_transform(original['plan']['prepared'], root/'public')
        pca, weights = arrays(old/'full/pca.npz'), arrays(old/'full/best.npz')
        config = read('configs/compact_proxy_gl4_v3.json')
        support = Support(config)
        if args.stage in ('all', 'audit'):
            if not (args.out/'audit.json').exists():
                audit(old, args.out, plan, transform, pca, weights, support, config)
        if args.stage in ('all', 'forward'):
            if not (args.out/'audit.json').exists():
                raise ValueError('Run audit first')
            finish_profiles(old, args.out, root, config)
        if args.stage in ('all', 'bayes'):
            if not (args.out/'audit.json').exists():
                raise ValueError('Run audit first')
            compare(old, args.out, root, config, plan, transform, pca, weights, support)
        if args.stage in ('all', 'report'):
            from report_initial import build_report
            build_report(old, args.out)
        if fingerprint(old) != binding['original_artifact_hashes']:
            raise ValueError('Original scientific archive changed during continuation')
        atomic_json(args.out/'status.json', {'utc': datetime.now(timezone.utc).isoformat(), 'last_stage': args.stage,
                    'original_archive_unchanged': True, 'new_test_scoring': False, 'neural_retrained': False,
                    'science_complete': False, 'note': 'Stage execution status is not a scientific acceptance certificate.'})
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
