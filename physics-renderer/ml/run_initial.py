"""Train, freeze, and evaluate the predeclared bounded synthetic experiment."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2, norm
from hii_sim.io import atomic_json, digest, canonical_hash, write_npz, versions
from hii_sim.loaders import NPEInputDataset, PairedNPEDataset
from hii_sim.training_data import load_transform, iter_batches, _target_values
from model import Support, fit_pca, project, initialize, loss_grad, predict, sample, covariance
from bayes import PhysicsLikelihood, fit_posterior


def utc():
    return datetime.now(timezone.utc).isoformat()


def memory():
    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)]+[
            (n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
             'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
             'PagefileUsage', 'PeakPagefileUsage')]
    value = PMC()
    value.cb = ctypes.sizeof(value)
    get_handle = ctypes.windll.kernel32.GetCurrentProcess
    get_handle.restype = wintypes.HANDLE
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    query.restype = wintypes.BOOL
    try:
        ok = query(get_handle(), ctypes.byref(value), value.cb)
    except OSError as exc:
        return {'memory_monitor_unavailable': str(exc)}
    if not ok:
        return {'memory_monitor_unavailable': 'Windows process memory query rejected'}
    return {'working_set_mib': value.WorkingSetSize/2**20, 'peak_working_set_mib': value.PeakWorkingSetSize/2**20}


def load_split(root, transform, split):
    xs, ys, books = [], [], []
    for b in iter_batches(root, transform, split=split, batch_size=64, include_targets=True):
        assert set(b['features']) == {'values'}
        xs.append(b['features']['values'])
        ys.append(b['targets']['original'])
        books.extend(b['bookkeeping'])
    return np.concatenate(xs).astype(float), np.concatenate(ys), books


def train(name, x, y, vx, vy, plan, out):
    rng = np.random.default_rng(plan['seed'])
    pca = fit_pca(x, plan['pca_components'])
    x, vx = project(x, pca), project(vx, pca)
    p = initialize(x.shape[1], plan['hidden'], rng)
    m, v = [{k: np.zeros_like(a) for k, a in p.items()} for _ in range(2)]
    best, stale, step = np.inf, 0, 0
    start, cpu = time.perf_counter(), time.process_time()
    history = []
    dest = out/name
    dest.mkdir(exist_ok=False)
    write_npz(dest/'pca.npz', **pca)
    print(json.dumps({'event': 'training_started', 'name': name, 'utc': utc(),
                      'rows': len(x), 'parameter_count': sum(a.size for a in p.values())}), flush=True)
    for epoch in range(1, plan['epochs']+1):
        order = rng.permutation(len(x))
        for offset in range(0, len(x), plan['batch_size']):
            idx = order[offset:offset+plan['batch_size']]
            loss, gradients = loss_grad(x[idx], y[idx], p, plan['weight_decay'])
            if not np.isfinite(loss) or not all(np.isfinite(a).all() for a in gradients.values()):
                raise FloatingPointError('Nonfinite loss/gradient')
            normg = np.sqrt(sum(np.sum(g*g) for g in gradients.values()))
            step += 1
            for k in p:
                g = gradients[k]*min(1., plan['gradient_clip']/max(normg, 1e-30))
                m[k] = .9*m[k]+.1*g
                v[k] = .999*v[k]+.001*g*g
                p[k] -= plan['learning_rate']*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        tl = float(loss_grad(x, y, p, gradient=False).mean())
        vl = float(loss_grad(vx, vy, p, gradient=False).mean())
        if not np.isfinite(tl+vl):
            raise FloatingPointError('Nonfinite epoch loss')
        row = {'epoch': epoch, 'train_nll': tl, 'validation_nll': vl,
               'wall_seconds': time.perf_counter()-start, 'cpu_seconds': time.process_time()-cpu, **memory()}
        history.append(row)
        with (dest/'epochs.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if vl < best-plan['min_delta']:
            best, stale, best_epoch = vl, 0, epoch
            write_npz(dest/'best.npz', **p)
            atomic_json(dest/'best.json', {**row, 'step': step})
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({'fit': name, **row}), flush=True)
        if stale >= plan['patience'] or time.perf_counter()-start > plan['fit_wall_seconds']:
            break
    reason = 'validation_patience' if stale >= plan['patience'] else ('wall_budget' if epoch < plan['epochs'] else 'epoch_budget')
    info = {'name': name, 'best_epoch': best_epoch, 'epochs': epoch, 'best_validation_nll': best,
            'stopping_reason': reason, 'elapsed_seconds': time.perf_counter()-start,
            'cpu_seconds': time.process_time()-cpu, 'train_rows': len(x),
            'pca_explained_fraction': float(pca['explained_fraction']),
            'parameter_count': sum(a.size for a in p.values()), **memory()}
    atomic_json(dest/'training.json', info)
    with np.load(dest/'best.npz') as a:
        p = {k: a[k].copy() for k in a.files}
    return pca, p, info


def metric_distribution(a):
    a = np.asarray(a)
    return {'mean': float(np.mean(a)), 'median': float(np.median(a)),
            'p10': float(np.quantile(a, .1)), 'p90': float(np.quantile(a, .9)), 'max': float(np.max(a))}


def evaluate(x, truth, books, pca, p, support, plan, out):
    rng = np.random.default_rng(plan['seed']+11)
    z = project(x, pca)
    mu, a, _ = predict(z, p)
    draws = support.decode(sample(z, p, plan['posterior_draws'], rng))
    med = support.decode(mu)
    levels = [.5, .8, .95]
    quantiles = np.quantile(draws, [.025, .1, .25, .5, .75, .9, .975], axis=1).transpose(1, 0, 2)
    groups = sorted({b['source_id'] for b in books})
    indices = [np.array([i for i, b in enumerate(books) if b['source_id'] == g]) for g in groups]
    boot = rng.integers(0, len(groups), size=(plan['bootstrap_parent_draws'], len(groups)))
    by_target = {}
    names = ('em0_pc_cm6', 'te_k', 'major_fwhm_arcsec', 'axis_ratio')
    nll = loss_grad(z, support.encode(truth), p, gradient=False)
    for j, name in enumerate(names):
        error = med[:, j]-truth[:, j]
        rel = abs(error)/truth[:, j]
        row = {'mae_original_units': float(abs(error).mean()), 'rmse_original_units': float(np.sqrt(np.mean(error**2))),
               'absolute_error_original_units': metric_distribution(abs(error)),
               'absolute_relative_error': metric_distribution(rel),
               'prior_midpoint_mae': float(abs(support.decode(np.zeros((1, 4)))[0, j]-truth[:, j]).mean())}
        if j in (0, 2):
            row['absolute_log10_error_dex'] = metric_distribution(abs(np.log10(med[:, j]/truth[:, j])))
        row['coverage'] = {}
        for level in levels:
            lo, hi = np.quantile(draws[:, :, j], [(1-level)/2, (1+level)/2], axis=1)
            covered = (truth[:, j] >= lo) & (truth[:, j] <= hi)
            parent = np.array([covered[idx].mean() for idx in indices])
            interval = np.quantile(parent[boot].mean(axis=1), [.025, .975])
            row['coverage'][str(level)] = {'observed': float(covered.mean()), 'parent_bootstrap_95ci': interval.tolist(),
                                         'mean_interval_width_original_units': float((hi-lo).mean())}
        pit = np.mean(draws[:, :, j] < truth[:, None, j], axis=1)
        row['pit_histogram_10bins'] = np.histogram(pit, bins=np.linspace(0, 1, 11))[0].tolist()
        parent_ae = np.array([abs(error[idx]).mean() for idx in indices])
        row['mae_parent_bootstrap_95ci'] = np.quantile(parent_ae[boot].mean(axis=1), [.025, .975]).tolist()
        by_target[name] = row
    mahal = np.sum(np.einsum('bij,bj->bi', a, support.encode(truth)-mu)**2, axis=1)
    joint = {str(level): float(np.mean(mahal <= chi2.ppf(level, 4))) for level in levels}
    report = {'observations': len(truth), 'independent_parents': len(groups), 'mean_nll_scaled_logit_coordinates': float(nll.mean()),
              'parameters': by_target, 'joint_transformed_ellipsoid_coverage': joint,
              'finite_outputs': bool(np.isfinite(draws).all()),
              'definitions': 'point estimate is marginal posterior median; absolute relative error=abs(pred-truth)/truth; two realizations/source, bootstrap resamples source groups; intervals central equal-tail; coverage is empirical on a small heldout set, not proof of calibration',
              'loss_unit': 'nats per observation density in scaled logit prior coordinates; not comparable to original-unit densities without Jacobian',
              'posterior_family': 'full covariance logit-normal, bounded support, single mode in logit coordinates; PCA compression may lose information'}
    write_npz(out/'test_predictions.npz', truth=truth, median=med, quantiles=quantiles, transformed_mean=mu,
              precision_factor=a, nll=nll, parent=np.array([b['source_id'] for b in books]),
              realization=np.array([b['realization_id'] for b in books]))
    atomic_json(out/'test_evaluation.json', report)
    return report, med


def original_to_unit(values, config):
    out = []
    for name, value in zip(('em0_pc_cm6', 'te_k', 'major_fwhm_arcsec', 'axis_ratio'), values):
        lo, hi, kind = config['sampling']['ranges'][name]
        out.append(np.log(value/lo)/np.log(hi/lo) if kind == 'log' else (value-lo)/(hi-lo))
    return np.array(out)


def comparator(root, config, support, pca, p, transform, plan, out):
    public = NPEInputDataset(root/'public', split='validation')
    chosen = sorted({r['source_id'] for r in public.records})[:plan['baseline_validation_sources']]
    results = []
    for number, sid in enumerate(chosen):
        index = next(i for i, r in enumerate(public.records) if r['source_id'] == sid)
        inp = public[index]
        like = PhysicsLikelihood(inp, config)
        started = time.perf_counter()
        point = np.r_[np.full(7, .5), 0., 0.]
        reference = like.vector(point, reference=True)
        accelerated = like.vector(point)
        discrepancy = float(np.linalg.norm(like.whiten(reference-accelerated)))
        if discrepancy > 1e-5:
            raise ValueError('Accelerated forward failed reference check')
        chains, stats = fit_posterior(like, plan, plan['seed']+100+number)
        write_npz(out/f'bayes_{number+1:02d}_chains.npz', coordinates=chains)
        flat = chains.reshape(-1, 9)
        bayes_original = np.array([like.unpack(t)[0] for t in flat])
        radio = transform.features(like.y)[None]
        neural = support.decode(sample(project(radio, pca), p, plan['posterior_draws'],
                                       np.random.default_rng(plan['seed']+200+number)))[0]
        stats.update(source_id=sid, realization_id=inp['bookkeeping']['realization_id'],
                     accelerated_reference_whitened_error=discrepancy,
                     bayesian_marginal_quantiles=np.quantile(bayes_original, [.025, .5, .975], axis=0).tolist(),
                     neural_marginal_quantiles=np.quantile(neural, [.025, .5, .975], axis=0).tolist())
        # Paired access only after inference, for descriptive comparison; never optimizer inputs.
        truth = _target_values(PairedNPEDataset(root, split='validation')[index][1])
        stats['truth'] = truth.tolist()
        stats['bayesian_median_absolute_relative_error'] = (abs(np.median(bayes_original, axis=0)-truth)/truth).tolist()
        stats['neural_median_absolute_relative_error'] = (abs(np.median(neural, axis=0)-truth)/truth).tolist()
        residuals = []
        rng = np.random.default_rng(plan['seed']+300+number)
        for k in rng.choice(len(flat), min(32, len(flat)), replace=False):
            r = like.whiten(like.vector(flat[k])-like.y)
            residuals.append({'expected_signal_residual_rms_sigma': float(np.sqrt(np.mean(r*r))),
                              'replicate_residual_rms_sigma': float(np.sqrt(np.mean((r+rng.normal(size=len(r)))**2)))})
        stats['posterior_forward_residuals'] = residuals
        stats['total_wall_seconds'] = time.perf_counter()-started
        results.append(stats)
        atomic_json(out/'bayesian_comparison.json', {'selection': plan['baseline_selection'], 'cases': results,
                    'fairness': 'Same full radio vectors, source/calibration/noise priors and forward physics; neural uses train-only PCA compression and an approximate marginal posterior; conventional samples all nine coordinates. Small validation subset, not a test-wide performance ranking.',
                    'winner': 'No superiority claim from two sources or unconverged chains'})
    return results


def forward_test(root, config, books, med, plan, out):
    public = NPEInputDataset(root/'public', split='test')
    by_id = {b['realization_id']: (i, med[i]) for i, b in enumerate(books)}
    seen, rows = set(), []
    start = time.perf_counter()
    for inp in public:
        book = inp['bookkeeping']
        if book['source_id'] in seen:
            continue
        seen.add(book['source_id'])
        _, point = by_id[book['realization_id']]
        like = PhysicsLikelihood(inp, config)
        fixed = original_to_unit(point, config)
        fits = []
        for pa in (.25, .75):
            fit = least_squares(lambda n: like.residual(np.r_[fixed, n]), [pa, .5, .5, 0., 0.],
                                bounds=([0, 0, 0, -7, -7], [1, 1, 1, 7, 7]),
                                max_nfev=40, diff_step=1e-4, ftol=1e-5)
            fits.append(fit)
        best = min(fits, key=lambda f: f.cost)
        r = like.residual(np.r_[fixed, best.x])[:-2]
        rows.append({**book, 'profiled_whitened_residual_rms_sigma': float(np.sqrt(np.mean(r*r))),
                     'optimizer_success': bool(best.success), 'nfev': int(best.nfev),
                     'nuisance_coordinates': best.x.tolist()})
        if len(rows) % 4 == 0:
            print(f'test forward profiles {len(rows)}/48 elapsed {time.perf_counter()-start:.1f}s', flush=True)
        atomic_json(out/'test_forward_checks.json', {'cases': rows,
              'definition': 'Targets fixed to neural marginal medians; PA, position and calibration fit to same measurement. Joint thermal-whitened residual RMS, gains priors included in fit. This optimistic nuisance-profile diagnostic is NOT a draw from the full neural posterior predictive distribution; neural estimator does not infer nuisance-target correlations.',
              'elapsed_seconds': time.perf_counter()-start})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, default=Path('ml/plan_v1.json'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(plan['dataset']).resolve()
    prepared = Path(plan['prepared']).resolve()
    config = json.loads(Path('configs/compact_proxy_gl4_v3.json').read_text())
    support = Support(config)
    transform = load_transform(prepared, root/'public')
    started = time.perf_counter()
    source_paths = sorted(Path('ml').glob('*.py'))+[args.plan]
    binding = {'created_utc': utc(), 'plan': plan, 'plan_sha256': digest(args.plan),
               'ml_source_sha256': {str(p): digest(p) for p in source_paths},
               'environment': versions(), 'cpu': platform.processor(), 'logical_cpus': os.cpu_count(),
               'compute': 'CPU only, two BLAS threads, GPU unused, VRAM allocated=0; CIM hardware inventory unavailable under sandbox',
               'thread_environment': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
               'data_manifest_sha256': digest(root/'public/manifest.json'),
               'data_provenance_sha256': digest(root/'protected/provenance.json'),
               'normalization_sha256': digest(prepared/'normalization.npz'),
               'split_membership_sha256': transform.metadata['train_membership_sha256'],
               'readiness_recheck_sha256': digest(Path('output/ml_initial_20261023/readiness_01/readiness_report.json'))}
    recheck = json.loads(Path('output/ml_initial_20261023/readiness_01/readiness_report.json').read_text())
    if recheck['status'] != 'PASS_FOR_INITIAL_DECLARED_PROXY_EXPERIMENT':
        raise ValueError('Fresh readiness failed')
    atomic_json(out/'run_binding.json', binding)
    # Record and block decoded test-label/truth file access during development.
    manifest = json.loads((root/'public/manifest.json').read_text())
    sealed = {str((root/'protected/labels'/(r['source_id']+'.json')).resolve()).lower()
              for r in manifest['records'] if r['split'] in ('test', 'challenge')}
    guard = {'active': True, 'denied': []}
    def hook(event, a):
        if event == 'open' and guard['active'] and isinstance(a[0], (str, bytes)):
            path = str(Path(str(a[0])).resolve()).lower()
            if path in sealed:
                guard['denied'].append(path)
                raise PermissionError('Holdout labels/truth sky sealed until frozen checkpoint')
    sys.addaudithook(hook)
    # The verified paired loader hashes committed truth FITS as raw bytes for
    # integrity. Permit hashing, but explicitly forbid decoding any FITS image.
    from astropy.io import fits
    original_fits_open = fits.open
    def no_fits_decode(*args, **kwargs):
        raise PermissionError('Truth/image decoding forbidden during ML development')
    fits.open = no_fits_decode
    x, original, books = load_split(root, transform, 'train')
    vx, vorig, vbooks = load_split(root, transform, 'validation')
    train_ids = {b['source_id'] for b in books}
    val_ids = {b['source_id'] for b in vbooks}
    test_ids = {r['source_id'] for r in manifest['records'] if r['split'] == 'test'}
    assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
    assert len(train_ids) == 397 and len(val_ids) == 67 and len(test_ids) == 48
    y, vy = support.encode(original), support.encode(vorig)
    pca, p, training = train('full', x, y, vx, vy, plan, out)
    half_ids = set(sorted(train_ids)[:len(train_ids)//2])
    half = np.array([b['source_id'] in half_ids for b in books])
    _, _, half_info = train('half_diagnostic', x[half], y[half], vx, vy, plan, out)
    freeze = {'utc': utc(), 'plan_sha256': digest(args.plan), 'checkpoint_sha256': digest(out/'full/best.npz'),
              'pca_sha256': digest(out/'full/pca.npz'), 'selected': training,
              'half_diagnostic': half_info, 'test_labels_opened': False, 'guard_denied_accesses': guard['denied'],
              'source_group_counts': {'train': len(train_ids), 'validation': len(val_ids), 'test': len(test_ids)},
              'test_once_policy': 'No architecture, preprocessing or interval tuning after this record; no rerun after viewing test scores'}
    atomic_json(out/'FROZEN_BEFORE_TEST.json', freeze)
    print(json.dumps({'event': 'model_frozen_test_unsealing', 'utc': utc(), 'checkpoint_sha256': freeze['checkpoint_sha256']}), flush=True)
    guard['active'] = False
    fits.open = original_fits_open
    tx, ty, tbooks = [], [], []
    for inp, labels in PairedNPEDataset(root, split='test'):
        tx.append(transform.features(inp['features']['measurements'].values_jy))
        ty.append(_target_values(labels))
        tbooks.append(inp['bookkeeping'])
    report, med = evaluate(np.array(tx), np.array(ty), tbooks, pca, p, support, plan, out)
    print(json.dumps({'event': 'frozen_test_evaluated', 'observations': len(tx), 'finite': report['finite_outputs']}), flush=True)
    # Evaluation has ended model selection. Subsequent diagnostics cannot change this checkpoint.
    comparisons = comparator(root, config, support, pca, p, transform, plan, out)
    forwards = forward_test(root, config, tbooks, med, plan, out)
    finish = {'utc': utc(), 'elapsed_seconds': time.perf_counter()-started, **memory(),
              'decision': 'INITIAL_PIPELINE_RUN_COMPLETE; scientific accuracy not established; needs more data/model work',
              'checkpoint_sha256_unchanged': digest(out/'full/best.npz') == freeze['checkpoint_sha256'],
              'baseline_converged_cases': sum(c['converged_targets_by_predeclared_gate'] for c in comparisons),
              'baseline_cases': len(comparisons), 'test_forward_profiles': len(forwards)}
    atomic_json(out/'completion.json', finish)
    print(json.dumps(finish), flush=True)


if __name__ == '__main__':
    main()
