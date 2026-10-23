"""Explicit train/validation-only runtime replay; never replaces frozen run_03.

The original optimizer timer omitted PCA/loading. This reproduces the original
fit in a NEW directory to measure preprocessing, fitting, and I/O separately.
No test/challenge targets or test measurements are evaluated.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from hii_sim.io import atomic_json, digest, versions
from hii_sim.training_data import fit_training_transform, load_transform, save_transform
from model import Support
from run_initial import load_split, train


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--run', type=Path, default=Path('output/ml_initial_20261023/run_03'))
    args = ap.parse_args()
    old = args.run
    binding = json.loads((old/'run_binding.json').read_text())
    frozen = json.loads((old/'FROZEN_BEFORE_TEST.json').read_text())
    if digest(old/'full/best.npz') != frozen['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    for path, sha in binding['ml_source_sha256'].items():
        if digest(path) != sha:
            raise ValueError(f'Original fitting code changed: {path}')
    plan = binding['plan']
    root = Path(plan['dataset'])
    lock = Path('output/ml_initial_20261023/continuation.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({'pid': os.getpid(), 'purpose': 'train_validation_runtime_replay', 'out': str(args.out)}).encode())
    os.close(fd)
    try:
        args.out.mkdir(parents=True, exist_ok=False)
        manifest = json.loads((root/'public/manifest.json').read_text())
        forbidden = {str((root/'protected/labels'/(r['source_id']+'.json')).resolve()).lower()
                     for r in manifest['records'] if r['split'] in ('test', 'challenge')}
        def guard(event, a):
            if event == 'open' and isinstance(a[0], (str, bytes)) and str(Path(str(a[0])).resolve()).lower() in forbidden:
                raise PermissionError('Runtime replay forbids test/challenge labels')
        sys.addaudithook(guard)
        from astropy.io import fits
        def forbid_decode(*a, **kw):
            raise PermissionError('Runtime replay forbids truth image decoding')
        fits.open = forbid_decode
        atomic_json(args.out/'binding.json', {'seed': plan['seed'], 'plan': plan,
                    'original_checkpoint_sha256': frozen['checkpoint_sha256'],
                    'benchmark_code_sha256': digest(Path(__file__)),
                    'original_code_sha256': binding['ml_source_sha256'], 'environment': versions(),
                    'threads': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                    'purpose': 'Instrumented exact replay on train/validation only; not new model selection or blind test'})
        start, cpu = time.perf_counter(), time.process_time()
        tick = time.perf_counter()
        transform = fit_training_transform(root)
        normalization_seconds = time.perf_counter()-tick
        original_transform = load_transform(plan['prepared'], root/'public')
        for k in ('feature_mean_jy', 'feature_scale_jy', 'target_mean', 'target_scale'):
            np.testing.assert_array_equal(getattr(transform, k), getattr(original_transform, k))
        save_transform(transform, args.out/'normalization')
        print(f'Train-only normalization fit: {normalization_seconds:.3f}s', flush=True)
        tick = time.perf_counter()
        x, y, books = load_split(root, transform, 'train')
        vx, vy, vbooks = load_split(root, transform, 'validation')
        loading_seconds = time.perf_counter()-tick
        if len(x) != 794 or len(vx) != 134 or {b['source_id'] for b in books} & {b['source_id'] for b in vbooks}:
            raise ValueError('Runtime replay split mismatch')
        config = json.loads(Path('configs/compact_proxy_gl4_v3.json').read_text())
        support = Support(config)
        tick = time.perf_counter()
        pca, weights, info = train('full', x, support.encode(y), vx, support.encode(vy), plan, args.out)
        fit_seconds = time.perf_counter()-tick
        differences = {}
        for path, current, prefix in [(old/'full/best.npz', weights, 'weights'), (old/'full/pca.npz', pca, 'pca')]:
            with np.load(path) as a:
                for k in a.files:
                    differences[prefix+'/'+k] = float(np.max(abs(current[k]-a[k])))
                    np.testing.assert_allclose(current[k], a[k], rtol=1e-10, atol=1e-10)
        if info['best_epoch'] != 22 or info['epochs'] != 52:
            raise ValueError('Runtime replay selection/stopping differs from archived fit')
        result = {'end_to_end_wall_seconds': time.perf_counter()-start,
                  'end_to_end_cpu_seconds': time.process_time()-cpu,
                  'train_only_normalization_fit_seconds': normalization_seconds,
                  'training_validation_verified_loading_seconds': loading_seconds,
                  'pca_and_optimizer_and_checkpoint_io_seconds': fit_seconds,
                  'optimizer_only_seconds': info['elapsed_seconds'],
                  'best_epoch': info['best_epoch'], 'epochs': info['epochs'],
                  'max_array_differences_from_frozen_fit': differences,
                  'frozen_checkpoint_preserved': digest(old/'full/best.npz') == frozen['checkpoint_sha256'],
                  'test_labels_opened': False, 'test_scored': False,
                  'interpretation': 'Measured replay cost on this CPU/two-thread environment, including integrity checks, repeated disk reads, normalization fit, PCA, optimizer, validation stopping and checkpoint I/O. Excludes initial dataset generation, readiness audit, half-data diagnostic and dependency import startup. Disk/cache/system load affect timing. Neural model used in comparison remains the original frozen checkpoint.'}
        atomic_json(args.out/'timing.json', result)
        (args.out/'REPORT.md').write_text(
            '# Instrumented training runtime replay\n\n'
            'This was an explicit replay using training and validation only, to fill the original missing end-to-end timing. '
            'It did not replace the frozen model or evaluate test/challenge observations.\n\n'
            f'- Train-only normalization fitting: {normalization_seconds:.3f} s.\n'
            f'- Verified training/validation loading: {loading_seconds:.3f} s.\n'
            f'- PCA, optimizer, validation stopping and checkpoint I/O: {fit_seconds:.3f} s.\n'
            f'- End-to-end measured replay: {result["end_to_end_wall_seconds"]:.3f} wall s; {result["end_to_end_cpu_seconds"]:.3f} CPU s.\n'
            f'- Same selected epoch 22 and stopping epoch 52; maximum saved-array difference {max(differences.values()):.3g}.\n\n'
            +result['interpretation']+'\n', encoding='utf-8')
        print(json.dumps(result), flush=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
