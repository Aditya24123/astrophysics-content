"""Render inspectable static scientific figures and a local experiment record."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hii_sim.io import atomic_json, digest

NAMES = ['em0_pc_cm6', 'te_k', 'major_fwhm_arcsec', 'axis_ratio']
LABELS = ['EM0 [pc cm$^{-6}$]', 'Temperature [K]', 'Major EM FWHM [arcsec]', 'Axis ratio [1]']


def read(path):
    return json.loads(Path(path).read_text())


def build_report(old, out):
    audit = read(out/'audit.json')
    test = read(old/'test_evaluation.json')
    validation = read(out/'validation/metrics.json')
    forward = read(out/'test_forward_checks.json') if (out/'test_forward_checks.json').exists() else read(old/'test_forward_checks.json')
    bayes = read(out/'bayesian_comparison.json') if (out/'bayesian_comparison.json').exists() else {'cases': []}
    curves = [[json.loads(line) for line in (old/d/'epochs.jsonl').read_text().splitlines()]
              for d in ('full', 'half_diagnostic')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, rows, title in zip(axes, curves, ['794 observations / 397 training sources', '396 observations / 198 training sources']):
        for key, label in [('train_nll', 'Train'), ('validation_nll', 'Validation')]:
            ax.plot([r['epoch'] for r in rows], [r[key] for r in rows], label=label)
        best = read(old/('full' if ax is axes[0] else 'half_diagnostic')/'training.json')['best_epoch']
        ax.axvline(best, color='black', linestyle=':', label=f'Selected epoch {best}')
        ax.set(title=title, xlabel='Epoch', ylabel='NLL [nats / scaled logit coordinates]')
        ax.legend()
    fig.suptitle('Archived training curves — validation selection; frozen model retained')
    fig.savefig(out/'training_curves.png', dpi=180)
    fig.savefig(out/'training_curves.svg')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        row = test['parameters'][NAMES[j]]
        levels = np.array([.5, .8, .95])
        observed = np.array([row['coverage'][str(v)]['observed'] for v in levels])
        ci = np.array([row['coverage'][str(v)]['parent_bootstrap_95ci'] for v in levels])
        ax.plot([0, 1], [0, 1], ':', color='gray')
        ax.errorbar(levels, observed, yerr=np.array([observed-ci[:, 0], ci[:, 1]-observed]), fmt='o-', capsize=4)
        ax.set(title=LABELS[j], xlabel='Nominal central interval probability', ylabel='Empirical coverage',
               xlim=(.4, 1), ylim=(.25, 1.03))
    fig.suptitle('Existing test archive: 96 observations / 48 sources\n95% source-bootstrap intervals; not a fresh blind evaluation')
    fig.savefig(out/'archived_test_coverage.png', dpi=180)
    fig.savefig(out/'archived_test_coverage.svg')
    plt.close(fig)
    # Trace evidence includes all nuisances; no cherry-picking targets alone.
    for number, case in enumerate(bayes['cases'], 1):
        with np.load(out/f'bayes_{number:02d}_chains.npz') as a:
            chain = a['coordinates']
        fig, axes = plt.subplots(3, 3, figsize=(12, 8), constrained_layout=True)
        for j, ax in enumerate(axes.flat):
            for c in range(4):
                ax.plot(chain[c, ::max(1, chain.shape[1]//1000), j], lw=.5, alpha=.6)
            ax.set(title=['EM prior unit', 'Te prior unit', 'Size prior unit', 'q prior unit', 'PA branch', 'East unit', 'North unit', 'Common gain z', 'Slope z'][j], xlabel='Retained draw (downsampled display)')
        fig.suptitle(f'Validation Bayesian case {number}; convergence gate passed={case["converged"]}')
        fig.savefig(out/f'bayes_{number:02d}_traces.png', dpi=160)
        plt.close(fig)
    training = read(old/'full/training.json')
    profiles = forward['cases']
    rms = np.array([r['profiled_whitened_residual_rms_sigma'] for r in profiles])
    summary = {'original_run': str(old), 'continuation': str(out),
               'neural_checkpoint_sha256': digest(old/'full/best.npz'),
               'neural_retrained': False, 'new_test_predictions': False,
               'existing_test_observations': 96, 'existing_test_independent_sources': 48,
               'forward_profiles': len(profiles), 'forward_optimizer_failures': sum(not r['optimizer_success'] for r in profiles),
               'forward_profile_rms_median': float(np.median(rms)), 'forward_profile_rms_max': float(rms.max()),
               'bayesian_converged_cases': sum(r['converged'] for r in bayes['cases']),
               'bayesian_cases': len(bayes['cases']),
               'neural_optimizer_training_seconds': training['elapsed_seconds'],
               'training_runtime_limit': 'Original training timer excludes PCA and data loading; do not call it end-to-end training cost.',
               'pca_verification_refit_seconds': audit['pca_verification_refit_wall_seconds'],
               'decision': 'Initial synthetic experiment, no accuracy threshold or real-nebula readiness claim; two validation comparators cannot establish population superiority.'}
    atomic_json(out/'summary.json', summary)
    lines = ['# Frozen initial H II posterior experiment — continuation', '',
        'The neural model was retained after verifying its saved checkpoint, selected epoch 22 and stopping epoch 52. '
        'The original test evaluation is an archive of 96 observations from 48 source groups. No new blind evaluation, retraining, '
        'test-based selection, real-observation training or data generation occurred.', '',
        '## Data, targets and fairness', '',
        'The complete immutable 3,712-file inventory passed again. Train-only normalization was reproduced exactly, and PCA '
        'was reproduced from the 794 training observations. Splits remain 397/67/48 independent training/validation/test '
        'sources (794/134/96 observations); 24 challenge observations remain separate. Predictor inputs contain only 3,906 '
        'normalized measured Fourier components. True targets stay separate in the supervised loss.', '',
        '| Target | Units | Prior | Transformation |', '|---|---|---|---|',
        '| em0_pc_cm6 | pc cm^-6 | Log-uniform [10000,500000] | log10 |',
        '| te_k | K | Uniform [5000,12000] | Linear |',
        '| major_fwhm_arcsec | arcsec, full major-axis EM FWHM | Log-uniform [12,60] | log10 |',
        '| axis_ratio | 1 | Uniform [0.5,1] | Linear |', '',
        'The verified loader standardizes these transformed labels using unique training parents. The frozen neural loss '
        'instead uses original labels mapped into prior-unit coordinates and then logit/(pi/sqrt(3)). This is a fixed bounded '
        'transform, not a holdout-fitted normalization. The original labels and simulator were preserved.', '',
        'Both methods use the same raw observations, target definitions and continuous prior bounds, uniform PA [0,180), '
        'uniform east/north offsets [-10,10] arcsec, mean-one 5% common lognormal calibration and Gaussian spectral slope '
        'sigma 0.01, and the same correlated thermal covariance in Jy^2. Conventional inference samples all nine coordinates. '
        'Neural inference approximates the four-target nuisance marginal through the simulator training distribution; its '
        'train-only PCA retains only 39.22% of standardized training variance and can discard information. Its single Gaussian '
        'in logit coordinates restricts posterior shape. The training bank uses a finite scrambled Sobol design rather than iid '
        'continuous prior draws. These computational approximations prevent calling the two methods mathematically identical.', '',
        '## Existing test archive (not a new blind evaluation)', '',
        '| Target | MAE | RMSE | Median abs relative error | MAE source-bootstrap 95% CI |', '|---|---:|---:|---:|---|']
    for n in NAMES:
        r = test['parameters'][n]
        ci = r['mae_parent_bootstrap_95ci']
        lines.append(f'| {n} | {r["mae_original_units"]:.6g} | {r["rmse_original_units"]:.6g} | {100*r["absolute_relative_error"]["median"]:.2f}% | [{ci[0]:.6g}, {ci[1]:.6g}] |')
    lines += ['', 'MAE/RMSE and interval widths use the units above. Temperature MAE is worse than the prior-midpoint '
              'diagnostic (1486.9 K). This is descriptive evidence, not a basis for subsequent test tuning.', '',
              '| Target | Nominal level | Coverage | Source-bootstrap 95% CI | Mean interval width |', '|---|---:|---:|---|---:|']
    for n in NAMES:
        for level, r in test['parameters'][n]['coverage'].items():
            lo, hi = r['parent_bootstrap_95ci']
            lines.append(f'| {n} | {float(level):.0%} | {r["observed"]:.2%} | [{lo:.2%}, {hi:.2%}] | {r["mean_interval_width_original_units"]:.6g} |')
    lines += ['', 'The 1,000 bootstrap resamples use source groups; the two noise realizations are not independent skies. '
              'Coverage has only 48 independent test groups, and bootstrap intervals at 0 or 1 are unreliable near boundaries. '
              '2,048 neural draws per observation add finite Monte Carlo uncertainty. Broad/overcovering intervals are not '
              'proof of calibration. PIT histograms, joint coverage, log10 errors (EM/size), and full metric definitions remain '
              'in the original test JSON. Validation metrics are in `validation/metrics.json`; they are for development, not blind claims.', '',
              '![Training curves](training_curves.png)', '', '![Archived coverage](archived_test_coverage.png)', '',
              '## Bayesian follow-up and predictive checks', '',
              'Both original 500-warmup/1,000-retained-draw pilots failed their original target gate. Slow mixing also affects '
              'nuisances, and shared MAP-region initialization gave weak mode exploration. Their posterior errors and predictive '
              'summaries remain provisional. The follow-up changes the sampler proposal only, using the pilot covariance for '
              'a fixed Student-t independence proposal with an exact Hastings correction plus symmetric local moves. Four '
              'independent chains start at measurement-only optimizations from dispersed physical/PA starts. No dependency '
              'was added. Batched thermal whitening is algebraically equivalent and regression-tested. Forward renders are '
              'checked against the original renderer at prior extrema/random points and saved posterior states.', '',
              'The new gate requires rank/folded split Rhat <1.01 and bulk/tail ESS >=400 for all targets, nuisances and PA '
              'sine/cosine. This is stricter than the historical pilot gate and is recorded before new sampling. '
              '[Diagnostic method](https://arxiv.org/abs/1903.08008); '
              '[Stan guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html). Finite diagnostics cannot prove global mode discovery.', '']
    for number, case in enumerate(bayes['cases'], 1):
        stats = case['sampling']
        variables = stats['diagnostics']['variables']
        lines += [f'### Validation case {number}: `{case["source_id"]}`', '',
                  f'Convergence gate: **{"PASS" if case["converged"] else "FAIL / INCONCLUSIVE"}**. '
                  f'{stats["draws_per_chain"]} retained draws per chain; wall {stats["elapsed_seconds"]:.2f} s; '
                  f'CPU {stats["cpu_seconds"]:.2f} s. Maximum checked reference-render discrepancy '
                  f'{case["reference_check_whitened_l2_max"]:.3g} in whitened L2 units.', '',
                  '| Variable | Rank/folded Rhat | Bulk ESS | Tail ESS | Mean MCSE |', '|---|---:|---:|---:|---:|']
        for n, r in variables.items():
            lines.append(f'| {n} | {r["rank_folded_split_rhat"]:.5f} | {r["bulk_ess"]:.1f} | {r["tail_ess"]:.1f} | {r["mean_mcse"]:.4g} |')
        b = case['neural_runtime']
        lines += ['', f'Neural median inference: {b["median_point_inference_seconds"]:.6g} s for the marginal median; '
                  f'{b["median_posterior_inference_seconds"]:.6g} s including {b["posterior_draws"]} draws. '
                  'Both include normalization and PCA, exclude observation disk loading, use the same CPU/two-thread environment, '
                  'and repeat 30 times after warmup. Neural draw count is not MCMC ESS; no matched-precision speedup is asserted.', '']
        if case['converged']:
            lines += ['| Target | Bayesian median [95% interval] | Neural median [95% interval] | Bayes abs error | Neural abs error |', '|---|---|---|---:|---:|']
            bq, nq = case['bayesian_quantiles_025_50_975'], case['neural_quantiles_025_50_975']
            for j, n in enumerate(NAMES):
                lines.append(f'| {n} | {bq[1][j]:.6g} [{bq[0][j]:.6g}, {bq[2][j]:.6g}] | '
                             f'{nq[1][j]:.6g} [{nq[0][j]:.6g}, {nq[2][j]:.6g}] | '
                             f'{case["bayesian_absolute_error_original_units"][j]:.6g} | {case["neural_absolute_error_original_units"][j]:.6g} |')
            lines += ['', f'Full-joint Bayesian posterior predictive discrepancy tail fraction: {case["posterior_predictive"]["tail_fraction"]:.4f} '
                      f'from 128 selections. This checks replicated thermal residual energy relative to the same sampled expected '
                      'signal; it is not the old sqrt(2)-scale replicate-minus-data RMS presented as a unit-noise test. '
                      'Detailed draws and interval inclusion/widths are in the case JSON. No coverage estimate is justified by two validation sources.', '']
        else:
            lines += ['Posterior quantiles/error comparisons and predictive conclusions are withheld because convergence failed.', '']
        lines += [f'![All-coordinate chain traces](bayes_{number:02d}_traces.png)', '']
    lines += [f'Completed forward profiles: **{len(profiles)}/48**, retaining the 46 original records and computing only the two '
              f'missing cases. {summary["forward_optimizer_failures"]} optimizer statuses were unsuccessful; all residual values '
              f'are retained. Median/max profiled whitened RMS: {np.median(rms):.4g}/{rms.max():.4g}. Targets stay at saved neural '
              'marginal medians while nuisances are fitted to the same data. These are optimistic profile residuals, not neural '
              'posterior predictive draws. A full neural nuisance-correlated predictive check remains unimplemented.', '',
              '## Runtime and limits', '',
              f'Original full neural optimizer loop: {training["elapsed_seconds"]:.6g} wall s, {training["cpu_seconds"]:.6g} CPU s. '
              f'That timer EXCLUDED PCA and data loading; it is not the total training cost. The present training-only PCA '
              f'verification refit took {audit["pca_verification_refit_wall_seconds"]:.6g} wall s. The original end-to-end elapsed '
              'time was never saved because the run was interrupted, so it cannot be reconstructed exactly. Half-data diagnostic '
              'and failed Bayesian pilots are separate development costs. Per-case follow-up timings include initialization, '
              'warmup and retained sampling; full setup/reference-check/predictive timing is saved separately. Inference comparisons '
              'use exactly the same validation measurements; two sources do not establish general runtime or accuracy superiority.', '',
              '## Pending scientific work and protected next evaluation', '',
              'A full neural posterior predictive treatment of nuisance correlations, broader validated conventional comparison, '
              'and stronger source-level calibration evidence remain pending. No real-nebula readiness or predetermined '
              'accuracy threshold is claimed. The small smooth pure-hydrogen/isothermal Gaussian source family and Fourier '
              'instrument proxy do not establish telescope or morphology transfer.', '',
              'If a later model is developed using train/validation, propose a separately generated final evaluation bank with '
              'an independent private design seed, independent source IDs and all realizations grouped; freeze sample size, '
              'prior/instrument/noise law, metrics, code and model hashes first. An evaluator should protect its labels until '
              'one final unsealing. Choose sample size from a source-level coverage precision analysis (e.g. roughly 300–500 '
              'independent sources for useful few-percent marginal coverage precision, with multiple-target and design caveats). '
              'This is a proposal only: no such data generation is authorized or performed.', '',
              '## Reproduction and artifact provenance', '',
              '`binding.json` records original archive hashes, continuation source/plan hashes, environment and thread settings. '
              '`audit.json` records source-group, normalization and inventory verification. Original `run_03/run_binding.json` '
              'records Python/NumPy/SciPy/Astropy/Matplotlib versions, seed 231026 and original code hashes. Follow-up seed '
              '231027 and chain offsets, fixed sampler settings, RNG end states, arrays and diagnostics are saved locally. '
              'See the root README and `ml/README.md` for executable commands.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Report and figures saved to {out}', flush=True)
