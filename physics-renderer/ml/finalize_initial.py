"""Verify and assemble completed local evaluation and runtime evidence.

No training, posterior sampling, test scoring or generation is performed.
The historical pilot cost is included because the new proposal uses its draws.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from hii_sim.io import atomic_json, digest


def read(path):
    return json.loads(Path(path).read_text())


def hashes(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def finalize(out, replay):
    binding = read(out/'binding.json')
    old = Path(binding['settings']['original_run'])
    if hashes(old) != binding['original_artifact_hashes']:
        raise ValueError('Original run archive changed')
    for path, sha in binding['code_hashes'].items():
        if digest(path) != sha:
            raise ValueError('Continuation source no longer matches executed binding')
    audit = read(out/'audit.json')
    if audit['inventory']['files_hashed'] != 3712 or not audit['normalization_exactly_reproduced_from_train_only']:
        raise ValueError('Inventory/preprocessing evidence incomplete')
    old_rows = read(old/'test_forward_checks.json')['cases']
    rows = read(out/'test_forward_checks.json')['cases']
    if rows[:46] != old_rows or len(rows) != 48 or len({r['source_id'] for r in rows}) != 48:
        raise ValueError('Forward continuation did not preserve original records/complete selection')
    cases = read(out/'bayesian_comparison.json')['cases']
    pilots = read(old/'bayesian_comparison.json')['cases']
    accounts = []
    for number, (case, pilot) in enumerate(zip(cases, pilots), 1):
        if case['realization_id'] != pilot['realization_id'] or case['source_id'] != pilot['source_id']:
            raise ValueError('Comparison observations changed')
        with np.load(out/f'bayes_{number:02d}_chains.npz') as a:
            if not np.isfinite(a['coordinates']).all() or a['coordinates'].shape[0] != 4:
                raise ValueError('Invalid saved chains')
        if case['converged']:
            for d in case['sampling']['diagnostics']['variables'].values():
                if d['rank_folded_split_rhat'] >= 1.01 or min(d['bulk_ess'], d['tail_ess']) < 400:
                    raise ValueError('Accepted comparator does not pass diagnostics')
            if len(case['posterior_predictive']['draws']) != 128:
                raise ValueError('Missing validated Bayesian predictive draws')
        elif case['bayesian_quantiles_025_50_975'] is not None:
            raise ValueError('Unconverged posterior was accepted')
        accounts.append({'source_id': case['source_id'], 'converged': case['converged'],
                         'historical_required_pilot_sampler_seconds': pilot['elapsed_seconds'],
                         'followup_sampler_seconds_including_new_initialization_warmup': case['sampling']['elapsed_seconds'],
                         'sum_pilot_plus_followup_sampler_seconds': pilot['elapsed_seconds']+case['sampling']['elapsed_seconds'],
                         'followup_total_seconds_including_setup_reference_checks_predictive': case['total_wall_seconds_including_setup_checks_and_predictive'],
                         'sum_original_pilot_workflow_plus_followup_workflow_seconds': pilot['total_wall_seconds']+case['total_wall_seconds_including_setup_checks_and_predictive'],
                         'neural': case['neural_runtime']})
    timing = read(replay/'timing.json')
    if not timing['frozen_checkpoint_preserved'] or timing['test_labels_opened'] or timing['test_scored']:
        raise ValueError('Runtime replay safeguards failed')
    account = {'cases': accounts, 'training_runtime_replay': timing,
               'interpretation': 'Follow-up proposal construction depends on the saved pilot, so its historical per-source cost belongs in a from-scratch comparison. Historical and follow-up times were measured in different sessions and system/cache conditions; sums describe recorded work, not a controlled repeated from-scratch benchmark. Neural training is an offline cost; per-observation inference is separate. Neural approximate independent draws and MCMC effective samples have different accuracy, so no numerical speedup or superiority claim is justified.',
               'limits': 'Two validation observations only; disk loading excluded from neural inference and separately explicit in training replay. Old end-to-end training cost was not recorded; the new instrumented replay measures the same training procedure, not a reconstructed historical duration.'}
    atomic_json(out/'runtime_accounting.json', account)
    report_path = out/'REPORT.md'
    text = report_path.read_text(encoding='utf-8')
    text = text.replace('No new blind evaluation, retraining, test-based selection, real-observation training or data generation occurred.',
                        'No new blind evaluation, test-based selection, real-observation training or data generation occurred. A separate instrumented train/validation-only replay reproduced the frozen fit without replacing it; see runtime accounting below.')
    text = text.replace('Half-data diagnostic and failed Bayesian pilots are separate development costs.',
                        'The half-data diagnostic is a separate development cost. The follow-up sampler uses its saved pilot draws to build a proposal, so the pilot cost must also be included in its from-scratch inference accounting; see the measured accounting below.')
    marker = '\n<!-- final-runtime-accounting -->\n'
    text = text.split(marker)[0]
    extra = ['## Additional precision and predictive scope notes', '',
             'Both algorithms start from the same stored observations. The verified neural loader emits normalized '
             'float32 inputs before float64 network/PCA arithmetic; the conventional likelihood promotes the stored '
             'measurement vector directly to float64. This finite-precision difference is additional to the explicitly '
             'documented PCA/posterior-family and finite-design approximations. No extra truth metadata enters either '
             'method. Neural forward profiles are not full neural posterior predictive draws; that check remains pending.', '',
             'The nuisance-profile optimizers and MAP initializations use calibration-coordinate bounds [-7,7] for numerical '
             'stability. Actual Bayesian sampling uses the unbounded standard-normal calibration priors. These bounded '
             'optimizations are diagnostics/initialization only and are never substituted for posterior inference.', '',
             '## Measured training replay and complete inference accounting', '',
             'An explicitly instrumented train/validation-only replay reproduced the frozen fit; it did not replace that model '
             'or reopen the test evaluation. The replay timing includes train-only normalization fitting, verified repeated '
             'data reads, PCA, optimization, validation stopping and checkpoint I/O. Startup imports and initial dataset '
             'generation/readiness are excluded. Cache and system load affect the I/O time.', '',
             f'- End-to-end replay: {timing["end_to_end_wall_seconds"]:.3f} wall s; {timing["end_to_end_cpu_seconds"]:.3f} CPU s.',
             f'- Training-only normalization fit: {timing["train_only_normalization_fit_seconds"]:.3f} s.',
             f'- Training/validation verified loading: {timing["training_validation_verified_loading_seconds"]:.3f} s.',
             f'- PCA + optimizer + checkpoint I/O: {timing["pca_and_optimizer_and_checkpoint_io_seconds"]:.3f} s.',
             f'- Selected/stopped epochs: {timing["best_epoch"]}/{timing["epochs"]}; maximum array difference from original: {max(timing["max_array_differences_from_frozen_fit"].values()):.3g}.', '',
             '| Validation case | Required historical pilot sampler [s] | Follow-up sampler [s] | Sum of recorded sampler work [s] | Neural median + 2048 draws [s] |',
             '|---|---:|---:|---:|---:|']
    for number, row in enumerate(accounts, 1):
        extra.append(f'| {number} | {row["historical_required_pilot_sampler_seconds"]:.3f} | '
                     f'{row["followup_sampler_seconds_including_new_initialization_warmup"]:.3f} | '
                     f'{row["sum_pilot_plus_followup_sampler_seconds"]:.3f} | '
                     f'{row["neural"]["median_posterior_inference_seconds"]:.6g} |')
    extra += ['', account['interpretation'], '', 'Full timing scopes and per-case setup/reference/predictive costs are in '
              '`runtime_accounting.json`. The training replay arrays, epoch log, configuration/code hashes and timing '
              'are in `training_runtime_replay/`. These measurements provide cost accounting, not a matched-accuracy speed claim.']
    report_path.write_text(text+marker+'\n'.join(extra)+'\n', encoding='utf-8')
    summary = read(out/'summary.json')
    summary.pop('neural_retrained', None)
    summary['training_replay_performed'] = True
    summary['comparison_model_retrained_or_replaced'] = False
    summary['training_replay'] = timing
    summary['runtime_accounting'] = 'runtime_accounting.json'
    summary['completed'] = ['Immutable inventory/bindings and training-only normalization/PCA verified',
                            'Original neural training/test archive verified, model retained',
                            '134 validation observations evaluated without new test scoring',
                            '48/48 neural median nuisance profiles saved; original 46 preserved',
                            f'{sum(c["converged"] for c in cases)}/{len(cases)} Bayesian validation cases pass all target/nuisance gates',
                            'Training replay, inference timings, curves, coverage/sharpness and local report saved']
    summary['pending'] = ['Full neural posterior predictive treatment of nuisance correlations',
                          'Larger independently source-based Bayesian comparison and calibration evidence',
                          'Future protected independently generated evaluation set if later model development requires a new blind test; proposal documented, no generation authorized']
    summary['blockers'] = []
    if len(cases) != 2 or not all(c['converged'] for c in cases):
        summary['pending'].append('Validated Bayesian comparison for every preselected case; failed/incomplete cases remain inconclusive')
    atomic_json(out/'summary.json', summary)
    expected_figures = ['training_curves.png', 'archived_test_coverage.png'] + [f'bayes_{i:02d}_traces.png' for i in range(1, len(cases)+1)]
    if not all((out/p).stat().st_size > 1000 for p in expected_figures):
        raise ValueError('Missing rendered figures')
    atomic_json(out/'final_verification.json', {'utc': datetime.now(timezone.utc).isoformat(),
                'original_archive_unchanged': True, 'original_forward_rows_preserved': True,
                'continuation_source_binding_unchanged': True, 'same_comparison_observations': True,
                'runtime_replay_reproduces_original_fit': True,
                'finalizer_code_sha256': digest(Path(__file__)), 'new_test_scoring': False,
                'summary_sha256': digest(out/'summary.json'), 'report_sha256': digest(report_path),
                'figure_hashes': {p: digest(out/p) for p in expected_figures}})
    print(json.dumps({'completed': summary['completed'], 'pending': summary['pending'],
                      'blockers': summary['blockers'], 'report': str(report_path)}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--replay', type=Path)
    args = ap.parse_args()
    finalize(args.out, args.replay or args.out/'training_runtime_replay')
