"""Evaluation-only sampler improvement; original likelihood and runs preserved."""
import json
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares
from bayes import PhysicsLikelihood
from diagnostics import diagnostic_report
from hii_sim.io import atomic_json, write_npz

NAMES = ['em0_pc_cm6', 'te_k', 'major_fwhm_arcsec', 'axis_ratio',
         'pa_branch', 'east_prior_unit', 'north_prior_unit', 'common_gain_z', 'slope_z']


class GroupedLikelihood(PhysicsLikelihood):
    """Identical forward calculation, batched thermal matrix multiplication."""
    def __init__(self, inputs, config):
        super().__init__(inputs, config)
        self.buckets = []
        for n in sorted({len(i) for i, _ in self.groups}):
            group = [(i, a) for i, a in self.groups if len(i) == n]
            self.buckets.append((np.stack([i for i, _ in group]), np.stack([a for _, a in group])))

    def whiten(self, delta):
        out = np.empty_like(delta)
        for idx, inverse in self.buckets:
            out[idx] = np.einsum('bij,bj->bi', inverse, delta[idx])
        return out


def checked_diagnostics(chains, like):
    original = chains.copy()
    for j, name in enumerate(NAMES[:4]):
        lo, hi, kind = like.config['sampling']['ranges'][name]
        original[:, :, j] = (lo*(hi/lo)**chains[:, :, j] if kind == 'log'
                              else lo+(hi-lo)*chains[:, :, j])
    angles = 2*np.pi*chains[:, :, 4]
    augmented = np.concatenate((original, np.sin(angles)[..., None], np.cos(angles)[..., None]), axis=2)
    return diagnostic_report(augmented, NAMES+['pa_sin', 'pa_cos'])


def sample_followup(like, pilot, settings, seed, out, tag):
    """Four independent fixed-kernel chains; pilot defines proposals, not target.

    Mixture of independent multivariate-t moves (with Hastings density ratio)
    and symmetric local Gaussian moves. PA uses one full period centered on
    the pilot circular mean, with zero density outside that branch. No wrapped
    correlated proposal density is silently substituted.
    """
    started, cpu = time.perf_counter(), time.process_time()
    pilot = pilot.copy()
    angle = np.angle(np.exp(2j*np.pi*pilot[:, :, 4]).mean())/(2*np.pi)
    pilot[:, :, 4] = angle + (pilot[:, :, 4]-angle+.5) % 1-.5
    flat = pilot.reshape(-1, 9)
    center = flat.mean(axis=0)
    cov = np.cov(flat.T)
    vals, vectors = np.linalg.eigh(cov)
    chol = cholesky((vectors*np.maximum(vals, 1e-9)) @ vectors.T, lower=True)
    # Proposal is fixed before new production draws, including heavy tails.
    independent_chol = chol*np.sqrt(settings['proposal_covariance_inflation'])
    inv_independent = solve_triangular(independent_chol, np.eye(9), lower=True)
    df = settings['student_df']
    def logq(t):
        v = inv_independent @ (t-center)
        return -.5*(df+9)*np.log1p(v@v/df)  # common normalizer cancels
    def logp(t):
        if np.any(t[[0, 1, 2, 3, 5, 6]] < 0) or np.any(t[[0, 1, 2, 3, 5, 6]] > 1) or abs(t[4]-angle) > .5:
            return -np.inf
        physical = t.copy()
        physical[4] %= 1
        return like.logp(physical)
    rngs = [np.random.default_rng(seed+c*1009) for c in range(4)]
    # Broaden starting regions using measurement-only optimizations. These are
    # initialization diagnostics, never posterior substitutes or neural labels.
    maps = []
    for c, rng in enumerate(rngs):
        start = np.r_[rng.uniform(.1, .9, 4), (c+.5)/4, .5, .5, 0., 0.]
        fit = least_squares(like.residual, start,
                            bounds=(np.r_[np.full(7, 1e-7), -7., -7.],
                                    np.r_[np.full(7, 1-1e-7), 7., 7.]),
                            max_nfev=settings['map_max_nfev'], diff_step=1e-4)
        t = fit.x.copy()
        t[4] = angle+(t[4]-angle+.5) % 1-.5
        maps.append({'cost': float(fit.cost), 'success': bool(fit.success), 'coordinates': t.tolist()})
    states = np.array([r['coordinates'] for r in maps])
    lps = np.array([logp(t) for t in states])
    warm = settings['warmup']
    retained = []
    accepts = np.zeros((4, 2), int)
    attempts = np.zeros((4, 2), int)
    initial_costs = [r['cost'] for r in maps]
    atomic_json(out/f'{tag}_initialization.json', {'seed': seed, 'maps': maps,
                'pilot_proposal_center': center.tolist(), 'pilot_proposal_covariance': cov.tolist(),
                'pa_branch_center': float(angle), 'cost_spread': max(initial_costs)-min(initial_costs)})
    diag = None
    for step in range(warm+settings['max_draws_per_chain']):
        for c, rng in enumerate(rngs):
            kind = int(rng.uniform() >= settings['independence_probability'])
            if kind == 0:
                proposal = center + independent_chol @ rng.normal(size=9)/np.sqrt(rng.chisquare(df)/df)
                correction = logq(states[c])-logq(proposal)
            else:
                proposal = states[c] + settings['local_scale']*chol @ rng.normal(size=9)
                correction = 0.
            lp = logp(proposal)
            attempts[c, kind] += 1
            if np.log(rng.uniform()) < lp-lps[c]+correction:
                states[c], lps[c] = proposal, lp
                accepts[c, kind] += 1
        if step >= warm:
            retained.append(states.copy())
        n = len(retained)
        if n and n % settings['diagnostic_every'] == 0:
            chains = np.array(retained).transpose(1, 0, 2)
            diag = checked_diagnostics(chains, like)
            write_npz(out/f'{tag}_chains.npz', coordinates=chains, states=states,
                      proposal_center=center, proposal_cholesky=chol)
            status = {'draws_per_chain': n, 'steps_including_warmup': step+1,
                      'elapsed_seconds': time.perf_counter()-started,
                      'cpu_seconds': time.process_time()-cpu,
                      'acceptance_by_chain_independent_local': (accepts/np.maximum(attempts, 1)).tolist(),
                      'diagnostics': diag, 'seed': seed,
                      'rng_states': [r.bit_generator.state for r in rngs]}
            atomic_json(out/f'{tag}_sampling.json', status)
            print(json.dumps({'case': tag, 'draws_per_chain': n, 'elapsed_seconds': status['elapsed_seconds'],
                              'max_rhat': max(r['rank_folded_split_rhat'] for r in diag['variables'].values()),
                              'min_bulk_ess': min(r['bulk_ess'] for r in diag['variables'].values()),
                              'min_tail_ess': min(r['tail_ess'] for r in diag['variables'].values()),
                              'passed': diag['passed']}), flush=True)
            if diag['passed'] or status['elapsed_seconds'] > settings['wall_seconds_per_source']:
                break
    return chains, status
