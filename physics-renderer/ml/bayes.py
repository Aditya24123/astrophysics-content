"""Conventional explicit-physics posterior; no protected truths are inputs.

Coordinates: four prior-unit targets, PA/180, (east+10)/20,
(north+10)/20, and two standard-normal calibration nuisances.
"""
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares
from astropy import units as u
from hii_sim.physics import optical_depth, RJ_JY_SR, render
from hii_sim.schema import Grid, Sky, Source


class PhysicsLikelihood:
    def __init__(self, inputs, config):
        self.m = inputs['features']['measurements']
        self.o = inputs['features']['observing']
        self.config = config
        self.y = self.m.values_jy.astype(float)
        self.grid = Grid(**config['grid'])
        f, b = self.o.frequencies_ghz, self.o.bandwidths_mhz
        nodes, weights = np.polynomial.legendre.leggauss(config['instrument']['frequency_quadrature'])
        self.nu = f[:, None]+nodes[None, :]*b[:, None]/2000
        self.freq_weights = weights/2
        nodes, weights = np.polynomial.legendre.leggauss(self.grid.subpixels)
        offsets = nodes*self.grid.pixel_arcsec/2
        east, north = self.grid.coordinates()
        self.east = np.stack([east+dx for dy in offsets for dx in offsets])
        self.north = np.stack([north+dy for dy in offsets for dx in offsets])
        self.spatial_weights = np.outer(weights, weights).ravel()/4
        self.logf = np.log(f)-np.log(f).mean()
        self.s = np.sqrt(np.log1p(self.o.calibration.fractional_common_amplitude_std**2))
        self.groups = []
        keys = np.stack((self.m.y_index, self.m.x_index, self.m.component), axis=1)
        for key in np.unique(keys, axis=0):
            idx = np.where(np.all(keys == key, axis=1))[0]
            cov = self.o.thermal.covariance(self.m.frequency_index[idx],
                                          self_conjugate=bool(self.m.self_conjugate[idx[0]]))
            self.groups.append((idx, np.linalg.inv(cholesky(cov, lower=True))))
        self.white = np.zeros((len(self.y), len(self.y))) if False else None

    def unpack(self, t):
        r = self.config['sampling']['ranges']
        targets = []
        for j, name in enumerate(('em0_pc_cm6', 'te_k', 'major_fwhm_arcsec', 'axis_ratio')):
            lo, hi, kind = r[name]
            targets.append(10**(np.log10(lo)+t[j]*np.log10(hi/lo)) if kind == 'log' else lo+t[j]*(hi-lo))
        return np.array(targets), 180*t[4], -10+20*t[5], -10+20*t[6]

    def vector(self, t, reference=False):
        targets, pa, east, north = self.unpack(t)
        if reference:
            cube = render(Source(*targets, pa), self.grid, Sky(east_arcsec=east, north_arcsec=north),
                          self.o.frequencies_ghz, self.o.bandwidths_mhz,
                          self.config['instrument']['frequency_quadrature'])
        else:
            angle = np.deg2rad(pa)
            e, n = self.east-east, self.north-north
            major = n*np.cos(angle)+e*np.sin(angle)
            minor = e*np.cos(angle)-n*np.sin(angle)
            g = np.exp(-4*np.log(2)*((major/targets[2])**2+(minor/(targets[2]*targets[3]))**2))
            tau = optical_depth(targets[0]*u.pc/u.cm**6, targets[1]*u.K, self.nu*u.GHz)
            # Taylor evaluation of the SAME nonlinear transfer, not a thin approximation.
            # At maximum prior tau < 0.4, 14 terms have a negligible remainder.
            coeff = np.ones_like(tau)
            power = np.ones_like(g)
            cube = np.zeros_like(self.o.primary_beam, dtype=float)
            for order in range(1, 15):
                power *= g
                coeff *= -tau/order
                spatial = np.einsum('i,ijk->jk', self.spatial_weights, power)
                spectral = np.sum(-coeff*RJ_JY_SR*(self.nu*1e9)**2*targets[1]*self.freq_weights, axis=1)
                cube += spectral[:, None, None]*spatial
        apparent = cube*self.o.primary_beam*self.grid.omega_sr
        vis = np.fft.fft2(np.fft.ifftshift(apparent, axes=(-2, -1)), axes=(-2, -1))
        gains = np.exp(-self.s**2/2+self.s*t[7]+self.o.calibration.spectral_log_slope_std*t[8]*self.logf)
        value = vis[self.m.frequency_index, self.m.y_index, self.m.x_index]*gains[self.m.frequency_index]
        return np.where(self.m.component == 0, value.real, value.imag)

    def whiten(self, delta):
        out = np.empty_like(delta)
        for idx, inverse in self.groups:
            out[idx] = inverse @ delta[idx]
        return out

    def residual(self, t):
        return np.r_[self.whiten(self.vector(t)-self.y), t[7:9]]

    def logp(self, t):
        if np.any(t[:7] < 0) or np.any(t[:7] > 1):
            return -np.inf
        r = self.residual(t)
        return -.5*float(r @ r)


def diagnostics(chains):
    """Split Rhat and initial-positive autocorrelation ESS (not rank normalized)."""
    n = chains.shape[1]//2
    split = np.concatenate((chains[:, :n], chains[:, -n:]), axis=0)
    means = split.mean(axis=1)
    w = split.var(axis=1, ddof=1).mean(axis=0)
    b = n*means.var(axis=0, ddof=1)
    var = (n-1)/n*w+b/n
    rhat = np.sqrt(var/np.maximum(w, 1e-30))
    ess = []
    centered = split-means[:, None]
    for j in range(chains.shape[2]):
        ac = []
        for lag in range(1, min(n-1, 300), 2):
            pair = 0.
            for ell in (lag, lag+1):
                autocov = np.mean(np.sum(centered[:, :-ell, j]*centered[:, ell:, j], axis=1)/(n-ell))
                pair += 1-(w[j]-autocov)/max(var[j], 1e-30)
            if pair < 0:
                break
            ac.append(pair)
        ess.append(min(split.shape[0]*n, split.shape[0]*n/max(1., 1+2*sum(ac))))
    return rhat, np.array(ess)


def fit_posterior(like, plan, seed):
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    lower = np.r_[np.full(7, 1e-7), [-7., -7.]]
    upper = np.r_[np.full(7, 1-1e-7), [7., 7.]]
    fits = []
    for pa in (.2, .7):
        initial = np.r_[[.5]*4, pa, .5, .5, 0., 0.]
        fit = least_squares(like.residual, initial, bounds=(lower, upper),
                            max_nfev=plan['baseline_map_max_nfev'], diff_step=1e-4,
                            ftol=1e-6, xtol=1e-6, gtol=1e-5)
        fits.append(fit)
    best = min(fits, key=lambda r: r.cost)
    h = best.jac.T @ best.jac
    vals, vecs = np.linalg.eigh(h)
    # Regularization only shapes the proposal; the Metropolis target is unchanged.
    chol = vecs @ np.diag(1/np.sqrt(np.maximum(vals, 1.)))
    states, logps = [], []
    for c in range(plan['baseline_chains']):
        state = best.x.copy()
        state[:4] = np.clip(state[:4]+rng.normal(0, .03, 4), .001, .999)
        states.append(state)
        logps.append(like.logp(state))
    states, logps = np.array(states), np.array(logps)
    scale = np.full(len(states), 2.38/np.sqrt(9))
    accepted = np.zeros(len(states), int)
    window = accepted.copy()
    samples = []
    warm = plan['baseline_warmup']
    total = warm+plan['baseline_draws_per_chain']
    completed = 0
    for step in range(total):
        for c in range(len(states)):
            proposal = states[c]+scale[c]*(chol @ rng.normal(size=9))
            # PA is periodic; wrap without affecting the uniform prior.
            proposal[4] %= 1
            lp = like.logp(proposal)
            if np.log(rng.uniform()) < lp-logps[c]:
                states[c], logps[c] = proposal, lp
                accepted[c] += 1
                window[c] += 1
        if step < warm and (step+1) % 50 == 0:
            scale *= np.exp(np.clip(window/50-.25, -.2, .2))
            window[:] = 0
        if step >= warm:
            samples.append(states.copy())
        completed = step+1
        if (step+1) % 100 == 0:
            print(f'bayes step {step+1}/{total}, elapsed {time.perf_counter()-started:.1f}s', flush=True)
        if time.perf_counter()-started > plan['baseline_wall_seconds_per_source']:
            break
    chains = np.array(samples).transpose(1, 0, 2) if samples else np.empty((len(states), 0, 9))
    if chains.shape[1] >= 10:
        rhat, ess = diagnostics(chains)
    else:
        rhat, ess = np.full(9, 1e10), np.zeros(9)
    converged = bool(np.all(rhat[:4] < 1.05) and np.all(ess[:4] > 100))
    report = {'elapsed_seconds': time.perf_counter()-started, 'steps': completed,
              'map_starts': [{'cost': float(f.cost), 'success': bool(f.success), 'nfev': int(f.nfev)} for f in fits],
              'map_coordinates': best.x.tolist(), 'proposal_eigenvalues': vals.tolist(),
              'acceptance_rate': (accepted/max(completed, 1)).tolist(), 'proposal_scale': scale.tolist(),
              'split_rhat': rhat.tolist(), 'approx_ess': ess.tolist(),
              'converged_targets_by_predeclared_gate': converged,
              'diagnostic_limit': 'ordinary split Rhat and initial-positive ESS; shared MAP-region starts can miss modes, short chains are only a pilot',
              'mcmc_target': 'exact bounded prior and correlated thermal likelihood, explicit Gaussian calibration nuisances; Taylor forward checked against existing renderer'}
    return chains, report
