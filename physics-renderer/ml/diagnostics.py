"""Rank/folded split Rhat and FFT Geyer bulk/tail ESS, without new dependencies.

Method: Vehtari et al. (2021), https://arxiv.org/abs/1903.08008;
Stan posterior R/convergence.R. Arrays are [independent chain, draw, variable].
ESS is capped at the actual draw count (conservative for anticorrelation).
"""
import numpy as np
from scipy.fft import next_fast_len
from scipy.special import ndtri
from scipy.stats import rankdata


def split(x):
    n = x.shape[1] // 2
    return np.concatenate((x[:, :n], x[:, -n:]), axis=0)


def rank_normalize(x):
    return ndtri((rankdata(x).reshape(x.shape) - .375) / (x.size + .25))


def rhat(x):
    n = x.shape[1]
    w = np.var(x, axis=1, ddof=1).mean()
    if w <= 0 or not np.isfinite(w):
        return float('inf')
    return float(np.sqrt(((n-1)/n*w + np.var(x.mean(axis=1), ddof=1))/w))


def ess(x):
    m, n = x.shape
    centered = x - x.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(centered, n=next_fast_len(2*n), axis=1)
    acov = np.fft.irfft(fft * fft.conj(), axis=1)[:, :n] / n
    w = np.var(x, axis=1, ddof=1).mean()
    vp = (n-1)/n*w + np.var(x.mean(axis=1), ddof=1)
    if vp <= 0 or not np.isfinite(vp):
        return 0.
    rho = 1 - (w - acov.mean(axis=0))/vp
    rho[0] = 1.
    pairs = []
    for i in range(0, n-1, 2):
        pair = rho[i]+rho[i+1]
        if pair < 0:
            break
        pairs.append(min(pair, pairs[-1]) if pairs else pair)
    return float(min(m*n, m*n/max(1., -1+2*sum(pairs))))


def diagnostic_report(chains, names):
    chains = np.asarray(chains)
    if chains.ndim != 3 or chains.shape[0] < 4 or chains.shape[1] < 20:
        raise ValueError('Need at least four chains and 20 retained draws')
    rows = {}
    for j, name in enumerate(names):
        original = chains[:, :, j]
        x = split(original)
        z = rank_normalize(x)
        folded = rank_normalize(split(abs(original - np.median(original))))
        lo, hi = np.quantile(original, [.05, .95])
        bulk = ess(z)
        rows[name] = {'rank_folded_split_rhat': max(rhat(z), rhat(folded)),
                      'bulk_ess': bulk,
                      'tail_ess': min(ess(split((original <= lo).astype(float))),
                                      ess(split((original <= hi).astype(float)))),
                      'mean_mcse': float(original.std(ddof=1)/np.sqrt(max(ess(x), 1.)))}
    passed = all(np.isfinite(r['rank_folded_split_rhat']) and
                 r['rank_folded_split_rhat'] < 1.01 and
                 r['bulk_ess'] >= 400 and r['tail_ess'] >= 400 for r in rows.values())
    return {'variables': rows, 'passed': passed,
            'gate': 'Every target AND nuisance: rank/folded split Rhat <1.01; bulk and 5/95% tail ESS >=400 across four chains',
            'limits': 'Finite diagnostics cannot prove global mode discovery; periodic PA is also checked through sine/cosine. ESS capped at retained draw count.'}
