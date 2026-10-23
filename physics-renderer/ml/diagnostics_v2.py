"""Strict, JSON-safe rank/folded diagnostics for new audits and conditional runs.

Rank split/folded Rhat, FFT initial-positive/monotone Geyer ESS. See
https://mc-stan.org/docs/reference-manual/analysis.html . ESS is conservatively
capped at retained draws, unlike implementations permitting antithetic ESS>N.
Original diagnostics.py is preserved for historical reproducibility.
"""
import numpy as np
from scipy.fft import next_fast_len
from scipy.special import ndtri
from scipy.stats import rankdata


def split(x):
    n=x.shape[1]//2
    return np.concatenate((x[:, :n], x[:, -n:]), axis=0)


def rank_normalize(x):
    return ndtri((rankdata(x).reshape(x.shape)-.375)/(x.size+.25))


def autocovariance(x):
    x=np.asarray(x, float)
    centered=x-x.mean(axis=1, keepdims=True)
    length=next_fast_len(2*x.shape[1])
    transformed=np.fft.rfft(centered, n=length, axis=1)
    # Explicit n matters when next_fast_len returns an ODD length.
    return np.fft.irfft(transformed*transformed.conj(), n=length, axis=1)[:, :x.shape[1]]/x.shape[1]


def ess(x):
    m,n=x.shape
    w=x.var(axis=1, ddof=1).mean()
    variance=(n-1)/n*w+x.mean(axis=1).var(ddof=1)
    if variance<=0 or not np.isfinite(variance):
        return 0.
    rho=1-(w-autocovariance(x).mean(axis=0))/variance
    rho[0]=1.
    pairs=[]
    for i in range(0,n-1,2):
        pair=rho[i]+rho[i+1]
        if pair<0:
            break
        pairs.append(min(pair,pairs[-1]) if pairs else pair)
    return float(min(m*n,m*n/max(1.,-1+2*sum(pairs))))


def rhat(x):
    n=x.shape[1]
    w=x.var(axis=1,ddof=1).mean()
    if w<=0 or not np.isfinite(w):
        return None
    value=np.sqrt(((n-1)/n*w+x.mean(axis=1).var(ddof=1))/w)
    return float(value) if np.isfinite(value) else None


def diagnostic_report(chains,names,rhat_limit=1.01,ess_min=400):
    chains=np.asarray(chains,float)
    if (chains.ndim!=3 or chains.shape[0]<4 or chains.shape[1]<20 or
        len(names)!=chains.shape[2] or not names or len(set(names))!=len(names)):
        raise ValueError('Need >=4 chains, >=20 draws, and one unique explicit name for EVERY variable')
    if not np.isfinite(rhat_limit) or rhat_limit<=1 or not np.isfinite(ess_min) or ess_min<=0:
        raise ValueError('Invalid convergence thresholds')
    rows={}
    for j,name in enumerate(names):
        original=chains[:,:,j]
        row={'rank_folded_split_rhat':None,'bulk_ess':0.,'tail_ess':0.,'mean_mcse':None,'passed':False,'failure':None}
        if not np.isfinite(original).all():
            row['failure']='nonfinite_chain_values'
        else:
            x=split(original)
            z=rank_normalize(x)
            a=rhat(z)
            b=rhat(rank_normalize(split(abs(original-np.median(original)))))
            if a is None or b is None:
                row['failure']='zero_within_chain_variance_or_degenerate_folded_distribution'
            else:
                lo,hi=np.quantile(original,[.05,.95])
                row.update(rank_folded_split_rhat=max(a,b),bulk_ess=ess(z),
                           tail_ess=min(ess(split((original<=lo).astype(float))),ess(split((original<=hi).astype(float)))),
                           mean_mcse=float(original.std(ddof=1)/np.sqrt(max(ess(x),1.))))
                row['passed']=bool(max(a,b)<rhat_limit and min(row['bulk_ess'],row['tail_ess'])>=ess_min)
                if not row['passed']:
                    row['failure']='rhat_or_ess_gate_failed'
        rows[name]=row
    return {'variables':rows,'passed':all(r['passed'] for r in rows.values()),
            'chain_shape':list(chains.shape),'rhat_limit':rhat_limit,'ess_min':ess_min,
            'method':'rank/folded split Rhat; explicit-length FFT Geyer positive/monotone bulk and 5/95% tail ESS; capped at retained count',
            'limits':'Finite diagnostic passes do not prove global mode coverage or model calibration.'}
