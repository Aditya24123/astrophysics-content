"""Exploratory validation residual checks by band and Fourier radius.

Uses ONLY already-selected, accepted posterior predictive draws. No new target
draws, posterior fitting, test evaluation, or weighting/selection of theta.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hii_sim.io import atomic_json,digest
from review_inference_v2 import inputs,verify_preserved,read,arrays
from review_contract import CheckedLikelihood,full_diagnostics
from conditional_predictive_v2 import conditional_diagnostics


def group_stats(like,vectors,epsilons):
    m=like.m
    scale=like.o.thermal.sigma_real_jy[m.frequency_index]*np.where(m.self_conjugate,np.sqrt(2),1.)
    whiten_inverse=[(idx,np.linalg.inv(inv)) for idx,inv in like.buckets]
    radius=np.hypot(np.fft.fftfreq(like.grid.npix)[m.x_index],np.fft.fftfreq(like.grid.npix)[m.y_index])
    edges=np.quantile(radius,[0,.25,.5,.75,1])
    radial=np.searchsorted(edges[1:-1],radius,side='right')
    groups=[('band_'+str(j),m.frequency_index==j) for j in range(6)]+[('radius_'+str(j),radial==j) for j in range(4)]
    observed=[];replicated=[]
    for vec,epsilon in zip(vectors,epsilons):
        r=(vec-like.y)/scale
        noise=np.empty_like(epsilon)
        for idx,chol in whiten_inverse:noise[idx]=np.einsum('bij,bj->bi',chol,epsilon[idx])
        noise/=scale
        observed.append([float(np.mean(r[mask]**2)) for _,mask in groups])
        replicated.append([float(np.mean(noise[mask]**2)) for _,mask in groups])
    observed,replicated=np.array(observed),np.array(replicated)
    return {'groups':{name:{'components':int(mask.sum()),'observed_rms_quantiles_025_50_975':np.quantile(np.sqrt(observed[:,j]),[.025,.5,.975]).tolist(),
                            'replicate_discrepancy_tail_fraction':float(np.mean(replicated[:,j]>=observed[:,j]))}
                      for j,(name,mask) in enumerate(groups)},
            'radius_edges_cycles_per_pixel':edges.tolist(),'draws':len(vectors),
            'interpretation':'Raw residuals / conditional per-component thermal sigma (self-mode sqrt(2) included); replicate noise unwhitened with the exact cross-frequency covariance. Groups remain correlated. Exploratory validation diagnostics, not omnibus fit tests or population calibration.'}


def execute(review):
    verify_preserved(review)
    parent=review/'v2_02';ext=review/'extensions_01';out=review/'predictive_details_01'
    out.mkdir(exist_ok=False)
    plan=read('ml/review_plan_v2.json');old,follow,_,_,config,selection=inputs(plan)
    ext_results=read(ext/'results.json')
    initial=read(parent/'neural_conditional_predictive.json')['records']
    results=[]
    for number,inp in enumerate(selection,1):
        like=CheckedLikelihood(inp,config)
        chain=arrays(follow/f'bayes_{number:02d}_chains.npz')['coordinates']
        if not full_diagnostics(chain,config)['passed']:raise ValueError('Conventional chain gate failed')
        flat=chain.reshape(-1,9)
        indices=arrays(parent/f'bayes_{number:02d}_predictive_replay.npz')['posterior_flat_indices']
        rng=np.random.default_rng(read('ml/followup_plan_v1.json')['seed']+number-1)
        for _ in range(6):rng.uniform(size=7);rng.normal(size=2)
        rng.normal(size=(1,4,2048))
        np.testing.assert_array_equal(rng.choice(len(flat),128,replace=False),indices)
        vectors=[];eps=[]
        for k in indices:
            t=flat[k].copy();t[4]%=1
            vectors.append(like.vector(t));eps.append(rng.normal(size=len(like.y)))
        case={'source_id':inp['bookkeeping']['source_id'],'bayesian':group_stats(like,vectors,eps),'neural_conditionals':[]}
        for j in range(2):
            index=(number-1)*2+j;result=ext_results[index];original=initial[index];tag=original['tag']
            if not result['converged']:
                case['neural_conditionals'].append({'tag':tag,'status':'WITHHELD: conditional posterior unconverged'});continue
            location=parent if result['used_parent_pass'] else ext
            raw=arrays(location/f'{tag}_chains.npz')['coordinates']
            if not conditional_diagnostics(raw)['passed']:raise ValueError('Conditional raw-chain gate failed')
            saved=arrays(location/f'{tag}_predictive.npz')
            seed=(plan['seed']+number*100000+j*10000+777) if result['used_parent_pass'] else 231029+index*1009
            rng=np.random.default_rng(seed)
            np.testing.assert_array_equal(rng.choice(raw.shape[0]*raw.shape[1],64,replace=False),saved['posterior_flat_indices'])
            fixed=arrays(location/f'{tag}_chains.npz')['fixed_prior_units']
            vectors=[];eps=[]
            for n in saved['nuisance_draws']:
                t=np.r_[fixed,n[0]%1,n[1:]]
                vectors.append(like.vector(t));eps.append(rng.normal(size=len(like.y)))
            # Reconfirm existing scalar receipts; these subgroup checks are additional.
            actual=np.array([[np.mean(like.whiten(v-like.y)**2),np.mean(e*e)] for v,e in zip(vectors,eps)])
            np.testing.assert_allclose(actual,saved['discrepancies'],rtol=1e-10,atol=1e-10)
            case['neural_conditionals'].append({'tag':tag,'status':'accepted conditional only, one fixed theta',
                                               'statistics':group_stats(like,vectors,eps)})
        results.append(case)
    atomic_json(out/'residual_checks.json',{'cases':results,'test_used':False,'new_inference':False,
                'neural_outer_draws_per_source':2,'aggregate_neural_predictive_claim':False,
                'source_sha256':digest(Path(__file__)),'exploratory':True})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    lines=['## Exploratory band/radius predictive checks','',
           'A global residual-energy check can hide structured discrepancy. These additional validation-only statistics '
           'use the same accepted posterior draws and exact replicate-noise streams, grouped by band and operator-only '
           'Fourier-radius quartiles. Residuals are divided by the correct conditional per-component thermal sigma; '
           'self-mode variance and cross-frequency covariance remain in the replicate calculation. Groups are correlated. '
           'This is exploratory diagnostic work, not a new blind evaluation or a calibrated omnibus test.', '',
           '| Case / posterior | Group | Median observed residual RMS / sigma | Replicate tail fraction |', '|---|---|---:|---:|']
    for i,(ax,case) in enumerate(zip(axes,results),1):
        series=[('Bayesian joint posterior',case['bayesian'])]+[(r['tag'],r['statistics']) for r in case['neural_conditionals'] if 'statistics' in r]
        for label,stats in series:
            q=np.array([stats['groups']['band_'+str(j)]['observed_rms_quantiles_025_50_975'] for j in range(6)])
            ax.plot(like.o.frequencies_ghz,q[:,1],marker='o',label=label)
            ax.fill_between(like.o.frequencies_ghz,q[:,0],q[:,2],alpha=.12)
            for name,g in stats['groups'].items():
                lines.append(f'| {i}: {label} | {name} | {g["observed_rms_quantiles_025_50_975"][1]:.4f} | {g["replicate_discrepancy_tail_fraction"]:.4f} |')
        ax.axhline(1,color='gray',ls=':')
        ax.set(title=f'Validation observation {i}',xlabel='Frequency [GHz]',ylabel='Residual RMS / thermal sigma')
        ax.legend(fontsize=8)
    fig.suptitle('Accepted joint Bayes vs fixed-theta neural conditional draws\nTwo neural target draws per source: insufficient for full predictive calibration')
    fig.savefig(out/'band_residuals.png',dpi=180);plt.close(fig)
    lines+=['','![Band residual checks](band_residuals.png)','',
            'Each neural curve conditions on ONE pre-frozen theta draw and integrates accepted nuisance samples; the band '
            'is conditional posterior variation, not a full neural marginal predictive interval. Unconverged conditionals '
            'are withheld and remain failures in the parent/extension record. No inference about unselected sources or '
            'real nebulae follows from these two validation observations.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verify_preserved(review)
    atomic_json(out/'completion.json',{'complete':True,'posterior_inputs_accepted':True,'no_new_test_or_model_fit':True,
                'output_hashes':{p.name:digest(p) for p in out.iterdir() if p.is_file()}})
    print('Exploratory validation band/radius residual evidence saved to '+str(out),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--review',type=Path,default=Path('output/ml_initial_20261023/review_01'));args=ap.parse_args()
    lock=Path('output/ml_initial_20261023/continuation.lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'purpose':'validation_predictive_details'}).encode());os.close(fd)
    try:execute(args.review)
    finally:lock.unlink()
