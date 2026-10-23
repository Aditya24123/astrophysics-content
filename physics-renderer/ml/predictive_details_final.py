"""Explore band/radius residuals using the latest accepted conditional phases."""
import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hii_sim.io import atomic_json,digest
from predictive_details import group_stats
from review_inference_v2 import inputs,verify_preserved,read,arrays
from review_contract import CheckedLikelihood,full_diagnostics
from conditional_predictive_v2 import conditional_diagnostics


def execute(review):
    verify_preserved(review)
    parent=review/'v2_02';ext=review/'extensions_01';angle=review/'angle_refresh_01';out=review/'predictive_details_01'
    out.mkdir(exist_ok=False)
    plan=read('ml/review_plan_v2.json');_,follow,_,_,config,selection=inputs(plan)
    original=read(parent/'neural_conditional_predictive.json')['records'];extended=read(ext/'results.json')
    angle_results={r['tag']:r for r in read(angle/'completion.json')['new_phase_results']}
    results=[];input_hashes={}
    for number,inp in enumerate(selection,1):
        like=CheckedLikelihood(inp,config)
        path=follow/f'bayes_{number:02d}_chains.npz';chain=arrays(path)['coordinates'];input_hashes[str(path)]=digest(path)
        if not full_diagnostics(chain,config)['passed']:raise ValueError('Bayesian chain gate failed')
        flat=chain.reshape(-1,9);indices=arrays(parent/f'bayes_{number:02d}_predictive_replay.npz')['posterior_flat_indices']
        rng=np.random.default_rng(read('ml/followup_plan_v1.json')['seed']+number-1)
        for _ in range(6):rng.uniform(size=7);rng.normal(size=2)
        rng.normal(size=(1,4,2048));np.testing.assert_array_equal(rng.choice(len(flat),128,replace=False),indices)
        vecs=[];eps=[]
        for k in indices:
            t=flat[k].copy();t[4]%=1;vecs.append(like.vector(t));eps.append(rng.normal(size=len(like.y)))
        case={'source_id':inp['bookkeeping']['source_id'],'bayesian':group_stats(like,vecs,eps),'neural_conditionals':[]}
        for j in range(2):
            index=(number-1)*2+j;old=extended[index];record=original[index];tag=record['tag']
            if old['converged']:
                location=parent if old['used_parent_pass'] else ext
                seed=(plan['seed']+number*100000+j*10000+777) if old['used_parent_pass'] else 231029+index*1009
            else:
                repaired=angle_results[tag]
                if not repaired['converged']:
                    case['neural_conditionals'].append({'tag':tag,'status':'WITHHELD: conditional convergence failed'});continue
                location=angle;seed=repaired['predictive_seed']
            chain_path=location/f'{tag}_chains.npz';archive=arrays(chain_path);raw=archive['coordinates'];input_hashes[str(chain_path)]=digest(chain_path)
            if not conditional_diagnostics(raw)['passed']:raise ValueError('Raw conditional gate failed')
            saved=arrays(location/f'{tag}_predictive.npz');rng=np.random.default_rng(seed)
            np.testing.assert_array_equal(rng.choice(raw.shape[0]*raw.shape[1],64,replace=False),saved['posterior_flat_indices'])
            vectors=[];epsilon=[]
            for n in saved['nuisance_draws']:
                vectors.append(like.vector(np.r_[archive['fixed_prior_units'],n[0]%1,n[1:]]));epsilon.append(rng.normal(size=len(like.y)))
            replay=np.array([[np.mean(like.whiten(v-like.y)**2),np.mean(e*e)] for v,e in zip(vectors,epsilon)])
            np.testing.assert_allclose(replay,saved['discrepancies'],rtol=1e-10,atol=1e-10)
            case['neural_conditionals'].append({'tag':tag,'accepted_phase':str(location),'neural_outer_index':j,
                                               'statistics':group_stats(like,vectors,epsilon)})
        results.append(case)
    atomic_json(out/'residual_checks.json',{'cases':results,'input_hashes':input_hashes,'source_sha256':digest(Path(__file__)),
                'test_used':False,'new_inference':False,'outer_neural_draws_per_observation':2,'exploratory':True,
                'aggregate_neural_predictive_calibration_claim':False})
    lines=['## Additional exploratory band/radius residual checks','',
           'Global residual energy is not an omnibus check. Using the SAME accepted posterior samples and exact replicate-noise '
           'streams, this validation-only analysis groups raw residuals by band and operator-defined Fourier-radius quartiles. '
           'Thermal per-component sigma includes the self-mode factor; replicate noise retains the cross-frequency covariance. '
           'Groups are correlated. These are exploratory posterior checks, not population calibration or new model fitting.', '',
           '| Case/posterior | Group | Median observed RMS / thermal sigma | Replicate tail fraction |','|---|---|---:|---:|']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for i,(ax,case) in enumerate(zip(axes,results),1):
        series=[('Bayesian joint',case['bayesian'])]+[(r['tag'],r['statistics']) for r in case['neural_conditionals'] if 'statistics' in r]
        for label,stats in series:
            q=np.array([stats['groups'][f'band_{j}']['observed_rms_quantiles_025_50_975'] for j in range(6)])
            ax.plot(config['instrument']['frequencies_ghz'],q[:,1],marker='o',label=label)
            ax.fill_between(config['instrument']['frequencies_ghz'],q[:,0],q[:,2],alpha=.12)
            for name,g in stats['groups'].items():
                lines.append(f'| {i}: {label} | {name} | {g["observed_rms_quantiles_025_50_975"][1]:.4f} | {g["replicate_discrepancy_tail_fraction"]:.4f} |')
        ax.axhline(1,color='gray',ls=':');ax.set(title=f'Validation observation {i}',xlabel='Frequency [GHz]',ylabel='Observed residual RMS / thermal sigma');ax.legend(fontsize=8)
    fig.suptitle('Joint Bayesian vs fixed-neural-theta conditional residuals\nOnly two neural theta draws per source: no full predictive calibration claim')
    fig.savefig(out/'band_residuals.png',dpi=180);plt.close(fig)
    lines+=['','![Accepted validation residuals](band_residuals.png)','',
            'Each neural curve conditions on ONE original frozen theta draw and integrates the latest accepted nuisance '
            'posterior. Shading is variation across its conditional nuisance samples, not a full neural marginal predictive '
            'interval. Low tail fractions can identify an awkward theta draw; four draws cannot estimate how often the '
            'neural posterior makes that error. No outer draw was removed or replaced.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verify_preserved(review)
    atomic_json(out/'completion.json',{'complete':True,'accepted_neural_outer_cases':sum(len(c['neural_conditionals']) for c in results),
                'output_hashes':{p.name:digest(p) for p in out.iterdir() if p.is_file()}})
    print('Latest accepted validation subgroup residual checks saved',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--review',type=Path,default=Path('output/ml_initial_20261023/review_01'));args=ap.parse_args()
    lock=Path('output/ml_initial_20261023/continuation.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,b'accepted_predictive_details');os.close(fd)
    try:execute(args.review)
    finally:lock.unlink()
