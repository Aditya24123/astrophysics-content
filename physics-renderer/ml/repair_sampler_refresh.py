"""Single fixed proposal refresh; failed initial chains are proposals, not results."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtr
from hii_sim.io import atomic_json,write_npz,digest
from review_inference_v2 import inputs,read,arrays
from review_contract import CheckedLikelihood
from repair_campaign import OUT,PLAN,CompressedLikelihood,gate,latent_to_unit,status

def proposal_parameters(pilot,inflation):
    flat=pilot.reshape(-1,9);mean=flat.mean(axis=0);cov=np.cov(flat.T);vals,vec=np.linalg.eigh(cov);cov=(vec*np.maximum(vals,1e-8))@vec.T
    others=np.array([j for j in range(9) if j!=4]);beta=cov[others,4]/cov[4,4];conditional=cov[np.ix_(others,others)]-np.outer(cov[others,4],cov[4,others])/cov[4,4]
    return mean,np.linalg.cholesky(cov*inflation),others,beta,np.linalg.cholesky(conditional*inflation)

def angle_logq(z,mean,others,beta,chol,df):
    center=mean[others]+beta*(z[4]-mean[4]);v=np.linalg.solve(chol,z[others]-center)
    return -.5*z[4]**2-.5*(df+8)*np.log1p(v@v/df)

def angle_proposal(rng,mean,others,beta,chol,df):
    z=np.empty(9);z[4]=rng.normal();z[others]=mean[others]+beta*(z[4]-mean[4])+chol@rng.normal(size=8)/np.sqrt(rng.chisquare(df)/df)
    return z

def refresh(like,config,parent,dest,settings,seed):
    initial=read(parent.with_suffix('.json'))
    if initial['passed']:raise ValueError('Refresh restricted to a failed initial gate')
    pilot=arrays(parent.with_suffix('.chains.npz'))['latent_coordinates'];mean,chol,others,beta,conditional=proposal_parameters(pilot,settings['proposal_covariance_inflation']);inv=np.linalg.inv(chol);df=settings['student_df']
    def logq(z):
        v=inv@(z-mean);return -.5*(df+9)*np.log1p(v@v/df)
    rngs=[np.random.default_rng(seed+k*1009) for k in range(4)];states=pilot[:,-1].copy();lp=np.array([like.logp(z) for z in states]);acc=np.zeros((4,3),int);attempt=np.zeros((4,3),int);draws=[];lls=[];start,cpu=time.perf_counter(),time.process_time()
    atomic_json(dest.with_suffix('.binding.json'),dict(settings=settings,plan_sha256=digest('ml/repair_refresh_plan.json'),code_sha256=digest(__file__),seed=seed,parent_chains_sha256=digest(parent.with_suffix('.chains.npz')),parent_failed=True,phase='new homogeneous production; pilot discarded; all 4 own chain states retained, warmup500',mean=mean.tolist(),chol=chol.tolist(),angle_others=others.tolist(),angle_beta=beta.tolist(),angle_chol=conditional.tolist(),target='IDENTICAL compressed likelihood and exact normal latent-coordinate prior'))
    for step in range(settings['warmup']+settings['max_draws']):
        for k,rng in enumerate(rngs):
            if rng.uniform()<settings['angle_probability']:
                kind=2;proposal=angle_proposal(rng,mean,others,beta,conditional,df);h=angle_logq(states[k],mean,others,beta,conditional,df)-angle_logq(proposal,mean,others,beta,conditional,df)
            else:
                kind=int(rng.uniform()>=settings['independence_probability'])
                proposal=(mean+chol@rng.normal(size=9)/np.sqrt(rng.chisquare(df)/df) if kind==0 else states[k]+settings['local_scale']*chol@rng.normal(size=9));h=logq(states[k])-logq(proposal) if kind==0 else 0.
            attempt[k,kind]+=1;value=like.logp(proposal)
            if np.log(rng.uniform())<value-lp[k]+h:states[k]=proposal;lp[k]=value;acc[k,kind]+=1
        if step>=settings['warmup']:draws.append(states.copy());lls.append(lp+.5*np.sum(states*states,axis=1))
        n=len(draws);budget=(time.perf_counter()-start>settings['case_wall_seconds'] or datetime.now(timezone.utc)>=datetime.fromisoformat(read(PLAN)['science_stop_utc'])) if step%100==0 else False
        if n and (n%settings['diagnostic_every']==0 or budget or step==settings['warmup']+settings['max_draws']-1):
            z=np.array(draws).transpose(1,0,2);ll=np.array(lls).T;diag=gate(z,ll,config) if n>=20 else None
            write_npz(dest.with_suffix('.chains.npz'),latent_coordinates=z,coordinates=latent_to_unit(z),log_likelihood=ll,states=states)
            result=dict(passed=bool(diag and diag['passed']),diagnostics=diag,draws_per_chain=n,warmup=settings['warmup'],wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,acceptance=(acc/np.maximum(attempt,1)).tolist(),seed=seed,budget_hit=budget,parent_phase_wall_seconds=initial['wall_seconds'],total_recorded_sampler_wall_seconds=initial['wall_seconds']+time.perf_counter()-start,rng_states=[r.bit_generator.state for r in rngs],posterior_comparison_allowed=bool(diag and diag['passed']))
            atomic_json(dest.with_suffix('.json'),result);print(json.dumps(dict(case=dest.name,draws=n,passed=result['passed'],wall_seconds=result['wall_seconds'])),flush=True)
            if result['passed'] or budget:return result
        if budget and not n:raise TimeoutError('Science stop before refreshed production')
    return result

def execute():
    settings=read('ml/repair_refresh_plan.json');old,follow,base,root,config,selection=inputs(read('ml/review_plan_v2.json'));out=OUT/'compressed_bayes';refreshdir=out/'refresh_01';refreshdir.mkdir(exist_ok=True)
    for route in ['old32','noise128']:
        for number,inp in enumerate(selection,1):
            parent=out/f'{route}_{number:02d}';dest=refreshdir/parent.name
            if not parent.with_suffix('.json').exists() or read(parent.with_suffix('.json'))['passed'] or dest.with_suffix('.json').exists():continue
            if datetime.now(timezone.utc)>=datetime.fromisoformat(read(PLAN)['science_stop_utc']):return
            c=arrays(parent.with_suffix('.operator.npz'))['compression'];like=CompressedLikelihood(CheckedLikelihood(inp,config),c);status('sampler_refresh',route=route,case=number)
            refresh(like,config,parent,dest,settings,settings['seed']+number+(10000 if route=='noise128' else 0))

if __name__=='__main__':
    lock=Path('output/ml_initial_20261023/continuation.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,f'repair refresh pid={os.getpid()}'.encode());os.close(fd)
    try:execute()
    finally:lock.unlink()
