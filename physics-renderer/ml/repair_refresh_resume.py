"""Exact homogeneous fixed-kernel continuation, only if refresh still fails."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
import numpy as np
from hii_sim.io import atomic_json,write_npz,digest
from review_inference_v2 import read,arrays,inputs
from review_contract import CheckedLikelihood
from repair_campaign import OUT,PLAN,CompressedLikelihood,gate,latent_to_unit,status
from repair_sampler_refresh import angle_proposal,angle_logq

def advance(like,states,lp,rngs,binding,attempt,acc):
    mean=np.array(binding['mean']);chol=np.array(binding['chol']);others=np.array(binding['angle_others']);beta=np.array(binding['angle_beta']);conditional=np.array(binding['angle_chol']);settings=binding['settings'];df=settings['student_df'];inv=np.linalg.inv(chol)
    def logq(z):
        v=inv@(z-mean);return -.5*(df+9)*np.log1p(v@v/df)
    for k,rng in enumerate(rngs):
        if rng.uniform()<settings['angle_probability']:
            kind=2;proposal=angle_proposal(rng,mean,others,beta,conditional,df);h=angle_logq(states[k],mean,others,beta,conditional,df)-angle_logq(proposal,mean,others,beta,conditional,df)
        else:
            kind=int(rng.uniform()>=settings['independence_probability']);proposal=(mean+chol@rng.normal(size=9)/np.sqrt(rng.chisquare(df)/df) if kind==0 else states[k]+settings['local_scale']*chol@rng.normal(size=9));h=logq(states[k])-logq(proposal) if kind==0 else 0.
        attempt[k,kind]+=1;value=like.logp(proposal)
        if np.log(rng.uniform())<value-lp[k]+h:states[k]=proposal;lp[k]=value;acc[k,kind]+=1

def resume(like,config,parent,dest,max_additional=3000,wall_seconds=180):
    previous=read(parent.with_suffix('.json'));binding=read(parent.with_suffix('.binding.json'));archive=arrays(parent.with_suffix('.chains.npz'));prefix=archive['latent_coordinates']
    if previous['passed']:raise ValueError('Do not extend an already passed case')
    rngs=[]
    for state in previous['rng_states']:
        rng=np.random.default_rng();rng.bit_generator.state=state;rngs.append(rng)
    states=archive['states'].copy();lp=np.array([like.logp(z) for z in states]);draws=[];lls=[];attempt=np.zeros((4,3),int);acc=np.zeros((4,3),int);start,cpu=time.perf_counter(),time.process_time()
    atomic_json(dest.with_suffix('.binding.json'),dict(parent=str(parent),parent_chain_sha256=digest(parent.with_suffix('.chains.npz')),kernel_binding_sha256=digest(parent.with_suffix('.binding.json')),kernel_source_sha256=digest('ml/repair_sampler_refresh.py'),resume_source_sha256=digest(__file__),kernel_changed=False,new_warmup=0,max_additional_draws=max_additional,additional_wall_seconds=wall_seconds,reason='Same homogeneous fixed proposal/target and exact saved PCG64 states; restore all production prefix, no adaptation or threshold changes. Prioritize original compressed cases.'))
    for step in range(max_additional):
        advance(like,states,lp,rngs,binding,attempt,acc);draws.append(states.copy());lls.append(lp+.5*np.sum(states*states,axis=1))
        budget=(time.perf_counter()-start>wall_seconds or datetime.now(timezone.utc)>=datetime.fromisoformat(read(PLAN)['science_stop_utc'])) if step%100==0 else False
        if len(draws)%1000==0 or budget or step==max_additional-1:
            z=np.concatenate((prefix,np.array(draws).transpose(1,0,2)),axis=1);ll=np.concatenate((archive['log_likelihood'],np.array(lls).T),axis=1);np.testing.assert_array_equal(z[:,:prefix.shape[1]],prefix);diag=gate(z,ll,config)
            write_npz(dest.with_suffix('.chains.npz'),latent_coordinates=z,coordinates=latent_to_unit(z),log_likelihood=ll,states=states)
            current=time.perf_counter()-start;result=dict(passed=diag['passed'],diagnostics=diag,draws_per_chain=z.shape[1],warmup=binding['settings']['warmup'],wall_seconds=previous['wall_seconds']+current,cpu_seconds=previous['cpu_seconds']+time.process_time()-cpu,acceptance_additional_segment=(acc/np.maximum(attempt,1)).tolist(),additional_wall_seconds=current,additional_draws_per_chain=len(draws),acceptance=previous['acceptance'],acceptance_scope='acceptance field is original refresh segment; acceptance_additional_segment separately records new segment',parent_phase_wall_seconds=previous['parent_phase_wall_seconds'],total_recorded_sampler_wall_seconds=previous['total_recorded_sampler_wall_seconds']+current,rng_states=[r.bit_generator.state for r in rngs],kernel_changed=False,prefix_array_equal=True,budget_hit=budget,posterior_comparison_allowed=diag['passed'])
            atomic_json(dest.with_suffix('.json'),result);print(json.dumps(dict(case=dest.name,continued_draws=len(draws),total_draws=z.shape[1],passed=diag['passed'],additional_wall_seconds=current)),flush=True)
            if diag['passed'] or budget:return result
    return result

def execute():
    # Admission reserves at least fifteen minutes for independent science checks.
    if datetime.now(timezone.utc)>=datetime(2026,10,23,13,15,tzinfo=timezone.utc):return
    _,_,_,_,config,selection=inputs(read('ml/review_plan_v2.json'));base=OUT/'compressed_bayes/refresh_01';out=base/'continuation_01';out.mkdir(exist_ok=True)
    for number,inp in enumerate(selection,1):
        parent=base/f'old32_{number:02d}';dest=out/parent.name
        if not parent.with_suffix('.json').exists() or read(parent.with_suffix('.json'))['passed'] or dest.with_suffix('.json').exists():continue
        if datetime.now(timezone.utc)>=datetime(2026,10,23,13,15,tzinfo=timezone.utc):return
        c=arrays(OUT/'compressed_bayes'/f'old32_{number:02d}.operator.npz')['compression'];status('exact_sampler_continuation',case=number);resume(CompressedLikelihood(CheckedLikelihood(inp,config),c),config,parent,dest)

if __name__=='__main__':
    lock=Path('output/ml_initial_20261023/continuation.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,f'repair homogeneous resume pid={os.getpid()}'.encode());os.close(fd)
    try:execute()
    finally:lock.unlink()
