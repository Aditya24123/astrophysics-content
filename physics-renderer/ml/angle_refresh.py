"""Targeted PA-tail kernel repair; prior/likelihood/outer theta unchanged."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from hii_sim.io import atomic_json,digest,write_npz
from conditional_predictive_v2 import ConditionalLikelihood,conditional_diagnostics
from review_contract import CheckedLikelihood
from review_inference_v2 import inputs,verify_preserved,read,arrays


def global_parameters(center,chol):
    cov=chol@chol.T
    beta=cov[1:,0]/cov[0,0]
    conditional=cov[1:,1:]-np.outer(cov[1:,0],cov[0,1:])/cov[0,0]
    values,vectors=np.linalg.eigh(conditional)
    factor=cholesky((vectors*np.maximum(values,1e-12))@vectors.T*1.5,lower=True)
    inverse=solve_triangular(factor,np.eye(4),lower=True)
    return beta,factor,inverse


def global_proposal(rng,center,branch,beta,factor):
    pa=branch+rng.uniform(-.5,.5)
    other=center[1:]+beta*(pa-center[0])+factor@rng.normal(size=4)/np.sqrt(rng.chisquare(5)/5)
    return np.r_[pa,other]


def global_logq(n,center,beta,inverse):
    r=inverse@(n[1:]-center[1:]-beta*(n[0]-center[0]))
    return -4.5*np.log1p(r@r/5)  # df=5, dimension=4; PA uniform full period.


def run(clike,snapshot,parent_status,plan,out,tag,seed):
    states=snapshot['states'].copy();lps=snapshot['logp'].copy();center=snapshot['proposal_center'];chol=snapshot['proposal_cholesky']
    branch=parent_status['branch_center'];rngs=[np.random.default_rng(seed+1009*c) for c in range(4)]
    np.testing.assert_allclose([clike.logp(n,branch) for n in states],lps,rtol=0,atol=1e-8)
    ind=chol*np.sqrt(1.5);inv=solve_triangular(ind,np.eye(5),lower=True)
    beta,factor,global_inv=global_parameters(center,chol)
    def logq(n):
        r=inv@(n-center);return -5*np.log1p(r@r/5)
    started,cpu=time.perf_counter(),time.process_time();history=[];accepted=np.zeros((4,4),int);attempted=accepted.copy()
    completed=0;reason='draw_budget'
    for step in range(plan['warmup']+plan['max_draws']):
        if time.perf_counter()-started>=plan['wall_seconds_per_case']:reason='wall_budget';break
        for c,rng in enumerate(rngs):
            p=rng.uniform()
            if p<.3:
                kind=0;proposal=center+ind@rng.normal(size=5)/np.sqrt(rng.chisquare(5)/5);correction=logq(states[c])-logq(proposal)
            elif p<.6:
                kind=1;proposal=states[c]+.7*chol@rng.normal(size=5);correction=0.
            elif p<.75:
                kind=2;proposal=states[c].copy();proposal[3:]+=rng.normal(size=2)*.15;correction=0.
            else:
                kind=3;proposal=global_proposal(rng,center,branch,beta,factor)
                correction=global_logq(states[c],center,beta,global_inv)-global_logq(proposal,center,beta,global_inv)
            lp=clike.logp(proposal,branch);attempted[c,kind]+=1
            if np.log(rng.uniform())<lp-lps[c]+correction:states[c],lps[c]=proposal,lp;accepted[c,kind]+=1
        completed=step+1
        if step>=plan['warmup']:history.append(states.copy())
        if history and len(history)%plan['diagnostic_every']==0:
            diag=conditional_diagnostics(np.array(history).transpose(1,0,2))
            print(f'{tag} PA-refresh phase: {len(history)} /chain; pass={diag["passed"]}',flush=True)
            if diag['passed']:reason='convergence_gate';break
    chain=np.array(history).transpose(1,0,2) if history else np.empty((4,0,5))
    diag=conditional_diagnostics(chain) if chain.shape[1]>=20 else {'passed':False,'variables':{}}
    write_npz(out/f'{tag}_chains.npz',coordinates=chain,states=states,logp=lps,fixed_prior_units=clike.fixed,
              proposal_center=center,proposal_cholesky=chol)
    result={'tag':tag,'converged':diag['passed'],'diagnostics':diag,'retained_per_chain':chain.shape[1],
            'steps_with_new_phase_warmup':completed,'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,
            'stop_reason':reason,'seed':seed,'branch_center':branch,'rng_end_states':[r.bit_generator.state for r in rngs],
            'acceptance_t_local_cal_globalPA':(accepted/np.maximum(attempted,1)).tolist(),
            'forward_calls':clike.forward_calls,'old_draws_discarded_from_artifacts':False,
            'diagnostic_phase':'Only new homogeneous-kernel retained phase; older chains remain unchanged and are not counted as passing'}
    atomic_json(out/f'{tag}_sampling.json',result)
    return chain,result


def execute(review):
    verify_preserved(review)
    parent=review/'v2_02';old_ext=review/'extensions_01';out=review/'angle_refresh_01';out.mkdir(exist_ok=False)
    plan=read('ml/angle_refresh_plan.json');_,_,_,_,config,selection=inputs(read('ml/review_plan_v2.json'))
    original={r['tag']:r for r in read(parent/'neural_conditional_predictive.json')['records']}
    old_results=read(old_ext/'results.json')
    atomic_json(out/'FROZEN_PROTOCOL.json',{'plan':plan,'plan_sha256':digest('ml/angle_refresh_plan.json'),
                'source_sha256':digest(Path(__file__)),'old_extension_results_sha256':digest(old_ext/'results.json'),
                'selected_failed_tags':[r['tag'] for r in old_results if not r['converged']],
                'prior_and_likelihood_changed':False,'new_neural_outer_draws':0})
    results=[]
    for number,old in enumerate(old_results):
        if old['converged']:continue
        tag=old['tag'];record=original[tag];snapshot=arrays(old_ext/f'{tag}_chains.npz');status=read(old_ext/f'{tag}_sampling.json')
        inp=next(s for s in selection if s['bookkeeping']['realization_id']==record['realization_id'])
        clike=ConditionalLikelihood(CheckedLikelihood(inp,config),np.array(record['neural_target_draw']))
        chain,result=run(clike,snapshot,status,plan,out,tag,plan['seed']+number*1009)
        result['source_state_sha256']=digest(old_ext/f'{tag}_chains.npz');result['new_theta_draws']=0
        if result['converged']:
            rng=np.random.default_rng(231033+number*1009);flat=chain.reshape(-1,5)
            indices=rng.choice(len(flat),plan['predictive_replicates'],replace=False);rows=[]
            for k in indices:
                r=clike.residual(flat[k])[:-2];epsilon=rng.normal(size=len(r));rows.append([float(np.mean(r*r)),float(np.mean(epsilon*epsilon))])
            write_npz(out/f'{tag}_predictive.npz',nuisance_draws=flat[indices],posterior_flat_indices=indices,
                      target_draw=np.array(record['neural_target_draw']),discrepancies=np.array(rows))
            result['conditional_observed_rms_median']=float(np.median(np.sqrt(np.array(rows)[:,0])))
            result['conditional_predictive_tail_fraction']=float(np.mean(np.array(rows)[:,1]>=np.array(rows)[:,0]))
            result['predictive_seed']=231033+number*1009
        else:result['predictive_status']='WITHHELD; even targeted PA proposal did not meet unchanged gate'
        atomic_json(out/f'{tag}_result.json',result);results.append(result)
    final_passes=sum(r['converged'] for r in old_results)+sum(r['converged'] for r in results)
    atomic_json(out/'completion.json',{'execution':'complete','final_conditional_passes':final_passes,'outer_draws_retained':len(old_results),
                'no_new_theta_draws':True,'new_phase_results':results,'original_failures_retained':True,
                'all_neural_predictive_calibration_claimed':False,
                'note':'Only four total outer draws; a passing conditional phase is not full marginal neural predictive calibration.'})
    lines=['## Targeted PA-tail proposal repair','',
           'After same-kernel continuation exhausted its fixed budget, a predeclared full-period PA proposal was added '
           'for the remaining under-mixed conditional. The original target draw, physical/nuisance priors and correlated '
           'likelihood were unchanged. All older chains/failures remain intact. A new homogeneous phase starts from the '
           'saved states, discards 500 declared equilibration steps, and is diagnosed separately; old failed draws are '
           'not counted as passing. The full-period PA proposal uses a correlated 4D Student-t conditional proposal with '
           'its exact Hastings ratio, not a deterministic nuisance fit.', '',
           f'Final accepted conditionals: **{final_passes}/{len(old_results)}**. Four total neural outer draws are still '
           'insufficient to establish full predictive calibration.', '',
           '| Conditional | New retained /chain | Max Rhat | Min bulk ESS | Min tail ESS | Gate | Observed RMS median |',
           '|---|---:|---:|---:|---:|---|---:|']
    for r in results:
        v=list(r['diagnostics']['variables'].values())
        lines.append(f'| {r["tag"]} | {r["retained_per_chain"]} | {max(d["rank_folded_split_rhat"] for d in v):.6f} | '
                     f'{min(d["bulk_ess"] for d in v):.1f} | {min(d["tail_ess"] for d in v):.1f} | '
                     f'{"PASS" if r["converged"] else "FAIL / INCONCLUSIVE"} | {r.get("conditional_observed_rms_median","withheld")} |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verify_preserved(review)
    print(f'Targeted angle phase complete: {final_passes}/{len(old_results)} accepted conditionals overall',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--review',type=Path,default=Path('output/ml_initial_20261023/review_01'));args=ap.parse_args()
    lock=Path('output/ml_initial_20261023/continuation.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,b'conditional_PA_refresh');os.close(fd)
    try:
        atomic_json(args.review/'STATUS.json',{'execution':'actively_executing','stage':'targeted_PA_refresh','pid':os.getpid()})
        execute(args.review)
        atomic_json(args.review/'STATUS.json',{'execution':'stage_complete','stage':'targeted_PA_refresh','scientific_limits_eliminated':False})
    except Exception as exc:
        atomic_json(args.review/'STATUS.json',{'execution':'failed','stage':'targeted_PA_refresh','error':str(exc)});raise
    finally:lock.unlink()
