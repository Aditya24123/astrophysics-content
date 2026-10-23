"""Exact fixed-kernel continuation of failed conditional pilots in NEW files.

Targets, proposals, posterior, old draws, RNG states and failed receipts are
preserved. No new MAP, warmup, neural draw, or target-dependent selection.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.linalg import solve_triangular
from hii_sim.io import atomic_json,digest,write_npz,versions
from conditional_predictive_v2 import ConditionalLikelihood,conditional_diagnostics
from review_contract import CheckedLikelihood
from review_inference_v2 import inputs,verify_preserved,read,arrays


def advance(conditional,snapshot,rng_states,branch_center,calibration_step,steps,wall_seconds):
    """Same transition/order as run_conditional; segmentation preserves RNG path."""
    states=snapshot['states'].copy();lps=snapshot['logp'].copy()
    center,chol=snapshot['proposal_center'],snapshot['proposal_cholesky']
    ind_chol=chol*np.sqrt(1.5)
    inv=solve_triangular(ind_chol,np.eye(5),lower=True)
    rngs=[]
    for state in rng_states:
        rng=np.random.default_rng(0);rng.bit_generator.state=state;rngs.append(rng)
    np.testing.assert_allclose([conditional.logp(n,branch_center) for n in states],lps,rtol=0,atol=1e-8)
    def logq(n):
        v=inv@(n-center)
        return -5*np.log1p(v@v/5)
    started=time.perf_counter();history=[];accepts=np.zeros((4,3),int);attempts=accepts.copy()
    for _ in range(steps):
        if time.perf_counter()-started>=wall_seconds:break
        for c,rng in enumerate(rngs):
            p=rng.uniform()
            if p<.4:
                kind=0;proposal=center+ind_chol@rng.normal(size=5)/np.sqrt(rng.chisquare(5)/5)
                correction=logq(states[c])-logq(proposal)
            elif p<.75:
                kind=1;proposal=states[c]+.7*chol@rng.normal(size=5);correction=0.
            else:
                kind=2;proposal=states[c].copy();proposal[3:]+=rng.normal(size=2)*calibration_step;correction=0.
            lp=conditional.logp(proposal,branch_center);attempts[c,kind]+=1
            if np.log(rng.uniform())<lp-lps[c]+correction:
                states[c],lps[c]=proposal,lp;accepts[c,kind]+=1
        history.append(states.copy())
    chunk=np.array(history).transpose(1,0,2) if history else np.empty((4,0,5))
    updated={**snapshot,'states':states,'logp':lps}
    return chunk,updated,[r.bit_generator.state for r in rngs],{'accepts':accepts.tolist(),'attempts':attempts.tolist()}


def execute(review):
    parent=review/'v2_02';out=review/'extensions_01'
    if read(parent/'status_conditional.json')['execution']!='complete':raise ValueError('Parent conditional job not complete')
    if out.exists():raise FileExistsError('Preserve continuation attempt; do not restart it')
    verify_preserved(review)
    plan=read('ml/review_plan_v2.json')
    _,_,_,_,config,selection=inputs(plan)
    parent_results=read(parent/'neural_conditional_predictive.json')['records']
    out.mkdir()
    source_paths=[Path(__file__),Path('ml/test_resume_conditionals.py'),Path('ml/predictive_details.py'),Path('ml/conditional_predictive_v2.py'),Path('ml/diagnostics_v2.py'),Path('ml/review_contract.py')]
    binding={'method':'Exact same fixed MH kernel, proposal and RNG continuation; old warmup/draws retained',
             'environment':versions(),'code_hashes':{str(p):digest(p) for p in source_paths},
             'parent_binding_sha256':digest(parent/'binding.json'),'parent_results_sha256':digest(parent/'neural_conditional_predictive.json'),
             'parent_input_hashes':{p.name:digest(p) for p in parent.glob('conditional*') if p.is_file()},
             'extra_draw_budget_per_failed_case':6000,'extra_wall_budget_per_failed_case':300,'diagnostic_every':1000,
             'no_new_outer_neural_draws':True,'no_reweighting_or_filtering':True,
             'threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
    atomic_json(out/'binding.json',binding)
    results=[]
    for index,record in enumerate(parent_results):
        tag=record['tag']
        if record['converged']:
            results.append({'tag':tag,'converged':True,'used_parent_pass':True,'additional_wall_seconds':0.,
                            'conditional_observed_rms_median':record['conditional_observed_rms_median'],
                            'parent_diagnostics':record['conditional_sampling']['diagnostics']})
            atomic_json(out/'results.json',results);continue
        archive_path=parent/f'{tag}_chains.npz';status_path=parent/f'{tag}_sampling.json'
        result={'tag':tag,'converged':False,'used_parent_pass':False,'neural_target_draw':record['neural_target_draw']}
        if not archive_path.exists() or not status_path.exists():
            result['failure']='No valid saved state to continue';result['additional_wall_seconds']=0.
            results.append(result);atomic_json(out/'results.json',results);continue
        snapshot=arrays(archive_path);status=read(status_path)
        if snapshot['coordinates'].shape[1]<20:
            result['failure']='Insufficient saved warmup/retained state; no automatic restart';result['additional_wall_seconds']=0.
            results.append(result);atomic_json(out/'results.json',results);continue
        inp=next(s for s in selection if s['bookkeeping']['realization_id']==record['realization_id'])
        like=CheckedLikelihood(inp,config);clike=ConditionalLikelihood(like,np.array(record['neural_target_draw']))
        np.testing.assert_array_equal(clike.fixed,snapshot['fixed_prior_units'])
        before_chain=snapshot['coordinates'].copy();chain=before_chain.copy()
        rng_states=status['rng_end_states'];branch=status['branch_center']
        started,cpu=time.perf_counter(),time.process_time();extra=0;history=[]
        diag=conditional_diagnostics(chain)
        while extra<6000 and time.perf_counter()-started<300:
            remaining=300-(time.perf_counter()-started)
            chunk,snapshot,rng_states,accept=advance(clike,snapshot,rng_states,branch,plan['conditional_sampler']['calibration_step'],min(1000,6000-extra),remaining)
            if not chunk.shape[1]:break
            extra+=chunk.shape[1];chain=np.concatenate((chain,chunk),axis=1)
            diag=conditional_diagnostics(chain)
            history.append({'extra_draws_per_chain':extra,'diagnostics':diag,'acceptance_counts':accept})
            write_npz(out/f'{tag}_chains.npz',**{**snapshot,'coordinates':chain})
            atomic_json(out/f'{tag}_sampling.json',{'converged':diag['passed'],'diagnostics':diag,
                        'parent_draws_per_chain':before_chain.shape[1],'extra_draws_per_chain':extra,
                        'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,
                        'rng_end_states':rng_states,'branch_center':branch,'history':history,
                        'forward_calls':clike.forward_calls,'parent_chain_sha256':digest(archive_path)})
            print(f'{tag} exact continuation: {chain.shape[1]} total/chain; pass={diag["passed"]}',flush=True)
            if diag['passed']:break
        # Literal equality proves no old retained draw was altered or dropped.
        np.testing.assert_array_equal(chain[:,:before_chain.shape[1]],before_chain)
        result.update(converged=diag['passed'],diagnostics=diag,extra_draws_per_chain=extra,
                      old_draws_preserved=True,new_outer_draws=0)
        if diag['passed']:
            rng=np.random.default_rng(231029+index*1009);flat=chain.reshape(-1,5)
            indices=rng.choice(len(flat),64,replace=False);rows=[]
            for k in indices:
                residual=clike.residual(flat[k])[:-2];epsilon=rng.normal(size=len(residual))
                rows.append([float(np.mean(residual**2)),float(np.mean(epsilon**2))])
            write_npz(out/f'{tag}_predictive.npz',posterior_flat_indices=indices,nuisance_draws=flat[indices],
                      target_draw=np.array(record['neural_target_draw']),discrepancies=np.array(rows))
            result['conditional_observed_rms_median']=float(np.median(np.sqrt(np.array(rows)[:,0])))
            result['conditional_predictive_tail_fraction']=float(np.mean(np.array(rows)[:,1]>=np.array(rows)[:,0]))
        else:result['predictive_status']='WITHHELD: extended conditional did not pass; outer draw retained as failed'
        result['additional_wall_seconds']=time.perf_counter()-started
        result['additional_forward_calls_including_predictive']=clike.forward_calls
        result['chain_sha256']=digest(out/f'{tag}_chains.npz') if (out/f'{tag}_chains.npz').exists() else None
        atomic_json(out/f'{tag}_result.json',result);results.append(result);atomic_json(out/'results.json',results)
    for name,expected in binding['parent_input_hashes'].items():
        if digest(parent/name)!=expected:raise ValueError('Original conditional pilot artifact changed')
    verify_preserved(review)
    atomic_json(out/'completion.json',{'execution':'complete','conditional_passes':sum(r['converged'] for r in results),
                'conditional_total':len(results),'aggregate_neural_calibration_claim':False,
                'outer_draw_limit':'Only four total fixed theta draws, so a reliable neural predictive distribution/calibration assessment is still unestablished',
                'parent_artifacts_unchanged':True,'original_dataset_runs_simulator_unchanged':True,
                'additional_wall_seconds':sum(r['additional_wall_seconds'] for r in results),
                'output_hashes':{p.name:digest(p) for p in out.glob('*') if p.is_file() and p.name!='completion.json'}})


def finalize(review):
    out=review/'extensions_01';parent=review/'v2_02'
    completion=read(out/'completion.json');results=read(out/'results.json')
    binding=read(out/'binding.json')
    for name,sha in completion['output_hashes'].items():
        if digest(out/name)!=sha:raise ValueError('Extension output changed before finalization: '+name)
    for path,sha in binding['code_hashes'].items():
        if digest(path)!=sha:raise ValueError('Extension source changed: '+path)
    for name,sha in binding['parent_input_hashes'].items():
        if digest(parent/name)!=sha:raise ValueError('Parent conditional artifact changed')
    verify_preserved(review)
    for result in results:
        tag=result['tag'];path=(parent if result['used_parent_pass'] else out)/f'{tag}_chains.npz'
        if not path.exists():
            if result['converged']:raise ValueError('Accepted conditional has no raw chains')
            continue
        archive=arrays(path);chain=archive['coordinates']
        if not np.isfinite(chain).all() or np.any((chain[:,:,1:3]<0)|(chain[:,:,1:3]>1)):
            raise ValueError('Conditional raw chains invalid')
        diag=conditional_diagnostics(chain)
        if result['converged'] and not diag['passed']:raise ValueError('Raw conditional chains do not pass acceptance')
        if not result['used_parent_pass']:
            initial=arrays(parent/f'{tag}_chains.npz')
            np.testing.assert_array_equal(chain[:,:initial['coordinates'].shape[1]],initial['coordinates'])
            np.testing.assert_array_equal(archive['fixed_prior_units'],initial['fixed_prior_units'])
    original=(review/'BEFORE_AFTER.md').read_text(encoding='utf-8')
    lines=['## Exact continuation of under-mixed conditional pilots','',
           'The initial conditional-pilot failures above remain preserved. For the same fixed neural target draws, the '
           'failed conditionals were continued from their saved current states, fixed proposals and exact RNG states in '
           'new files. No optimizer, target redraw, warmup restart, sample filtering, likelihood change or convergence '
           'threshold relaxation occurred. Regression tests check segmented versus uninterrupted transition paths bit for bit.', '',
           '| Conditional | Original status | Final status | Extra draws/chain | Extra wall seconds incl predictive | Conditional observed RMS median |',
           '|---|---|---|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["tag"]} | {"PASS retained" if r["used_parent_pass"] else "FAIL retained in pilot"} | '
                     f'{"PASS" if r["converged"] else "INCONCLUSIVE"} | {r.get("extra_draws_per_chain",0)} | '
                     f'{r["additional_wall_seconds"]:.2f} | {r.get("conditional_observed_rms_median","withheld")} |')
    lines+=['', f'Final nuisance-conditional convergence: **{completion["conditional_passes"]}/{completion["conditional_total"]}**. '
            'The initial pilot table above is historical and is superseded for convergence status by this continuation table. '
            'Each accepted conditional now supports a posterior predictive check conditional on its single theta draw. '
            'There are still only TWO neural theta draws per validation observation; this does not establish an accurate '
            'full neural predictive marginal or population calibration.', '',
            'An additional restriction of the neural family is visible even with an uninformative likelihood: a uniform '
            'prior-unit coordinate has a logistic logit distribution, while the frozen estimator uses a Gaussian in that '
            'coordinate. It cannot represent the exact uninformative prior or arbitrary curved/multimodal conditionals. '
            'This is an approximation limit, not repaired by extending nuisance chains. The validation temperature and '
            'axis-ratio MAEs both remain slightly worse than their prior-midpoint diagnostics.', '',
            'Additional conditional continuation cost: '
            f'{completion["additional_wall_seconds"]:.2f} wall s, on top of the pilot work already recorded. '
            'This is conventional conditional inference work, never the sub-millisecond amortized neural inference cost.', '',
            '**Completed:** bounded audit, strict verification/contract fixes, independent replay of archived evidence, '
            'mode exploration, coherent conditional-nuisance implementation and the exact continuations shown above. '
            '**Unresolved:** frozen neural model accuracy/posterior-family limitations, few outer predictive draws, limited '
            'source counts, finite mode discovery and synthetic-to-real transfer. All scientific limits have NOT been eliminated. '
            '**No new blind data generation or test rescoring occurred.**']
    (out/'BEFORE_AFTER.md').write_text(original+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    detailed=review/'predictive_details_01/RESULTS.md'
    if detailed.exists():
        with (out/'BEFORE_AFTER.md').open('a',encoding='utf-8') as f:
            f.write('\n\n'+detailed.read_text(encoding='utf-8').replace('(band_residuals.png)','(../predictive_details_01/band_residuals.png)'))
    atomic_json(out/'SUMMARY.json',{**completion,'report':'BEFORE_AFTER.md','scientific_limits_eliminated':False,
                'initial_report_sha256':digest(review/'BEFORE_AFTER.md'),'final_report_sha256':digest(out/'BEFORE_AFTER.md'),
                'original_repair_summary':'../SUMMARY.json','final_conditional_results':results})
    atomic_json(review/'LATEST.json',{'execution':'complete','report':'extensions_01/BEFORE_AFTER.md',
                'summary':'extensions_01/SUMMARY.json','bounded_audit_complete':True,'all_scientific_issues_eliminated':False})
    atomic_json(review/'STATUS.json',{'execution':'complete','stage':'conditional_extension_and_report',
                'utc':datetime.now(timezone.utc).isoformat(),'scientific_limits_eliminated':False})
    print(json.dumps({'report':str(out/'BEFORE_AFTER.md'),'passes':completion['conditional_passes'],
                      'total':completion['conditional_total'],'scientific_limits_eliminated':False}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--review',type=Path,default=Path('output/ml_initial_20261023/review_01'))
    ap.add_argument('--finalize',action='store_true');args=ap.parse_args()
    lock=Path('output/ml_initial_20261023/continuation.lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'purpose':'exact_conditional_continuation'}).encode());os.close(fd)
    try:
        if args.finalize:finalize(args.review)
        else:
            atomic_json(args.review/'STATUS.json',{'execution':'actively_executing','stage':'exact_conditional_continuation','pid':os.getpid()})
            execute(args.review)
    except Exception as exc:
        atomic_json(args.review/'STATUS.json',{'execution':'failed','stage':'exact_conditional_continuation','failure':str(exc)});raise
    finally:lock.unlink()
