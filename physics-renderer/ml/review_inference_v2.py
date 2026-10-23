"""Versioned validation-only audit/repair. Never modifies earlier scientific runs."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import least_squares
from hii_sim.io import atomic_json,digest,write_npz,versions
from hii_sim.loaders import NPEInputDataset
from hii_sim.training_data import load_transform,fit_training_transform,iter_batches
from model import Support,project,predict,sample,fit_pca
from review_contract import CheckedLikelihood,TARGETS,UNITS,physical_targets,full_diagnostics,validate_case_set
from conditional_predictive_v2 import ConditionalLikelihood,run_conditional


def read(p):return json.loads(Path(p).read_text())


def arrays(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def fingerprint(root):
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def verify_preserved(review):
    before=read(review/'BEFORE_BINDING.json')
    for directory,expected in before['preserved_directories'].items():
        if fingerprint(Path(directory))!=expected:raise ValueError('Preserved data/run/simulator changed: '+directory)
    return before


def seal_holdout_labels(root):
    manifest=read(root/'public/manifest.json')['records']
    sealed={str((root/'protected/labels'/(r['source_id']+'.json')).resolve()).lower()
            for r in manifest if r['split'] in ('test','challenge')}
    def guard(event,a):
        if event=='open' and isinstance(a[0],(str,bytes)) and str(Path(str(a[0])).resolve()).lower() in sealed:
            # Permit ONLY the frozen digest function's raw-byte checksum open.
            # It returns a SHA256 and does not decode or return file contents.
            if sys._getframe(1).f_code is not digest.__code__:
                raise PermissionError('Review forbids test/challenge label decoding; raw digest only')
    sys.addaudithook(guard)
    from astropy.io import fits
    def no_image_decode(*a,**kw):raise PermissionError('Review forbids truth/image decoding')
    fits.open=no_image_decode


def inputs(plan):
    old=Path(plan['preserved_runs'][0]);follow=Path(plan['preserved_runs'][1])
    base=read(old/'run_binding.json')['plan'];root=Path(base['dataset'])
    seal_holdout_labels(root)
    config=read('configs/compact_proxy_gl4_v3.json')
    public=NPEInputDataset(root/'public',split='validation')
    selection=[]
    for sid in sorted({r['source_id'] for r in public.records})[:2]:
        index=next(i for i,r in enumerate(public.records) if r['source_id']==sid)
        selection.append(public[index])
    return old,follow,base,root,config,selection


def verify(review,out,plan):
    before=verify_preserved(review)
    old,follow,base,root,config,selection=inputs(plan)
    cases=read(follow/'bayesian_comparison.json')['cases']
    validate_case_set(cases,[(s['bookkeeping']['source_id'],s['bookkeeping']['realization_id']) for s in selection])
    transform=load_transform(base['prepared'],root/'public')
    pca,weights=arrays(old/'full/pca.npz'),arrays(old/'full/best.npz')
    support=Support(config)
    recomputed_norm=fit_training_transform(root)
    for k in ('feature_mean_jy','feature_scale_jy','target_mean','target_scale'):
        np.testing.assert_array_equal(getattr(transform,k),getattr(recomputed_norm,k))
    xs=[]
    for batch in iter_batches(root,transform,split='train',batch_size=64):
        if set(batch['features'])!={'values'}:raise ValueError('Feature leakage')
        xs.append(batch['features']['values'])
    x=np.concatenate(xs).astype(float)
    recomputed_pca=fit_pca(x,base['pca_components'])
    for k in pca:np.testing.assert_allclose(pca[k],recomputed_pca[k],rtol=1e-10,atol=1e-10)
    del x,xs,recomputed_pca
    vx,vy=[],[]
    for batch in iter_batches(root,transform,split='validation',batch_size=64,include_targets=True):
        vx.append(batch['features']['values']);vy.append(batch['targets']['original'])
    vx,vy=np.concatenate(vx).astype(float),np.concatenate(vy)
    mu,_,_=predict(project(vx,pca),weights)
    saved_validation=arrays(follow/'validation/predictions.npz')
    np.testing.assert_array_equal(support.decode(mu),saved_validation['median'])
    np.testing.assert_array_equal(vy,saved_validation['truth'])
    results=[]
    for number,(inp,case) in enumerate(zip(selection,cases),1):
        like=CheckedLikelihood(inp,config)
        path=follow/f'bayes_{number:02d}_chains.npz'
        expected=before['preserved_directories'][str(follow)][path.name]
        if digest(path)!=expected:raise ValueError('Chain changed after before snapshot')
        archive=arrays(path);chain=archive['coordinates']
        if chain.shape!=(4,case['sampling']['draws_per_chain'],9):raise ValueError('Chain dimensions/count do not match case')
        np.testing.assert_array_equal(archive['states'],chain[:,-1])
        diag=full_diagnostics(chain,config)
        quantiles=np.quantile(physical_targets(chain,config).reshape(-1,4),[.025,.5,.975],axis=0)
        np.testing.assert_allclose(quantiles,case['bayesian_quantiles_025_50_975'],rtol=1e-12,atol=1e-10)
        # Reconstruct the original RNG stream independently from the bound code:
        # six prior reference points, then 2048 neural samples, then choice/noise.
        rng=np.random.default_rng(read('ml/followup_plan_v1.json')['seed']+number-1)
        for _ in range(6):rng.uniform(size=7);rng.normal(size=2)
        neural=support.decode(sample(project(transform.features(like.y)[None],pca),weights,2048,rng))[0]
        np.testing.assert_allclose(np.quantile(neural,[.025,.5,.975],axis=0),case['neural_quantiles_025_50_975'],rtol=1e-12,atol=1e-10)
        flat=chain.reshape(-1,9)
        indices=rng.choice(len(flat),128,replace=False)
        discrepancies=[]
        for k in indices:
            t=flat[k].copy();t[4]%=1
            residual=like.whiten(like.vector(t)-like.y)
            epsilon=rng.normal(size=len(residual))
            discrepancies.append([float(np.mean(residual**2)),float(np.mean(epsilon**2)),float(np.sqrt(np.mean((residual+epsilon)**2)))])
        stored=np.array([[r['observed_discrepancy_mean_square'],r['replicated_discrepancy_mean_square'],r['replicate_minus_observed_rms']]
                         for r in case['posterior_predictive']['draws']])
        np.testing.assert_allclose(discrepancies,stored,rtol=1e-12,atol=1e-12)
        write_npz(out/f'bayes_{number:02d}_predictive_replay.npz',posterior_flat_indices=indices,discrepancies=np.array(discrepancies))
        init=read(follow/f'bayes_{number:02d}_initialization.json')
        low,high=init['pa_branch_center']-.5,init['pa_branch_center']+.5
        if np.any((chain[:,:,4]<low)|(chain[:,:,4]>high)):raise ValueError('PA branch support mismatch')
        truth=np.array(case['truth'])  # Validation-only, after posterior inference; descriptive metric audit.
        errors=abs(quantiles[1]-truth)
        np.testing.assert_allclose(errors,case['bayesian_absolute_error_original_units'],rtol=1e-12,atol=1e-10)
        results.append({'source_id':case['source_id'],'realization_id':case['realization_id'],'chain_sha256':digest(path),
                        'recomputed_diagnostics':diag,'physical_quantiles_025_50_975':quantiles.tolist(),
                        'max_predictive_scalar_replay_difference':float(np.max(abs(np.array(discrepancies)-stored))),
                        'predictive_tail_fraction':float(np.mean(np.array(discrepancies)[:,1]>=np.array(discrepancies)[:,0])),
                        'bayesian_absolute_errors':errors.tolist(),
                        'neural_to_bayesian_95_width_ratio':((np.array(case['neural_quantiles_025_50_975'])[2]-np.array(case['neural_quantiles_025_50_975'])[0])/(quantiles[2]-quantiles[0])).tolist(),
                        'acceptance':'PASS' if diag['passed'] else 'FAIL; no validated comparison',
                        'mode_limit':'No global-mode completeness proof; initial optimizer inputs were dispersed but optimized starts often collapsed.'})
        print(f'Raw chain and all 128 predictive receipts independently verified for validation case {number}; pass={diag["passed"]}',flush=True)
    atomic_json(out/'independent_verification.json',{'cases':results,'exact_train_normalization_reproduced':True,
                'train_only_pca_reproduced':True,'validation_predictions_reproduced_exactly':True,
                'no_new_test_score':True,'units':dict(zip(TARGETS,UNITS)),'all_cases_pass':all(c['recomputed_diagnostics']['passed'] for c in results)})


def modes(review,out,plan):
    verify_preserved(review)
    _,follow,_,_,config,selection=inputs(plan)
    all_cases=[]
    for number,inp in enumerate(selection,1):
        path=out/f'mode_search_{number:02d}.json'
        if path.exists():all_cases.append(read(path));continue
        like=CheckedLikelihood(inp,config)
        rng=np.random.default_rng(plan['seed']+5000+number*10000)
        started=time.perf_counter();rows=[]
        for index in range(plan['mode_search_starts_per_validation_observation']):
            start=np.r_[rng.uniform(.05,.95,4),(index+.5)/plan['mode_search_starts_per_validation_observation'],rng.uniform(.05,.95,2),rng.normal(size=2)]
            def residual(t):
                if time.perf_counter()-started>180:raise TimeoutError('Mode-search case wall budget')
                return like.residual(t)
            tick=time.perf_counter()
            try:
                fit=least_squares(residual,start,bounds=(np.r_[np.full(7,1e-7),-np.inf,-np.inf],np.r_[np.full(7,1-1e-7),np.inf,np.inf]),
                                  max_nfev=plan['mode_search_max_nfev'],diff_step=1e-4)
                row={'index':index,'initial_coordinates':start.tolist(),'cost':float(fit.cost),'optimizer_success':bool(fit.success),
                     'nfev':int(fit.nfev),'coordinates':fit.x.tolist(),'wall_seconds':time.perf_counter()-tick}
            except TimeoutError as exc:
                rows.append({'index':index,'failure':str(exc),'wall_seconds':time.perf_counter()-tick});break
            rows.append(row)
            print(f'Mode search case {number}: start {index+1}; cost={row["cost"]:.4f}; optimizer_success={row["optimizer_success"]}',flush=True)
        chain=arrays(follow/f'bayes_{number:02d}_chains.npz')['coordinates']
        sample_indices=np.linspace(0,chain.shape[1]-1,25,dtype=int)
        sampled_logps=[]
        for t in chain[:,sample_indices].reshape(-1,9):
            physical=t.copy();physical[4]%=1
            sampled_logps.append(like.logp(physical))
        result={'source_id':inp['bookkeeping']['source_id'],'realization_id':inp['bookkeeping']['realization_id'],
                'seed':plan['seed']+5000+number*10000,'starts':rows,'elapsed_seconds':time.perf_counter()-started,
                'best_cost':min((r['cost'] for r in rows if 'cost' in r),default=None),
                'best_of_100_sampled_chain_costs':-max(sampled_logps),
                'meaning':'Measurement-only multistart optimization checks mode hypotheses; no optimizer output is a posterior or a training target. Finite starts cannot prove global mode discovery.'}
        atomic_json(path,result);all_cases.append(result)
    atomic_json(out/'mode_search.json',{'cases':all_cases,'posterior_replaced':False})


def conditional(review,out,plan):
    verify_preserved(review)
    old,_,base,root,config,selection=inputs(plan)
    transform=load_transform(base['prepared'],root/'public')
    pca,weights=arrays(old/'full/pca.npz'),arrays(old/'full/best.npz')
    support=Support(config)
    draw_path=out/'neural_outer_draws.json'
    if draw_path.exists():outer=read(draw_path)
    else:
        outer=[]
        for number,inp in enumerate(selection,1):
            rng=np.random.default_rng(plan['seed']+number*10000)
            x=project(transform.features(inp['features']['measurements'].values_jy)[None],pca)
            theta=support.decode(sample(x,weights,plan['outer_neural_draws_per_validation_observation'],rng))[0]
            outer.append({'source_id':inp['bookkeeping']['source_id'],'realization_id':inp['bookkeeping']['realization_id'],
                          'seed':plan['seed']+number*10000,'target_order':list(TARGETS),'draws':theta.tolist()})
        atomic_json(draw_path,outer)  # Frozen BEFORE any conditional inference.
    records=[]
    for number,(inp,draws) in enumerate(zip(selection,outer),1):
        like=CheckedLikelihood(inp,config)
        for j,theta in enumerate(draws['draws']):
            tag=f'conditional_{number:02d}_{j+1:02d}'
            path=out/f'{tag}_result.json'
            if path.exists():records.append(read(path));continue
            tick=time.perf_counter()
            result={'tag':tag,'source_id':draws['source_id'],'realization_id':draws['realization_id'],
                    'neural_target_draw':theta,'outer_index':j,'converged':False}
            clike=ConditionalLikelihood(like,np.array(theta))
            try:
                chain,status=run_conditional(clike,plan['conditional_sampler'],plan['seed']+number*100000+j*10000,out,tag)
                result['conditional_sampling']=status;result['converged']=status['converged']
                if status['converged']:
                    flat=chain.reshape(-1,5);rng=np.random.default_rng(plan['seed']+number*100000+j*10000+777)
                    indices=rng.choice(len(flat),plan['replicates_per_outer_draw'],replace=False)
                    rows=[]
                    for k in indices:
                        residual=clike.residual(flat[k])[:-2]
                        epsilon=rng.normal(size=len(residual))
                        rows.append([float(np.mean(residual**2)),float(np.mean(epsilon**2))])
                    write_npz(out/f'{tag}_predictive.npz',posterior_flat_indices=indices,nuisance_draws=flat[indices],
                              target_draw=np.array(theta),discrepancies=np.array(rows))
                    result['predictive_discrepancies']=rows
                    result['conditional_observed_rms_median']=float(np.median(np.sqrt(np.array(rows)[:,0])))
                    result['conditional_predictive_tail_fraction']=float(np.mean(np.array(rows)[:,1]>=np.array(rows)[:,0]))
                else:result['predictive_status']='WITHHELD: conditional convergence gate did not pass'
            except Exception as exc:
                result['failure']={'type':type(exc).__name__,'message':str(exc)}
                result['predictive_status']='WITHHELD: failed conditional solve; no target draw discarded or substituted'
            result['total_wall_seconds_including_predictive']=time.perf_counter()-tick
            result['forward_calls_including_predictive']=clike.forward_calls
            atomic_json(path,result);records.append(result)
            atomic_json(out/'neural_conditional_predictive.json',{'records':records,'all_outer_draws_included':len(records)==sum(len(r['draws']) for r in outer),
                        'all_conditionals_converged':all(r['converged'] for r in records),
                        'method':'q_NN(theta|y) p(nuisance|theta,y), with conditional Bayesian integration; not a profile plug-in or prior-independent nuisance draw',
                        'aggregate_predictive_claim':'NONE: only two fixed outer neural draws per source; failed conditionals cannot be dropped. This is a bounded implementation pilot, not a calibrated full neural predictive assessment.',
                        'limits':plan['limits']})
            print(f'{tag} finished: converged={result["converged"]}; {result["total_wall_seconds_including_predictive"]:.1f}s',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--review',type=Path,default=Path('output/ml_initial_20261023/review_01'))
    ap.add_argument('--stage',choices=['verify','modes','conditional','report'],required=True)
    ap.add_argument('--attempt',default='v2_02',choices=['v2_02'])
    args=ap.parse_args();plan=read('ml/review_plan_v2.json');out=args.review/args.attempt
    lock=Path('output/ml_initial_20261023/continuation.lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,json.dumps({'pid':os.getpid(),'purpose':'review_v2','stage':args.stage}).encode());os.close(fd)
    try:
        atomic_json(args.review/'STATUS.json',{'execution':'actively_executing','stage':args.stage,
                    'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),'scientific_limits_eliminated':False})
        sources=sorted(Path('ml').glob('*_v2.py'))+[Path('ml/review_contract.py'),Path('ml/review_plan_v2.json')]
        current={'sources':{str(p):digest(p) for p in sources},'plan':plan,'environment':versions(),
                 'before_binding_sha256':digest(args.review/'BEFORE_BINDING.json'),
                 'threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
        if args.stage=='verify':
            out.mkdir(exist_ok=False);atomic_json(out/'binding.json',current)
        elif read(out/'binding.json')!=current:raise ValueError('Review source/environment changed; preserve this attempt and create a new version')
        if args.stage=='verify':verify(args.review,out,plan)
        elif args.stage=='modes':modes(args.review,out,plan)
        elif args.stage=='conditional':conditional(args.review,out,plan)
        else:
            from report_review_v2 import build
            build(args.review,args.attempt)
        verify_preserved(args.review)
        atomic_json(out/f'status_{args.stage}.json',{'utc':datetime.now(timezone.utc).isoformat(),'execution':'complete',
                    'stage':args.stage,'preserved_archives_dataset_simulator_unchanged':True,'test_evaluation_reopened':False})
        atomic_json(args.review/'STATUS.json',{'execution':'complete' if args.stage=='report' else 'stage_complete',
                    'stage':args.stage,'utc':datetime.now(timezone.utc).isoformat(),'scientific_limits_eliminated':False})
    except Exception as exc:
        atomic_json(args.review/'STATUS.json',{'execution':'failed','stage':args.stage,'attempt':args.attempt,
                    'error_type':type(exc).__name__,'message':str(exc),'scientific_limits_eliminated':False})
        raise
    finally:lock.unlink()


if __name__=='__main__':main()
