"""Frozen local validation repair; never evaluates test/challenge labels."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from scipy.special import ndtr, ndtri
from hii_sim.io import atomic_json, write_npz, digest, versions
from hii_sim.loaders import PairedNPEDataset
from hii_sim.training_data import _target_values, load_transform
from review_inference_v2 import inputs, read, arrays, fingerprint, verify_preserved
from review_contract import CheckedLikelihood, TARGETS, UNITS, physical_targets, COORDS
from diagnostics_v2 import diagnostic_report
from model import fit_pca, project, initialize, loss_grad, predict, covariance, Support
from probit_probe import ProbitSupport, probit_prior_unit_nll
from run_initial import evaluate, utc, train
from diagnose_model import source_interval, prior_unit_nll

OUT=Path('output/ml_initial_20261023/repair_02')
PLAN=Path('ml/repair_campaign_plan.json')

def status(stage, **kw):
    atomic_json(OUT/'STATUS.json',dict(stage=stage,utc=utc(),pid=os.getpid(),**kw))

def whiten_rows(x, buckets):
    out=np.empty_like(x)
    for idx,w in buckets:out[:,idx]=np.einsum('gij,bgj->bgi',w,x[:,idx])
    return out

def compression_matrix(pca,buckets):
    b=pca['basis']/pca['scale'];c=np.empty_like(b)
    for idx,w in buckets:c[idx]=np.einsum('gji,gjk->gik',w,b[idx])
    return c

def compressed_covariance(c,buckets):
    sc=np.empty_like(c)
    for idx,w in buckets:
        l=np.linalg.inv(w);sigma=l@l.transpose(0,2,1)
        sc[idx]=np.einsum('gij,gjk->gik',sigma,c[idx])
    value=c.T@sc
    return (value+value.T)/2

def latent_to_unit(z):
    t=np.array(z,copy=True);t[...,:7]=ndtr(t[...,:7]);return t

class CompressedLikelihood:
    def __init__(self,base,c):
        self.base,self.c=base,c
        self.cov=compressed_covariance(c,base.buckets)
        self.chol=np.linalg.cholesky(self.cov)
        self.y=c.T@base.y
    def data_residual(self,z):
        delta=self.c.T@self.base.vector(latent_to_unit(z))-self.y
        return solve_triangular(self.chol,delta,lower=True,check_finite=False)
    def residual(self,z):return np.r_[self.data_residual(z),z]
    def logp(self,z):
        r=self.residual(z);return -.5*float(r@r)

def gate(z,loglike,config):
    t=latent_to_unit(z);v=t.copy();v[:,:,:4]=physical_targets(t,config)
    a=2*np.pi*t[:,:,4]
    v=np.concatenate((v,np.sin(a)[...,None],np.cos(a)[...,None],loglike[...,None]),axis=2)
    return diagnostic_report(v,COORDS+['pa_sin','pa_cos','log_likelihood'])

def fit_projected(x,y,vx,vy,plan,out):
    out.mkdir();rng=np.random.default_rng(plan['seed']);p=initialize(x.shape[1],plan['hidden'],rng)
    m,v=[{k:np.zeros_like(a) for k,a in p.items()} for _ in range(2)]
    best,stale,step=np.inf,0,0;start,cpu=time.perf_counter(),time.process_time()
    for epoch in range(1,plan['epochs']+1):
        order=rng.permutation(len(x))
        for off in range(0,len(x),plan['batch_size']):
            i=order[off:off+plan['batch_size']];loss,g=loss_grad(x[i],y[i],p,plan['weight_decay'])
            if not np.isfinite(loss) or not all(np.isfinite(a).all() for a in g.values()):raise FloatingPointError('nonfinite training')
            norm=np.sqrt(sum(np.sum(a*a) for a in g.values()));step+=1
            for k in p:
                a=g[k]*min(1,plan['gradient_clip']/max(norm,1e-30));m[k]=.9*m[k]+.1*a;v[k]=.999*v[k]+.001*a*a
                p[k]-=plan['learning_rate']*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        tl=float(loss_grad(x,y,p,gradient=False).mean());vl=float(loss_grad(vx,vy,p,gradient=False).mean())
        if not np.isfinite(tl+vl):raise FloatingPointError('nonfinite epoch')
        row=dict(epoch=epoch,train_nll=tl,validation_nll=vl,wall_seconds=time.perf_counter()-start)
        with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if vl<best-plan['min_delta']:
            best,stale,best_epoch=vl,0,epoch;write_npz(out/'best.npz',**p)
        else:stale+=1
        if stale>=plan['patience'] or time.perf_counter()-start>plan['fit_wall_seconds']:break
    info=dict(best_epoch=best_epoch,epochs=epoch,best_validation_nll=best,stopping_reason='validation_patience' if stale>=plan['patience'] else 'budget',elapsed_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,parameter_count=sum(a.size for a in p.values()),train_rows=len(x))
    atomic_json(out/'training.json',info);return arrays(out/'best.npz'),info

def prepare():
    OUT.mkdir(exist_ok=False)
    plan=read(PLAN);old,follow,base,root,config,selection=inputs(read('ml/review_plan_v2.json'))
    verify_preserved(Path('output/ml_initial_20261023/review_01'))
    status('preparing')
    before=dict(preserved_directories={str(Path('output/ml_initial_20261023/review_01')):fingerprint(Path('output/ml_initial_20261023/review_01'))},old_binding=read('output/ml_initial_20261023/review_01/BEFORE_BINDING.json'))
    atomic_json(OUT/'BEFORE_BINDING.json',before)
    atomic_json(OUT/'FROZEN_PROTOCOL.json',dict(plan=plan,plan_sha256=digest(PLAN),utc=utc(),software=versions(),source_sha256={str(p):digest(p) for p in [Path(__file__),Path('ml/model.py'),Path('ml/probit_probe.py')]},test_scored=False))
    start,cpu=time.perf_counter(),time.process_time();raw={};truth={};books={};train_input=None
    for split,count in [('train',794),('validation',134)]:
        ds=PairedNPEDataset(root,split=split);xs=[];ys=[];bs=[]
        for i in range(len(ds)):
            inp,label=ds[i]
            if train_input is None:train_input=inp
            xs.append(inp['features']['measurements'].values_jy);ys.append(_target_values(label));bs.append(inp['bookkeeping'])
        raw[split]=np.asarray(xs,dtype=np.float64);truth[split]=np.array(ys);books[split]=bs
        if len(xs)!=count:raise ValueError('split size')
    if set(b['source_id'] for b in books['train'])&set(b['source_id'] for b in books['validation']):raise ValueError('source overlap')
    like=CheckedLikelihood(train_input,config);mean=raw['train'].mean(axis=0)
    wx=whiten_rows(raw['train']-mean,like.buckets);wv=whiten_rows(raw['validation']-mean,like.buckets)
    pca=fit_pca(wx,plan['representation']['components']);x,vx=project(wx,pca),project(wv,pca)
    c=compression_matrix(pca,like.buckets);cov=compressed_covariance(c,like.buckets)
    np.testing.assert_allclose(cov,np.diag(1/pca['scale']**2),atol=1e-10,rtol=1e-9)
    np.testing.assert_allclose(vx,(raw['validation']-mean)@c-pca['mean']@(pca['basis']/pca['scale']),atol=1e-9)
    write_npz(OUT/'representation.npz',raw_mean=mean,compression=c,conditional_thermal_covariance=cov,**pca)
    write_npz(OUT/'development_cache.npz',train_raw=raw['train'],validation_raw=raw['validation'],train_targets=truth['train'],validation_targets=truth['validation'],train_features=x,validation_features=vx)
    atomic_json(OUT/'bookkeeping.json',books)
    prep=dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,pca_explained_fraction=float(pca['explained_fraction']),training_observations=794,training_sources=397,validation_observations=134,validation_sources=67,feature_contract='Measured Jy only; metadata defines fixed known thermal whitening. Labels/bookkeeping stored separately and never fed to predictor.',covariance_max_off_diagonal=float(np.max(abs(cov-np.diag(np.diag(cov))))))
    atomic_json(OUT/'preparation.json',prep);status('prepared');print(json.dumps(prep),flush=True)

def neural():
    plan=read(PLAN);old,follow,base,root,config,_=inputs(read('ml/review_plan_v2.json'));d=arrays(OUT/'development_cache.npz');books=read(OUT/'bookkeeping.json')
    status('training_noise_pca128');support=ProbitSupport(config);fitplan=dict(plan['neural'],seed=plan['model_seed'])
    p,info=fit_projected(d['train_features'],support.encode(d['train_targets']),d['validation_features'],support.encode(d['validation_targets']),fitplan,OUT/'neural')
    vd=OUT/'validation';vd.mkdir();identity=dict(mean=np.zeros(128),basis=np.eye(128),scale=np.ones(128))
    metrics,med=evaluate(d['validation_features'],d['validation_targets'],books['validation'],identity,p,support,fitplan,vd)
    (vd/'test_predictions.npz').rename(vd/'predictions.npz');(vd/'test_evaluation.json').unlink()
    metrics.update(split='reused validation, development only',posterior_family='Full covariance probit-normal with thermal PCA128',loss_unit='Nats in prior probit coordinates',definitions=metrics['definitions'].replace('heldout set','reused validation set'))
    metrics['mean_nll_probit_coordinates']=metrics.pop('mean_nll_scaled_logit_coordinates');atomic_json(vd/'metrics.json',metrics)
    transform=load_transform(base['prepared'],root/'public');vx=transform.features(d['validation_raw']).astype(float)
    mu,a,_=predict(d['validation_features'],p);joint,marg=probit_prior_unit_nll(d['validation_targets'],mu,a,support)
    parent=np.array([b['source_id'] for b in books['validation']]);groups=sorted(set(parent));boot=np.random.default_rng(plan['seed']).integers(0,len(groups),(5000,len(groups)))
    before={};prior=support.decode(np.zeros((1,4)))[0];err=abs(med-d['validation_targets'])
    for label,cp,pc,sp in [('original',old/'full/best.npz',old/'full/pca.npz',Support(config)),('probit32',Path('output/ml_initial_20261023/review_01/probit_probe_01/full/best.npz'),old/'full/pca.npz',support)]:
        m,b,_=predict(project(vx,arrays(pc)),arrays(cp));bm=sp.decode(m);be=abs(bm-d['validation_targets']);bj,bmar=(prior_unit_nll if label=='original' else probit_prior_unit_nll)(d['validation_targets'],m,b,sp)
        before[label]=dict(mae=be.mean(axis=0).tolist(),joint_prior_unit_nll=float(bj.mean()),marginal_prior_unit_nll=bmar.mean(axis=0).tolist(),mae_change_new_minus_before=(err-be).mean(axis=0).tolist(),mae_change_source_bootstrap95=source_interval(err-be,parent,boot).tolist(),joint_nll_change=float((joint-bj).mean()),joint_nll_change_source_bootstrap95=source_interval(joint-bj,parent,boot).tolist())
    var=np.diagonal(covariance(a),axis1=1,axis2=2);kl=.5*(mu*mu+var-1-np.log(var));lo=support.decode(mu-1.95996398454*np.sqrt(var));hi=support.decode(mu+1.95996398454*np.sqrt(var));prior_width=support.decode(np.full((1,4),1.95996398454))[0]-support.decode(np.full((1,4),-1.95996398454))[0]
    flags=(kl<.1)&((hi-lo)/prior_width>.8)
    write_npz(vd/'prior_dominance.npz',marginal_kl_to_prior_nats=kl,prior95_width_ratio=(hi-lo)/prior_width,prior_dominated=flags)
    timing=[]
    from model import sample
    for _ in range(10):
        tick=time.perf_counter();support.decode(sample(d['validation_features'][:1],p,2048,np.random.default_rng(1)));timing.append(time.perf_counter()-tick)
    result=dict(targets=TARGETS,units=UNITS,training=info,before=before,mae=err.mean(axis=0).tolist(),prior_median_mae=abs(prior-d['validation_targets']).mean(axis=0).tolist(),joint_prior_unit_nll=float(joint.mean()),marginal_prior_unit_nll=marg.mean(axis=0).tolist(),prior_dominated_counts=flags.sum(axis=0).tolist(),te_rule='Keep all four posterior outputs. Flag KL(q_u||uniform)<0.1 nats AND central95 width/prior95 width>0.8; heuristic informational abstention from a data-constrained Te claim. Unflagged neural Te precision is also unvalidated. Never fix Te or substitute a point prior.',inference_2048_draws_seconds_median=float(np.median(timing)),inference_timing_scope='Already projected single observation; no disk, raw whitening or PCA; neural posterior only, no nuisance draws',checkpoint_sha256=digest(OUT/'neural/best.npz'),test_scored=False)
    atomic_json(OUT/'neural_comparison.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(s) for s in (OUT/'neural/epochs.jsonl').read_text().splitlines()];fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    for key,label in [('train_nll','Training'),('validation_nll','Validation')]:ax.plot([r['epoch'] for r in rows],[r[key] for r in rows],label=label)
    ax.axvline(info['best_epoch'],ls=':',c='k');ax.set(xlabel='Epoch',ylabel='NLL [prior probit coordinates]',title='Frozen thermal PCA128 repair; reused validation');ax.legend();fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    status('neural_complete');print(json.dumps(result),flush=True)

def sample_case(like,config,settings,seed,dest):
    start,cpu=time.perf_counter(),time.process_time();rngs=[np.random.default_rng(seed+1009*k) for k in range(4)];maps=[]
    def timed_residual(z):
        if time.perf_counter()-start>settings['case_wall_seconds'] or datetime.now(timezone.utc)>=datetime.fromisoformat(read(PLAN)['science_stop_utc']):raise TimeoutError('fixed scientific budget')
        return like.residual(z)
    for k,rng in enumerate(rngs):
        z=rng.normal(0,.8,9);z[4]=ndtri((k+.5)/4)
        fit=least_squares(timed_residual,z,max_nfev=settings['map_max_nfev'],diff_step=1e-4)
        maps.append(dict(z=fit.x.tolist(),cost=float(fit.cost),success=bool(fit.success),jac=fit.jac))
    best=min(maps,key=lambda m:m['cost']);center=np.array(best['z']);h=best['jac'].T@best['jac'];chol=np.linalg.cholesky(np.linalg.inv(h))*np.sqrt(settings['proposal_covariance_inflation']);inv=np.linalg.inv(chol);df=settings['student_df']
    def logq(z):
        v=inv@(z-center);return -.5*(df+9)*np.log1p(v@v/df)
    states=np.array([m['z'] for m in maps]);lp=np.array([like.logp(z) for z in states]);draws=[];lls=[];acc=np.zeros((4,2),int);attempt=np.zeros((4,2),int);diag=None
    atomic_json(dest.with_suffix('.initialization.json'),dict(seed=seed,maps=[{k:v for k,v in m.items() if k!='jac'} for m in maps],center=center.tolist(),proposal_chol=chol.tolist(),prior='z_first7=Phi^-1(u), all z9 standard normal; exact Jacobians included; PA full [0,180] period',settings=settings))
    for step in range(settings['warmup']+settings['max_draws_per_chain']):
        for k,rng in enumerate(rngs):
            kind=int(rng.uniform()>=settings['independence_probability']);attempt[k,kind]+=1
            proposal=(center+chol@rng.normal(size=9)/np.sqrt(rng.chisquare(df)/df) if kind==0 else states[k]+settings['local_scale']*chol@rng.normal(size=9))
            value=like.logp(proposal);ratio=value-lp[k]+(logq(states[k])-logq(proposal) if kind==0 else 0.)
            if np.log(rng.uniform())<ratio:states[k]=proposal;lp[k]=value;acc[k,kind]+=1
        if step>=settings['warmup']:
            draws.append(states.copy());lls.append(lp+.5*np.sum(states*states,axis=1))
        n=len(draws)
        budget=time.perf_counter()-start>settings['case_wall_seconds'] or datetime.now(timezone.utc)>=datetime.fromisoformat(read(PLAN)['science_stop_utc']) if step%100==0 else False
        if n and (n%settings['diagnostics_every']==0 or budget or step==settings['warmup']+settings['max_draws_per_chain']-1):
            chain=np.array(draws).transpose(1,0,2);loglike=np.array(lls).T
            diag=gate(chain,loglike,config) if n>=20 else None
            write_npz(dest.with_suffix('.chains.npz'),latent_coordinates=chain,coordinates=latent_to_unit(chain),log_likelihood=loglike,states=states)
            result=dict(passed=bool(diag and diag['passed']),diagnostics=diag,draws_per_chain=n,warmup=settings['warmup'],wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,acceptance=(acc/np.maximum(attempt,1)).tolist(),seed=seed,budget_hit=budget,rng_states=[r.bit_generator.state for r in rngs],posterior_comparison_allowed=bool(diag and diag['passed']))
            atomic_json(dest.with_suffix('.json'),result);print(json.dumps(dict(case=dest.name,draws=n,passed=result['passed'],wall_seconds=result['wall_seconds'])),flush=True)
            if result['passed'] or budget:break
        if budget and not n:raise TimeoutError('budget before production; initialization retained')
    return result

def bayesian(route):
    plan=read(PLAN);old,follow,base,root,config,selection=inputs(read('ml/review_plan_v2.json'));settings=dict(plan['bayesian']);settings.update(case_wall_seconds=settings['wall_seconds_per_case'],max_draws_per_chain=settings['max_draws'],diagnostics_every=settings['diagnostic_every'],proposal_covariance_inflation=settings['proposal_inflation']);out=OUT/'compressed_bayes';out.mkdir(exist_ok=True)
    if route=='old32':
        transform=load_transform(base['prepared'],root/'public');pca=arrays(old/'full/pca.npz');c=pca['basis']/pca['scale']/transform.feature_scale_jy[:,None]
    else:c=arrays(OUT/'representation.npz')['compression']
    for number,inp in enumerate(selection,1):
        dest=out/f'{route}_{number:02d}'
        if dest.with_suffix('.json').exists():continue
        if datetime.now(timezone.utc)>=datetime.fromisoformat(plan['science_stop_utc']):break
        status('compressed_bayes',route=route,case=number);base_like=CheckedLikelihood(inp,config);like=CompressedLikelihood(base_like,c)
        check=np.r_[np.full(7,.5),0.,0.];error=float(np.linalg.norm(base_like.whiten(base_like.vector(check)-base_like.vector(check,reference=True))))
        if error>1e-5:raise ValueError('renderer reference mismatch')
        write_npz(dest.with_suffix('.operator.npz'),compression=c,covariance=like.cov)
        atomic_json(dest.with_suffix('.observation.json'),dict(bookkeeping=inp['bookkeeping'],route=route,forward_reference_error=error,noise='C^T Sigma_thermal C; same explicit calibration priors, all source/nuisance priors and forward model; offset cancels; same observed y compressed linearly',distinction='Matched to ideal float64 compression. Historical NPE used float32 normalized inputs; rounding difference recorded separately. Full-y posterior is a separate information experiment.'))
        try:sample_case(like,config,settings,plan['seed']+10000*(1 if route=='old32' else 2)+number,dest)
        except TimeoutError as exc:atomic_json(dest.with_suffix('.failure.json'),dict(error=str(exc),posterior_comparison_allowed=False));print(str(exc),flush=True)
    status('compressed_bayes_route_complete',route=route)

def half():
    plan=read(PLAN);old,_,base,root,config,_=inputs(read('ml/review_plan_v2.json'));d=arrays(OUT/'development_cache.npz');books=read(OUT/'bookkeeping.json');parents=np.array([b['source_id'] for b in books['train']]);keep=np.isin(parents,sorted(set(parents))[:198]);raw=d['train_raw'][keep];mean=raw.mean(axis=0);scale=raw.std(axis=0)
    if np.any(scale<=0):raise ValueError('Constant half-data feature requires explicit policy')
    x=((raw-mean)/scale).astype(np.float32).astype(float);vx=((d['validation_raw']-mean)/scale).astype(np.float32).astype(float)
    out=OUT/'half_corrected';out.mkdir();write_npz(out/'normalization.npz',feature_mean_jy=mean,feature_scale_jy=scale)
    unique=[np.flatnonzero(parents==sid)[0] for sid in sorted(set(parents[keep]))];ty=d['train_targets'][unique].copy();ty[:,[0,2]]=np.log10(ty[:,[0,2]])
    write_npz(out/'target_statistics.npz',target_mean=ty.mean(axis=0),target_scale=ty.std(axis=0))
    support=Support(config);fp=dict(base);fp['seed']=231026
    _,_,info=train('full',x,support.encode(d['train_targets'][keep]),vx,support.encode(d['validation_targets']),fp,out)
    atomic_json(out/'scope.json',dict(observations=int(keep.sum()),sources=198,own_half_feature_normalization=True,own_half_pca=True,prior_fixed_target_transform=True,target_standardization_used_by_model=False,selection='first198 sorted training source IDs, no validation/test selection',limits='One half/full point at one seed is not a data sufficiency finding',training=info))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','neural','old32','noise128','half']);a=ap.parse_args()
    lock=Path('output/ml_initial_20261023/continuation.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,f'repair02 {a.stage} pid={os.getpid()}'.encode());os.close(fd)
    try:
        if a.stage=='prepare':prepare()
        elif a.stage=='neural':neural()
        elif a.stage=='half':half()
        else:bayesian(a.stage)
    except Exception as exc:
        if OUT.exists():status('failed',action=a.stage,error=repr(exc))
        raise
    finally:lock.unlink()

if __name__=='__main__':main()
