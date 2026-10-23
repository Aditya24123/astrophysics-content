"""One predeclared validation-only prior-correct target-transform probe."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtri,ndtr
from hii_sim.io import atomic_json,digest,write_npz,versions
from hii_sim.training_data import load_transform
from model import Support,project,predict,covariance,loss_grad
from run_initial import train,load_split,evaluate
from review_inference_v2 import inputs,verify_preserved,read,arrays
from review_contract import TARGETS,UNITS
from diagnose_model import prior_unit_nll,source_interval


class ProbitSupport(Support):
    def unit(self,original):
        x=np.array(original,float,copy=True);x[...,[0,2]]=np.log10(x[...,[0,2]])
        return (x-self.lower)/(self.upper-self.lower)
    def encode(self,original):
        u=self.unit(original)
        if not np.isfinite(u).all() or np.any((u<=0)|(u>=1)):raise ValueError('Targets must be strictly inside prior support')
        return ndtri(u)
    def decode(self,z):
        x=self.lower+(self.upper-self.lower)*ndtr(np.asarray(z))
        x[...,[0,2]]=10.**x[...,[0,2]]
        return x


def probit_prior_unit_nll(truth,mu,a,support):
    z=support.encode(truth);v=np.diagonal(covariance(a),axis1=1,axis2=2)
    jac=.5*z*z+.5*np.log(2*np.pi)
    marginal=.5*((z-mu)**2/v+np.log(2*np.pi*v))-jac
    joint=.5*np.sum(np.einsum('bij,bj->bi',a,z-mu)**2,axis=1)-np.log(np.diagonal(a,axis1=1,axis2=2)).sum(axis=1)+2*np.log(2*np.pi)-jac.sum(axis=1)
    return joint,marginal


def execute(review):
    verify_preserved(review)
    plan=read('ml/probit_probe_plan.json');out=review/'probit_probe_01';out.mkdir(exist_ok=False)
    old,_,base,root,config,_=inputs(read('ml/review_plan_v2.json'))
    binding={'protocol':plan,'protocol_sha256':digest('ml/probit_probe_plan.json'),'created_before_training':True,
             'frozen_utc':datetime.now(timezone.utc).isoformat(),
             'source_sha256':{p:digest(p) for p in ['ml/probit_probe.py','ml/model.py','ml/run_initial.py','ml/diagnose_model.py']},
             'environment':versions(),'frozen_original_checkpoint_sha256':digest(old/'full/best.npz'),
             'normalization_sha256':digest(Path(base['prepared'])/'normalization.npz'),
             'new_test_evaluation':False,'fit_count':1,'model_replaced':False}
    atomic_json(out/'FROZEN_PROTOCOL.json',binding)
    start,cpu=time.perf_counter(),time.process_time()
    transform=load_transform(base['prepared'],root/'public')
    x,y,books=load_split(root,transform,'train');vx,vy,vbooks=load_split(root,transform,'validation')
    if len(x)!=794 or len(vx)!=134 or {b['source_id'] for b in books}&{b['source_id'] for b in vbooks}:raise ValueError('Source split violation')
    original_support=Support(config);support=ProbitSupport(config)
    pca,p,info=train('full',x,support.encode(y),vx,support.encode(vy),plan,out)
    for k,v in arrays(old/'full/pca.npz').items():np.testing.assert_array_equal(pca[k],v)
    valdir=out/'validation';valdir.mkdir()
    metrics,median=evaluate(vx,vy,vbooks,pca,p,support,plan,valdir)
    (valdir/'test_predictions.npz').rename(valdir/'predictions.npz')
    metrics['split']='validation development; not a new blind test'
    metrics['posterior_family']='Full covariance probit-normal; exact prior representable at zero mean/identity covariance; still a restricted Gaussian conditional family'
    metrics['loss_unit']='Nats in probit prior coordinates; common prior-unit density scores/Jacobians are saved in comparison.json'
    atomic_json(valdir/'metrics.json',metrics);(valdir/'test_evaluation.json').unlink()
    parents=np.array([b['source_id'] for b in vbooks]);groups=sorted(set(parents));rng=np.random.default_rng(231031)
    bootstrap=rng.integers(0,len(groups),size=(plan['bootstrap_parent_draws'],len(groups)))
    old_pca,old_p=arrays(old/'full/pca.npz'),arrays(old/'full/best.npz')
    old_mu,old_a,_=predict(project(vx,old_pca),old_p);old_median=original_support.decode(old_mu)
    mu,a,_=predict(project(vx,pca),p)
    old_joint,old_marginal=prior_unit_nll(vy,old_mu,old_a,original_support)
    joint,marginal=probit_prior_unit_nll(vy,mu,a,support)
    prior=original_support.decode(np.zeros((1,4)))[0]
    before_error,after_error,prior_error=abs(old_median-vy),abs(median-vy),abs(prior-vy)
    changed_ci=source_interval(after_error-before_error,parents,bootstrap)
    prior_ci=source_interval(after_error-prior_error,parents,bootstrap)
    density_ci=source_interval(joint-old_joint,parents,bootstrap)
    parameters={name:{'unit':UNITS[j],'original_mae':float(before_error[:,j].mean()),'probe_mae':float(after_error[:,j].mean()),
                       'prior_median_mae':float(prior_error[:,j].mean()),'probe_rmse':float(np.sqrt(np.mean((median[:,j]-vy[:,j])**2))),
                       'mae_change_vs_original':float(np.mean(after_error[:,j]-before_error[:,j])),
                       'mae_change_vs_original_source_resampling95':changed_ci[:,j].tolist(),
                       'mae_change_vs_prior_source_resampling95':prior_ci[:,j].tolist(),
                       'original_marginal_prior_unit_nll':float(old_marginal[:,j].mean()),
                       'probe_marginal_prior_unit_nll':float(marginal[:,j].mean()),
                       'coverage_and_width':metrics['parameters'][name]['coverage']}
                for j,name in enumerate(TARGETS)}
    result={'parameters':parameters,'original_joint_prior_unit_nll':float(old_joint.mean()),'probe_joint_prior_unit_nll':float(joint.mean()),
            'joint_density_change_probe_minus_original':float(np.mean(joint-old_joint)),'joint_density_change_source_resampling95':density_ci.tolist(),
            'validation_observations':134,'validation_sources':67,'training_observations':794,'training_sources':397,
            'selected_epoch':info['best_epoch'],'stopped_epoch':info['epochs'],'training_info':info,
            'wall_seconds_with_loading_and_evaluation':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
            'test_evaluated':False,'original_model_replaced':False,'probe_fits':1,
            'decision':'One development sensitivity probe only; no automatic promotion or blind generalization claim. Report all parameter/density/coverage changes. This changes prior representability, not raw data information or family flexibility.',
            'statistical_limit':'Paired source-resampling intervals are descriptive on reused validation/Sobol sources; they omit selection uncertainty. Small changes are not guaranteed population improvements.'}
    atomic_json(out/'comparison.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(s) for s in (out/'full/epochs.jsonl').read_text().splitlines()]
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    ax.plot([r['epoch'] for r in rows],[r['train_nll'] for r in rows],label='Train')
    ax.plot([r['epoch'] for r in rows],[r['validation_nll'] for r in rows],label='Validation')
    ax.axvline(info['best_epoch'],color='black',ls=':',label='Selected epoch')
    ax.set(xlabel='Epoch',ylabel='Gaussian NLL [probit coordinates]',title='Single predeclared probit-prior sensitivity probe');ax.legend()
    fig.savefig(out/'training_curves.png',dpi=180);plt.close(fig)
    lines=['# Single frozen-budget probit target-transform probe','',
           '**Development comparison only. The original model remains the comparison model; no test scoring or automatic replacement.**', '',
           'Only the target transform changed: probit(prior-unit) replaces scaled logit(prior-unit). The same 794 training '
           'observations /397 sources, 134 validation observations /67 sources, PCA32, hidden32, seed 231026, optimizer and '
           'stopping protocol were used. A Gaussian probit posterior can exactly represent the uninformative uniform prior '
           'at zero mean/identity covariance; it still cannot represent arbitrary multimodal or curved conditionals. '
           'PCA arrays reproduced exactly. Protocol and source hashes were saved BEFORE training.', '',
           '| Target | Unit | Frozen model MAE | Probit probe MAE | Prior median MAE | Probe RMSE | MAE change vs frozen, source-resampling 95% interval |',
           '|---|---|---:|---:|---:|---:|---|']
    for name,r in parameters.items():
        lo,hi=r['mae_change_vs_original_source_resampling95']
        lines.append(f'| {name} | {r["unit"]} | {r["original_mae"]:.6g} | {r["probe_mae"]:.6g} | {r["prior_median_mae"]:.6g} | {r["probe_rmse"]:.6g} | {r["mae_change_vs_original"]:.6g} [{lo:.6g}, {hi:.6g}] |')
    lines+=['',f'Joint density NLL in common prior-unit coordinates (uniform prior=0): frozen {old_joint.mean():.6g}; '
            f'probe {joint.mean():.6g}; change {np.mean(joint-old_joint):.6g} with source-resampling interval {density_ci.tolist()}. '
            'Label Jacobians were included, so different target coordinates are not compared by raw Gaussian NLL.', '',
            '| Target | Level | Coverage | Mean width in stated physical units |','|---|---:|---:|---:|']
    for name,r in parameters.items():
        for level,c in r['coverage_and_width'].items():lines.append(f'| {name} | {float(level):.0%} | {c["observed"]:.2%} | {c["mean_interval_width_original_units"]:.6g} |')
    lines+=['',f'Selected epoch {info["best_epoch"]}; stopped epoch {info["epochs"]}. These are selected on reused validation '
            'sources; coverage and paired resampling intervals are descriptive development evidence. Small error changes '
            'can reflect statistical uncertainty. No heldout test comparison, new protected set, hyperparameter sweep or '
            'second score-driven fit was performed.', '', '![Probe curves](training_curves.png)', '',
            result['decision'], '', 'The frozen original neural outer predictive draws and all conditional pilot/continuation '
            'evidence remain untouched; this probe has not undergone that full conditional predictive experiment.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verify_preserved(review)
    print(str(out)+': '+str({k:(v['original_mae'],v['probe_mae'],v['prior_median_mae']) for k,v in parameters.items()}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--review',type=Path,default=Path('output/ml_initial_20261023/review_01'));args=ap.parse_args()
    lock=Path('output/ml_initial_20261023/continuation.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,b'single_probit_probe');os.close(fd)
    try:
        atomic_json(args.review/'STATUS.json',{'execution':'actively_executing','stage':'single_probit_probe','pid':os.getpid()})
        execute(args.review)
        atomic_json(args.review/'STATUS.json',{'execution':'stage_complete','stage':'single_probit_probe','model_replaced':False})
    except Exception as exc:
        atomic_json(args.review/'STATUS.json',{'execution':'failed','stage':'single_probit_probe','error':str(exc)});raise
    finally:lock.unlink()
