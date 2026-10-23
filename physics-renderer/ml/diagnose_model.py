"""Validation-only decomposition of prior dominance, density/point loss and PCA."""
import argparse
import os
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from scipy.special import expit
from hii_sim.io import atomic_json,digest
from hii_sim.training_data import load_transform,iter_batches
from model import Support,project,predict,sample,covariance,loss_grad,LOGIT_SCALE
from review_contract import CheckedLikelihood,TARGETS,UNITS,full_diagnostics
from review_inference_v2 import inputs,verify_preserved,read,arrays


def source_interval(values,parents,bootstrap):
    grouped=np.array([values[parents==g].mean(axis=0) for g in sorted(set(parents))])
    return np.quantile(grouped[bootstrap].mean(axis=1),[.025,.975],axis=0)


def prior_unit_nll(original,mu,precision,support):
    z=support.encode(original)
    prior_z=np.sum(-np.log(LOGIT_SCALE)+np.logaddexp(0,z*LOGIT_SCALE)+np.logaddexp(0,-z*LOGIT_SCALE),axis=1)
    covariance_matrix=covariance(precision)
    variance=np.diagonal(covariance_matrix,axis1=1,axis2=2)
    marginal_z=.5*((z-mu)**2/variance+np.log(2*np.pi*variance))
    marginal_prior_z=-np.log(LOGIT_SCALE)+np.logaddexp(0,z*LOGIT_SCALE)+np.logaddexp(0,-z*LOGIT_SCALE)
    # Exact prior is uniform u. The label Jacobian cancels in NN-vs-prior differences.
    joint_z=.5*np.sum(np.einsum('bij,bj->bi',precision,z-mu)**2,axis=1)-np.log(np.diagonal(precision,axis1=1,axis2=2)).sum(axis=1)+2*np.log(2*np.pi)
    return joint_z-prior_z,marginal_z-marginal_prior_z


def local_information(like,t,transform,pca,step):
    columns=[]
    for j in range(9):
        h=min(step,.4*t[j],.4*(1-t[j])) if j in (0,1,2,3,5,6) else step
        if h<=0:raise ValueError('Local information state at prior boundary')
        a,b=t.copy(),t.copy();a[j]+=h;b[j]-=h
        if j==4:a[j]%=1;b[j]%=1
        columns.append((like.vector(a)-like.vector(b))/(2*h))
    jac=np.stack(columns,axis=1)
    white=np.stack([like.whiten(jac[:,j]) for j in range(9)],axis=1)
    compression=pca['basis']/pca['scale'][None,:]/transform.feature_scale_jy[:,None]
    cov_times=np.empty_like(compression)
    for idx,inverse in like.buckets:
        l=np.linalg.inv(inverse);c=l@l.transpose(0,2,1)
        cov_times[idx]=np.einsum('bij,bjk->bik',c,compression[idx])
    compressed_cov=compression.T@cov_times
    compressed_jac=solve_triangular(cholesky(compressed_cov,lower=True),compression.T@jac,lower=True)
    prior=np.zeros((2,9));prior[0,7]=1;prior[1,8]=1
    def effective(matrix):
        augmented=np.r_[matrix,prior]
        residual=augmented[:,:4]-augmented[:,4:]@np.linalg.lstsq(augmented[:,4:],augmented[:,:4],rcond=None)[0]
        return residual.T@residual
    f,g=white.T@white,compressed_jac.T@compressed_jac
    fe,ge=effective(white),effective(compressed_jac)
    ratios=np.diag(g)[:4]/np.maximum(np.diag(f)[:4],1e-30)
    eff_ratios=np.diag(ge)/np.maximum(np.diag(fe),1e-30)
    if np.any(ratios>1+1e-5) or np.any(eff_ratios>1+1e-5):raise ValueError('Information retention violates linear data-processing bound')
    correlation=fe/np.sqrt(np.outer(np.diag(fe),np.diag(fe)))
    return {'prior_unit_coordinates':t.tolist(),'conditional_information_retention':ratios.tolist(),
            'nuisance_projected_information_retention':eff_ratios.tolist(),
            'full_nuisance_projected_information':fe.tolist(),'compressed_nuisance_projected_information':ge.tolist(),
            'full_information_correlation':correlation.tolist(),
            'compressed_noise_condition_number':float(np.linalg.cond(compressed_cov)),
            'meaning':'Local derivatives at one existing joint posterior state; exact compressed conditional thermal covariance, calibration-prior curvature included in nuisance projection. Other physical targets held fixed for each diagonal. Uniform bounds/nonlinear global geometry and float32 rounding are not Fisher information.'}


def execute(review):
    verify_preserved(review)
    out=review/'model_diagnosis_01';out.mkdir(exist_ok=False)
    plan=read('ml/model_diagnosis_plan.json');parentplan=read('ml/review_plan_v2.json')
    old,follow,base,root,config,selection=inputs(parentplan)
    atomic_json(out/'binding.json',{'plan':plan,'plan_sha256':digest('ml/model_diagnosis_plan.json'),
                'code_sha256':digest(Path(__file__)),'model_sha256':digest(old/'full/best.npz'),
                'pca_sha256':digest(old/'full/pca.npz'),'source_role':'validation development only; no new model/test selection'})
    transform=load_transform(base['prepared'],root/'public')
    pca,weights=arrays(old/'full/pca.npz'),arrays(old/'full/best.npz');support=Support(config)
    x=[];truth=[];parents=[]
    for batch in iter_batches(root,transform,split='validation',include_targets=True,batch_size=64):
        x.append(batch['features']['values']);truth.append(batch['targets']['original']);parents += [b['source_id'] for b in batch['bookkeeping']]
    x,truth,parents=np.concatenate(x).astype(float),np.concatenate(truth),np.array(parents)
    mu,a,_=predict(project(x,pca),weights);median=support.decode(mu)
    prior=support.decode(np.zeros((1,4)))[0]
    errors=abs(median-truth);prior_errors=abs(prior-truth);difference=errors-prior_errors
    joint_score,marginal_score=prior_unit_nll(truth,mu,a,support)
    rng=np.random.default_rng(plan['seed']);n=len(set(parents));bootstrap=rng.integers(0,n,size=(plan['bootstrap_source_resamples'],n))
    error_ci=source_interval(difference,parents,bootstrap);score_ci=source_interval(marginal_score,parents,bootstrap)
    parameter_metrics={name:{'unit':UNITS[j],'neural_mae':float(errors[:,j].mean()),'prior_median_mae':float(prior_errors[:,j].mean()),
                            'mae_difference_neural_minus_prior':float(difference[:,j].mean()),'paired_source_resampling_95_interval':error_ci[:,j].tolist(),
                            'marginal_nll_prior_unit_vs_uniform_prior':float(marginal_score[:,j].mean()),
                            'marginal_nll_difference_source_resampling_95_interval':score_ci[:,j].tolist()}
                       for j,name in enumerate(TARGETS)}
    cases=[]
    for number,inp in enumerate(selection,1):
        like=CheckedLikelihood(inp,config)
        chain=arrays(follow/f'bayes_{number:02d}_chains.npz')['coordinates']
        if not full_diagnostics(chain,config)['passed']:raise ValueError('Comparator not accepted')
        unit_bayes=chain[:,:,:4].reshape(-1,4)
        rng=np.random.default_rng(plan['seed']+number*1009)
        z=sample(project(transform.features(like.y)[None],pca),weights,plan['neural_comparator_draws'],rng)[0]
        unit_neural=expit(z*LOGIT_SCALE)
        descriptions={}
        for j,name in enumerate(TARGETS):
            rows={}
            for label,draws in [('bayesian',unit_bayes[:,j]),('neural',unit_neural[:,j])]:
                counts=np.histogram(draws,bins=np.linspace(0,1,plan['histogram_bins']+1))[0];p=counts/counts.sum();nonzero=p>0
                rows[label]={'descriptive_histogram_KL_from_uniform_prior_nats':float(np.sum(p[nonzero]*np.log(p[nonzero]*plan['histogram_bins']))),
                             'central95_prior_unit_width':float(np.diff(np.quantile(draws,[.025,.975]))[0]),
                             'histogram_counts':counts.tolist()}
            descriptions[name]=rows
        flat=chain.reshape(-1,9);info=[]
        for fraction in plan['local_information_chain_fractions']:
            t=flat[int(fraction*len(flat))].copy();t[4]%=1
            info.append(local_information(like,t,transform,pca,plan['finite_difference_step_prior_units']))
        cases.append({'source_id':inp['bookkeeping']['source_id'],'posterior_prior_comparison':descriptions,'local_information':info})
    result={'parameters':parameter_metrics,'validation_observations':len(truth),'independent_validation_sources':n,
            'mean_joint_nll_prior_unit_vs_uniform_prior':float(joint_score.mean()),
            'joint_nll_difference_source_resampling_95_interval':source_interval(joint_score,parents,bootstrap).tolist(),
            'cases':cases,'model_changed':False,'test_used':False,'bootstrap_limit':plan['interpretation']}
    atomic_json(out/'diagnosis.json',result)
    lines=['# Frozen neural model: validation diagnosis','',
           '**Prominent unresolved result:** temperature and axis-ratio validation MAEs exceed their prior-median diagnostics. '
           'The source-paired uncertainty below determines how strongly this sample supports that difference. These '
           'validation sources were used for early stopping; this is development evidence, not fresh blind inference.', '',
           '| Target | Unit | Neural MAE | Prior median MAE | Difference | Paired source-resampling 95% interval for difference | Marginal density NLL difference from prior |',
           '|---|---|---:|---:|---:|---|---:|']
    for name,r in parameter_metrics.items():
        lo,hi=r['paired_source_resampling_95_interval']
        lines.append(f'| {name} | {r["unit"]} | {r["neural_mae"]:.6g} | {r["prior_median_mae"]:.6g} | {r["mae_difference_neural_minus_prior"]:.6g} | [{lo:.6g}, {hi:.6g}] | {r["marginal_nll_prior_unit_vs_uniform_prior"]:.6g} |')
    lines+=['',f'Joint density NLL difference from the exact uniform prior in common prior-unit coordinates: {joint_score.mean():.6g} nats '
            '(negative favors the neural density). All label Jacobians are included/cancelled correctly; original physical units '
            'are not mixed in one point-error score. Proper density loss does not directly optimize the realized sample MAE, '
            'but an exact posterior median is the Bayes action for expected absolute error. Thus loss/point-summary mismatch '
            'is not a justification for calling poorer point estimates accurate.', '',
            '## Prior dominance and PCA information','',
            'The following KL values are descriptive 20-bin histograms, not hypothesis tests; conventional samples are correlated. '
            'Small values and prior-like interval widths suggest weak information in those observations, not global '
            'non-identifiability. Local Fisher retention uses the FULL correlated thermal covariance transformed through PCA; '
            'it is not merely variance retention and is not a global sufficiency proof.', '',
            '| Case / target | Bayes KL vs prior [nats] | Neural KL vs prior [nats] | Bayes 95% width / prior-unit range | Neural 95% width / prior-unit range | Median PCA retained nuisance-projected local information |',
            '|---|---:|---:|---:|---:|---:|']
    for number,c in enumerate(cases,1):
        for j,name in enumerate(TARGETS):
            b=c['posterior_prior_comparison'][name]['bayesian'];q=c['posterior_prior_comparison'][name]['neural']
            ratio=np.median([r['nuisance_projected_information_retention'][j] for r in c['local_information']])
            lines.append(f'| {number} / {name} | {b["descriptive_histogram_KL_from_uniform_prior_nats"]:.5f} | {q["descriptive_histogram_KL_from_uniform_prior_nats"]:.5f} | {b["central95_prior_unit_width"]:.5f} | {q["central95_prior_unit_width"]:.5f} | {ratio:.5f} |')
    lines+=['','## Interpretation limits and possible bounded remedy','',
            'No leakage, target-unit conversion, precision-factor sampling or point-median implementation defect was found '
            'in the frozen route: raw predictions and preprocessing reproduce exactly and analytic/known-distribution '
            'regressions pass. This does not prove the estimator is statistically adequate. Only 397 training sources and '
            '67 reused validation sources, one training seed and a single restricted conditional family limit conclusions. '
            'The half-data/full-data validation NLL comparison is only one diagnostic and cannot establish data sufficiency.', '',
            'At zero information, a uniform prior-unit u transforms to a logistic distribution under logit(u), while the '
            'model assumes a Gaussian in that coordinate. An otherwise identical Gaussian in probit(u) can represent the '
            'exact prior when its mean is zero and covariance identity. One frozen-budget probit-transform sensitivity '
            'probe would therefore isolate a recoverable prior-family choice without changing observations or physical '
            'priors. It cannot restore information discarded by PCA or ensure a complex posterior is Gaussian. Any such '
            'probe must freeze its protocol before training and remain a validation-development comparison with no test '
            'scoring or automatic model replacement.']
    (out/'DIAGNOSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verify_preserved(review)
    print('Validation-only model diagnosis saved to '+str(out),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--review',type=Path,default=Path('output/ml_initial_20261023/review_01'));args=ap.parse_args()
    lock=Path('output/ml_initial_20261023/continuation.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,b'model_diagnosis');os.close(fd)
    try:
        atomic_json(args.review/'STATUS.json',{'execution':'actively_executing','stage':'model_diagnosis','pid':os.getpid()})
        execute(args.review)
        atomic_json(args.review/'STATUS.json',{'execution':'stage_complete','stage':'model_diagnosis','scientific_limits_eliminated':False})
    except Exception as exc:
        atomic_json(args.review/'STATUS.json',{'execution':'failed','stage':'model_diagnosis','error':str(exc)});raise
    finally:lock.unlink()
