"""Validation-only summaries and exact-likelihood predictive checks of repair02."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr
from scipy.stats import norm
from astropy import units as u
from hii_sim.physics import optical_depth
from hii_sim.io import atomic_json,write_npz,digest
from hii_sim.training_data import load_transform
from hii_sim.loaders import NPEInputDataset
from review_inference_v2 import inputs,read,arrays,verify_preserved,fingerprint
from review_contract import CheckedLikelihood,TARGETS,UNITS,physical_targets,full_diagnostics,COORDS
from diagnostics_v2 import diagnostic_report
from repair_campaign import OUT,PLAN,CompressedLikelihood,gate,latent_to_unit,compressed_covariance,whiten_rows
from model import project,predict,sample,covariance,Support
from probit_probe import ProbitSupport
from diagnose_model import local_information

def prior_hist(values):
    hist=np.histogram(values,bins=np.linspace(0,1,21))[0];p=hist/hist.sum();p=p[p>0]
    return dict(histogram_bins=20,histogram_kl_prior_nats=float(np.sum(p*np.log(20*p))),prior_unit95_width=float(np.diff(np.quantile(values,[.025,.975]))[0]),caution='Finite correlated draws and arbitrary fixed bins bias histogram KL; descriptive only, not a calibrated test of prior equality')

def full_gate_with_likelihood(chain,like,config,number):
    path=OUT/f'full_y_{number:02d}_log_likelihood.npz';start=time.perf_counter()
    if path.exists():ll=arrays(path)['log_likelihood']
    else:
        unique,inverse=np.unique(chain.reshape(-1,9),axis=0,return_inverse=True);values=[]
        for index,t in enumerate(unique):
            t=t.copy();t[4]%=1;r=like.whiten(like.vector(t)-like.y);values.append(-.5*float(r@r))
            if (index+1)%1000==0:print(f'Full-y case {number}: likelihood diagnostics replay {index+1}/{len(unique)} unique states',flush=True)
        ll=np.array(values)[inverse].reshape(chain.shape[:2]);write_npz(path,log_likelihood=ll)
        atomic_json(path.with_suffix('.timing.json'),dict(wall_seconds=time.perf_counter()-start,unique_forward_states=len(unique),scope='Offline diagnostic replay of already-saved posterior, not fresh posterior sampling'))
    v=chain.copy();v[:,:,:4]=physical_targets(chain,config);a=2*np.pi*chain[:,:,4];v=np.concatenate((v,np.sin(a)[...,None],np.cos(a)[...,None],ll[...,None]),axis=2)
    return diagnostic_report(v,COORDS+['pa_sin','pa_cos','log_likelihood'])

def local_info(like,t,c):
    jac=[]
    for j in range(9):
        h=min(1e-4,.4*t[j],.4*(1-t[j])) if j in [0,1,2,3,5,6] else 1e-4
        a,b=t.copy(),t.copy();a[j]+=h;b[j]-=h
        if j==4:a[j]%=1;b[j]%=1
        jac.append((like.vector(a)-like.vector(b))/(2*h))
    jac=np.stack(jac,axis=1);full=np.stack([like.whiten(jac[:,j]) for j in range(9)],axis=1)
    cov=compressed_covariance(c,like.buckets);small=solve_triangular(np.linalg.cholesky(cov),c.T@jac,lower=True)
    prior=np.zeros((2,9));prior[0,7]=1;prior[1,8]=1
    def fisher(a):
        a=np.r_[a,prior];r=a[:,:4]-a[:,4:]@np.linalg.lstsq(a[:,4:],a[:,:4],rcond=None)[0];return r.T@r
    f,g=fisher(full),fisher(small);ratio=np.diag(g)/np.diag(f)
    def all_profiled(a):
        a=np.r_[a,prior];out=[]
        for j in range(4):
            other=[k for k in range(9) if k!=j];r=a[:,j]-a[:,other]@np.linalg.lstsq(a[:,other],a[:,j],rcond=None)[0];out.append(float(r@r))
        return out
    af,ag=all_profiled(full),all_profiled(small)
    if np.any(ratio>1+1e-5):raise ValueError('information exceeds full-data bound')
    return dict(state=t.tolist(),nuisance_projected_retention=ratio.tolist(),full_information=f.tolist(),compressed_information=g.tolist(),all_other_targets_and_nuisance_profiled_full_information=af,all_other_targets_and_nuisance_profiled_compressed_information=ag,te_local_unbounded_sigma_k=7000/np.sqrt(af[1]) if af[1]>0 else None,limits='Local derivatives at fixed existing full-y posterior states. Nuisance-profiled diagonals hold other physical targets fixed; all-other-profiled values remove those too. Local unbounded Gaussian sigma is not a posterior credible interval. Uniform bounds/global nonlinear geometry not included; not global sufficiency.')

def analyze():
    analysis_start,analysis_cpu=time.perf_counter(),time.process_time()
    old,follow,base,root,config,selection=inputs(read('ml/review_plan_v2.json'));plan=read(PLAN)
    transform=load_transform(base['prepared'],root/'public');oldpca=arrays(old/'full/pca.npz');rep=arrays(OUT/'representation.npz');cache=arrays(OUT/'development_cache.npz');books=read(OUT/'bookkeeping.json');newp=arrays(OUT/'neural/best.npz');probit=ProbitSupport(config)
    reference=None;verified=0
    for split in ['train','validation']:
        for inp in NPEInputDataset(root/'public',split=split):
            m,o=inp['features']['measurements'],inp['features']['observing']
            signature=[getattr(m,k) for k in ['frequency_index','y_index','x_index','component','self_conjugate']]+[o.thermal.sigma_real_jy,o.primary_beam,o.imaging_taper,o.frequencies_ghz,o.bandwidths_mhz]
            if reference is None:reference=[np.array(a,copy=True) for a in signature]
            if any(not np.array_equal(a,b) for a,b in zip(reference,signature)):raise ValueError('Fixed whitening/operator not identical across development observations')
            verified+=1
    atomic_json(OUT/'fixed_operator_audit.json',dict(observations=verified,splits=['train','validation'],exact_layout_primary_beam_taper_bands_and_thermal_sigma_identical=True,test_or_challenge_loaded=False,role='Post-fit safety validation of the frozen common operator, not fitting from validation data'))
    oldc=oldpca['basis']/oldpca['scale']/transform.feature_scale_jy[:,None];newc=rep['compression'];npe_models=[('original',oldc,arrays(old/'full/best.npz'),Support(config)),('probit32',oldc,arrays('output/ml_initial_20261023/review_01/probit_probe_01/full/best.npz'),probit),('thermal128',newc,newp,probit)]
    comp=[];infos=[];ppc=[]
    for number,inp in enumerate(selection,1):
        like=CheckedLikelihood(inp,config);full=arrays(follow/f'bayes_{number:02d}_chains.npz')['coordinates'];fullgate=full_gate_with_likelihood(full,like,config,number)
        if not full_diagnostics(full,config)['passed']:raise ValueError('Historical physical/nuisance diagnostic replay changed')
        flat=full.reshape(-1,9);idx=next(i for i,b in enumerate(books['validation']) if b['realization_id']==inp['bookkeeping']['realization_id'])
        info=dict(case=number,source_id=inp['bookkeeping']['source_id'],historical_parameter_gate_passed=True,current_extended_gate_passed=fullgate['passed'],states=[])
        for f in [.2,.5,.8]:
            t=flat[int(f*len(flat))].copy();t[4]%=1
            info['states'].append(dict(old32=local_info(like,t,oldc),thermal128=local_info(like,t,newc)))
        infos.append(info)
        rawscale=np.empty(len(like.y))
        for ii,w in like.buckets:
            l=np.linalg.inv(w);rawscale[ii]=np.sqrt(np.diagonal(l@l.transpose(0,2,1),axis1=1,axis2=2))
        spatial=np.fft.fftfreq(like.grid.npix);radius=np.hypot(spatial[like.m.x_index],spatial[like.m.y_index]);radius_cuts=np.quantile(radius,[.25,.5,.75]);radius_bin=np.searchsorted(radius_cuts,radius,side='right')
        for route,c in [('old32',oldc),('noise128',newc),('full_y',None)]:
            initial_phase=None;refresh_phase=None;accepted_phase='historical_full_y' if route=='full_y' else 'initial'
            if route=='full_y':chain=full;accepted=fullgate;timing=read(follow/'bayesian_comparison.json')['cases'][number-1]['sampling']
            else:
                dest=OUT/'compressed_bayes'/f'{route}_{number:02d}'
                if not dest.with_suffix('.json').exists():
                    comp.append(dict(case=number,route=route,validated=False,status='not completed'));continue
                result=read(dest.with_suffix('.json'));saved=arrays(dest.with_suffix('.chains.npz'));accepted=gate(saved['latent_coordinates'],saved['log_likelihood'],config);chain=saved['coordinates'];timing={k:result[k] for k in ['wall_seconds','cpu_seconds','draws_per_chain','warmup','acceptance']}
                history=[]
                for n in sorted(set(list(range(1000,chain.shape[1]+1,1000))+[chain.shape[1]])):
                    history.append(dict(draws_per_chain=n,diagnostics=gate(saved['latent_coordinates'][:,:n],saved['log_likelihood'][:,:n],config)))
                atomic_json(dest.with_suffix('.diagnostic_history.json'),dict(recomputed_from_preserved_raw_chain_prefixes=True,checkpoints=history))
                initial_phase=dict(passed=accepted['passed'],diagnostics=accepted,timing=timing,artifact=str(dest))
                refresh=dest.parent/'refresh_01'/dest.name
                if not accepted['passed'] and refresh.with_suffix('.json').exists():
                    result=read(refresh.with_suffix('.json'));saved=arrays(refresh.with_suffix('.chains.npz'));accepted=gate(saved['latent_coordinates'],saved['log_likelihood'],config);chain=saved['coordinates'];accepted_phase='refresh_01';timing={k:result[k] for k in ['wall_seconds','cpu_seconds','draws_per_chain','warmup','acceptance','parent_phase_wall_seconds','total_recorded_sampler_wall_seconds']}
                    refresh_phase=dict(passed=accepted['passed'],diagnostics=accepted,timing=timing,artifact=str(refresh))
                    history=[]
                    for n in sorted(set(list(range(1000,chain.shape[1]+1,1000))+[chain.shape[1]])):history.append(dict(draws_per_chain=n,diagnostics=gate(saved['latent_coordinates'][:,:n],saved['log_likelihood'][:,:n],config)))
                    atomic_json(refresh.with_suffix('.diagnostic_history.json'),dict(recomputed_from_preserved_raw_chain_prefixes=True,checkpoints=history))
                    continuation=refresh.parent/'continuation_01'/refresh.name
                    if not accepted['passed'] and continuation.with_suffix('.json').exists():
                        result=read(continuation.with_suffix('.json'));continued=arrays(continuation.with_suffix('.chains.npz'));np.testing.assert_array_equal(continued['latent_coordinates'][:,:saved['latent_coordinates'].shape[1]],saved['latent_coordinates']);saved=continued;accepted=gate(saved['latent_coordinates'],saved['log_likelihood'],config);chain=saved['coordinates'];accepted_phase='refresh_01 homogeneous continuation_01';timing={k:result[k] for k in ['wall_seconds','cpu_seconds','draws_per_chain','warmup','acceptance','parent_phase_wall_seconds','total_recorded_sampler_wall_seconds','additional_wall_seconds','additional_draws_per_chain']}
            row=dict(case=number,route=route,source_id=inp['bookkeeping']['source_id'],validated=accepted['passed'],accepted_phase=accepted_phase,initial_phase=initial_phase,refresh_phase=refresh_phase,diagnostics=accepted,timing=timing)
            if not accepted['passed']:row['status']='FAIL: retain raw diagnostics; no posterior comparison';comp.append(row);continue
            theta=physical_targets(chain,config).reshape(-1,4);uflat=chain.reshape(-1,9)
            row['bayes_quantiles_025_50_975']=np.quantile(theta,[.025,.5,.975],axis=0).tolist();row['prior_comparison']={n:prior_hist(uflat[:,j]) for j,n in enumerate(TARGETS)};row['neural']=[]
            for name,mc,p,support in npe_models:
                if route=='old32' and name=='thermal128' or route=='noise128' and name!='thermal128':continue
                feature=cache['validation_features'][idx:idx+1] if name=='thermal128' else project(transform.features(like.y)[None].astype(float),oldpca)
                mu,a,_=predict(feature,p);draw=support.decode(sample(feature,p,16384,np.random.default_rng(plan['seed']+number)))[0]
                lo,median,hi=np.quantile(draw,[.025,.5,.975],axis=0);bq=np.quantile(theta,[.025,.5,.975],axis=0)
                # Monte Carlo CDF sup-distance is descriptive, no iid significance claim.
                dist=[]
                for j in range(4):
                    grid=np.sort(np.r_[theta[:,j],draw[:,j]]);dist.append(float(np.max(abs(np.searchsorted(np.sort(theta[:,j]),grid,side='right')/len(theta)-np.searchsorted(np.sort(draw[:,j]),grid,side='right')/len(draw)))))
                row['neural'].append(dict(model=name,information_match=route!='full_y',quantiles_025_50_975=[lo.tolist(),median.tolist(),hi.tolist()],interval_width_ratio=((hi-lo)/(bq[2]-bq[0])).tolist(),marginal_empirical_CDF_sup_distance=dist,limits='Only two preselected validation sources. Gaussian conditional approximation and training-size effects remain; no population ranking. Full-y comparisons include information loss.'))
            # Genuine posterior draws of ALL nine parameters; replicate both observed compression and full vector.
            rng=np.random.default_rng(plan['seed']+30000+number);flat=chain.reshape(-1,9);chosen=rng.choice(len(flat),128,replace=False);s=[];band=[];radial=[]
            cl=CompressedLikelihood(like,c) if c is not None else None
            for ix in chosen:
                t=flat[ix].copy();t[4]%=1;delta=like.vector(t)-like.y;white=like.whiten(delta);eps=rng.normal(size=len(white));observed=float(white@white/len(white));replicated=float(eps@eps/len(eps))
                if cl is not None:
                    cr=solve_triangular(cl.chol,c.T@delta,lower=True);ce=rng.normal(size=c.shape[1]);cobs=float(cr@cr/len(cr));crep=float(ce@ce/len(ce))
                else:cobs,crep=observed,replicated
                s.append([observed,replicated,cobs,crep,float(np.sqrt(np.mean((white+eps)**2)))])
                standardized=delta/rawscale;band.append([np.sqrt(np.mean(standardized[like.m.frequency_index==f]**2)) for f in range(6)]);radial.append([np.sqrt(np.mean(standardized[radius_bin==q]**2)) for q in range(4)])
            s=np.array(s);band=np.array(band);radial=np.array(radial);write_npz(OUT/'compressed_bayes'/f'{route}_{number:02d}_predictive.npz',flat_draw_indices=chosen,discrepancies=s,raw_sigma_band_residual_rms=band,raw_sigma_radius_residual_rms=radial)
            row['predictive']=dict(draws=128,full_y_expected_signal_rms_median=float(np.median(np.sqrt(s[:,0]))),full_y_pp_tail_fraction=float(np.mean(s[:,1]>=s[:,0])),compressed_pp_tail_fraction=float(np.mean(s[:,3]>=s[:,2])),compressed_expected_signal_rms_median=float(np.median(np.sqrt(s[:,2]))),band_sigma_rms_median=np.median(band,axis=0).tolist(),radius_sigma_rms_median=np.median(radial,axis=0).tolist(),radius_bin_cuts_fftfreq_cycles_per_pixel=radius_cuts.tolist(),limits='Full-y residual from compressed posterior is an information-discard check. The fitted posterior predictive check is the compressed discrepancy; correlated posterior samples, only two cases, no calibration or incidence-rate claim. 128 Monte Carlo draws are not 128 independent observed sources. Band/radius RMS divides marginal sigma; covariance is retained in fitted discrepancy.')
            comp.append(row)
    atomic_json(OUT/'matched_comparison.json',dict(cases=comp,targets=TARGETS,units=UNITS,limits='Two validation sources reused for development. Only diagnostic-passing chains are compared. Historical original NPE float32 rounding makes old32 match accurate to the saved measured numerical rounding, not exact bitwise likelihood of a quantizer. All likelihoods keep explicit normal calibration nuisances.'))
    atomic_json(OUT/'information_retention.json',dict(cases=infos))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for row in comp:
        if not row.get('validated'):continue
        saved=arrays(OUT/'compressed_bayes'/f'{row["route"]}_{row["case"]:02d}_predictive.npz');label=f'Case {row["case"]}: {row["route"]}'
        for ax,key,x in [(axes[0],'raw_sigma_band_residual_rms',config['instrument']['frequencies_ghz']),(axes[1],'raw_sigma_radius_residual_rms',np.arange(1,5))]:
            values=saved[key];line=ax.plot(x,np.median(values,axis=0),'o-',label=label)[0];ax.fill_between(x,*np.quantile(values,[.1,.9],axis=0),alpha=.12,color=line.get_color())
    axes[0].set(xlabel='Frequency [GHz]',ylabel='Signal-minus-data residual RMS / marginal sigma',title='Band structure; 10–90% posterior-draw envelope');axes[1].set(xlabel='Observed UV-radius quartile',title='Spatial-frequency structure');axes[1].set_xticks([1,2,3,4])
    for ax in axes:ax.axhline(1,c='k',ls=':',lw=1);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(OUT/'bayesian_residual_checks.png',dpi=160);plt.close(fig)
    # Fixed prior quartiles expose edge behavior without changing this model.
    pred=arrays(OUT/'validation/predictions.npz');truth=cache['validation_targets'];unit=probit.unit(truth);groups=np.array([b['source_id'] for b in books['validation']]);sub=[]
    for j,name in enumerate(TARGETS):
        for q in range(4):
            keep=(unit[:,j]>=q/4)&(unit[:,j]<(q+1)/4);n=int(keep.sum());quant=pred['quantiles'][keep];y=truth[keep];err=abs(pred['median'][keep]-y)
            sub.append(dict(stratification_target=name,prior_unit_range=[q/4,(q+1)/4],observations=n,sources=len(set(groups[keep])),mae_all_targets=err.mean(axis=0).tolist(),central95_coverage_all_targets=np.mean((y>=quant[:,0])&(y<=quant[:,-1]),axis=0).tolist(),limits='Small descriptive validation subgroups; no fitting/tuning or subgroup confidence claim'))
    atomic_json(OUT/'validation/source_stress.json',dict(rows=sub,challenge_observations_used=0,test_observations_used=0))
    buckets=CheckedLikelihood(selection[0],config).buckets;train_stat=np.sqrt(np.mean(whiten_rows(cache['train_raw'],buckets)**2,axis=1));val_stat=np.sqrt(np.mean(whiten_rows(cache['validation_raw'],buckets)**2,axis=1));cuts=np.quantile(train_stat,[.25,.5,.75]);observable=[]
    for q in range(4):
        keep=np.searchsorted(cuts,val_stat,side='right')==q;y=truth[keep];quant=pred['quantiles'][keep]
        observable.append(dict(bin=q,observations=int(keep.sum()),sources=len(set(groups[keep])),mae_all_targets=abs(pred['median'][keep]-y).mean(axis=0).tolist(),central95_coverage_all_targets=np.mean((y>=quant[:,0])&(y<=quant[:,-1]),axis=0).tolist()))
    atomic_json(OUT/'validation/observable_stress.json',dict(statistic='RMS of raw measured y in exact conditional thermal-whitened coordinates; includes noise and calibration, not a debiased true SNR',thresholds_fitted_on='794 training observations only',thresholds=cuts.tolist(),rows=observable,limits='Descriptive reused-validation stress check only. Source groups may straddle measurement-defined bins; rows are not independent subexperiments. No score-driven refit.'))
    challenges=[]
    for inp in NPEInputDataset(root/'public',split='challenge',include_challenges=True):
        m=inp['features']['measurements'];challenges.append(dict(bookkeeping=inp['bookkeeping'],canonical_values=len(m.values_jy),status='quarantined: no base-model posterior score or labels read, irrespective of whether dimensional layout happens to match'))
    if len(challenges)!=24 or len(set(x['bookkeeping']['source_id'] for x in challenges))!=12:raise ValueError('challenge count changed')
    atomic_json(OUT/'validation/challenge_quarantine.json',dict(observations=24,sources=12,rows=challenges,protected_labels_decoded=False,posterior_scored=False,why='Different layout and/or generative challenge assumptions must remain separate; never pad, merge, fit or select this base model using these rows.'))
    # Quantify float32 ideal-C mismatch in the OLD route in compressed-noise units.
    raw=cache['validation_raw'];ideal=(raw-transform.feature_mean_jy)@oldc-oldpca['mean']@(oldpca['basis']/oldpca['scale']);actual=project(transform.features(raw).astype(float),oldpca);cov=compressed_covariance(oldc,CheckedLikelihood(selection[0],config).buckets);error=solve_triangular(np.linalg.cholesky(cov),(actual-ideal).T,lower=True).T
    atomic_json(OUT/'rounding_and_runtime.json',dict(old32_float32_rounding_whitened_l2_max=float(np.max(np.linalg.norm(error,axis=1))),old32_float32_rounding_whitened_l2_median=float(np.median(np.linalg.norm(error,axis=1))),definition='Difference historical float32 standardized feature PCA vs ideal float64 affine C(y), divided by exact compressed thermal covariance; rounding approximation explicitly retained'))
    # Physics identifiability from the preserved equation, not a fitted Te threshold.
    fs=np.array(config['instrument']['frequencies_ghz']);taus=[]
    for em in [1e4,5e5]:
        for te in [5000,12000]:taus.append(dict(em=em,te=te,tau_by_frequency=optical_depth(em*u.pc/u.cm**6,te*u.K,fs*u.GHz).tolist()))
    atomic_json(OUT/'temperature_identifiability.json',dict(prior_corner_optical_depth=taus,thin_equation='I_nu proportional to EM * Te^(-1/2) * g_ff(Te,nu) when tau << 1; morphology and gain can trade against normalization. The narrow six-band spectrum has weak distinct Te leverage.',additional_observation_hypotheses=['Measure resolved optically thick brightness with beam/foreground/calibration modeled, over frequencies that bracket optical-depth turnover for the individual source','Add an independent temperature-sensitive diagnostic with a jointly specified forward model and nuisance assumptions; outside this continuum-only data contract'],scope='Inference from the preserved physical equations and two-case posteriors; does not establish universal non-identifiability or authorize new observations/data generation.'))
    half=OUT/'half_corrected'
    if (half/'full/best.npz').exists():
        n=arrays(half/'normalization.npz');hpca=arrays(half/'full/pca.npz');hp=arrays(half/'full/best.npz');hx=((cache['validation_raw']-n['feature_mean_jy'])/n['feature_scale_jy']).astype(np.float32).astype(float);support=Support(config);out=half/'validation';out.mkdir(exist_ok=True)
        from run_initial import evaluate
        metrics,med=evaluate(hx,truth,books['validation'],hpca,hp,support,dict(base,bootstrap_parent_draws=5000),out)
        (out/'test_predictions.npz').replace(out/'predictions.npz');(out/'test_evaluation.json').unlink();metrics['split']='reused validation development only';atomic_json(out/'metrics.json',metrics)
        atomic_json(half/'comparison.json',dict(half=read(half/'full/training.json'),full=read(old/'full/training.json'),half_validation_mae=[metrics['parameters'][n]['mae_original_units'] for n in TARGETS],full_validation_mae=read(OUT/'neural_comparison.json')['before']['original']['mae'],half_validation_nll=metrics['mean_nll_scaled_logit_coordinates'],scope='Correct own-half feature means/std and PCA; fixed prior logit coordinates. One seed,198 vs397 training sources, reused validation; no trend/data-sufficiency proof. Historical half diagnostic remains preserved and used full-training feature normalization.'))
        fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True,constrained_layout=True)
        for ax,path,label in [(axes[0],old/'full','Full:397 sources'),(axes[1],half/'full','Corrected half:198 sources')]:
            rows=[json.loads(s) for s in (path/'epochs.jsonl').read_text(encoding='utf-8').splitlines()];selected=read(path/'training.json')['best_epoch']
            for key,desc in [('train_nll','Train'),('validation_nll','Reused validation')]:ax.plot([r['epoch'] for r in rows],[r[key] for r in rows],label=desc)
            ax.axvline(selected,c='k',ls=':');ax.set(title=label,xlabel='Epoch');ax.legend(fontsize=8)
        axes[0].set_ylabel('NLL [same fixed scaled-logit prior coordinates]');fig.savefig(half/'training_curves.png',dpi=160);plt.close(fig)
    before=read(OUT/'BEFORE_BINDING.json')
    for directory,expected in before['preserved_directories'].items():
        if fingerprint(Path(directory))!=expected:raise ValueError('preserved previous review changed')
    verify_preserved(Path('output/ml_initial_20261023/review_01'))
    atomic_json(OUT/'analysis_receipt.json',dict(utc=datetime.now(timezone.utc).isoformat(),source_sha256=digest(__file__),preservation_passed=True,test_scored=False,wall_seconds=time.perf_counter()-analysis_start,cpu_seconds=time.process_time()-analysis_cpu,scope='Offline complete likelihood diagnostic replay, development-operator audit, posterior predictive checks, validation/half-data summaries and preservation verification; separate from model training or posterior sampling'))
    print('Analysis complete; preservation passed',flush=True)

if __name__=='__main__':analyze()
