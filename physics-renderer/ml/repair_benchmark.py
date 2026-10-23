"""Cached four-target neural inference latency on one fixed public validation input."""
import time
import numpy as np
from hii_sim.io import atomic_json,digest,versions
from hii_sim.training_data import load_transform
from review_inference_v2 import inputs,read,arrays
from review_contract import CheckedLikelihood
from repair_campaign import OUT,PLAN
from model import Support,project,sample
from probit_probe import ProbitSupport

def main():
    plan=read(PLAN);old,_,base,root,config,selection=inputs(read('ml/review_plan_v2.json'));like=CheckedLikelihood(selection[0],config);y=like.y[None];transform=load_transform(base['prepared'],root/'public');pc=arrays(old/'full/pca.npz');rep=arrays(OUT/'representation.npz');rows=[]
    for name,path,support in [('canonical_logit32',old/'full/best.npz',Support(config)),('rejected_thermal_probit128',OUT/'neural/best.npz',ProbitSupport(config))]:
        p=arrays(path);walls=[];cpus=[]
        for k in range(21):
            wall,cpu=time.perf_counter(),time.process_time()
            x=(project(transform.features(y).astype(float),pc) if name=='canonical_logit32' else (y-rep['raw_mean'])@rep['compression']-rep['mean']@(rep['basis']/rep['scale']))
            draws=support.decode(sample(x,p,2048,np.random.default_rng(plan['seed']+k)));walls.append(time.perf_counter()-wall);cpus.append(time.process_time()-cpu)
            if draws.shape!=(1,2048,4) or not np.isfinite(draws).all():raise ValueError('benchmark output')
        rows.append(dict(model=name,checkpoint_sha256=digest(path),posterior_draws=2048,first_wall_seconds=walls[0],warm_median_wall_seconds=float(np.median(walls[1:])),warm_p90_wall_seconds=float(np.quantile(walls[1:],.9)),warm_mean_cpu_seconds=float(np.mean(cpus[1:])),wall_repeats_seconds=walls,cpu_repeats_seconds=cpus,model_promoted=False if name.startswith('rejected') else 'canonical retained, scientific limitations remain'))
    atomic_json(OUT/'inference_benchmarks.json',dict(rows=rows,bookkeeping=selection[0]['bookkeeping'],software=versions(),seed_base=plan['seed'],repeats=21,scope='Same raw public observation in RAM; raw feature transform/PCA, RNG creation, conditional posterior sampling and physical decoding included; disk, model/likelihood loading, flag reporting and nuisance posterior sampling excluded. Four-target marginal draws only.',comparison_limit='Canonical32 runtime can be placed beside matched old32 Bayesian phase accounting, but approximation accuracy differs and only two validation cases exist: no universal or matched-accuracy speedup claim. Rejected128 timing is not a comparison on the same observed compression as old32 Bayes.',test_evaluated=False))
    print('Cached raw-observation neural latency measured; no label/test access',flush=True)

if __name__=='__main__':main()
