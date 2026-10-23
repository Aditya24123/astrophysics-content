"""Explicit new-directory reproduction and measured-data posterior prediction."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import ndtr
from hii_sim.io import atomic_json, write_npz, digest
from hii_sim.loaders import NPEInputDataset
from review_inference_v2 import read, arrays
from review_contract import CheckedLikelihood,TARGETS,UNITS
from probit_probe import ProbitSupport
from model import predict,covariance,sample

def posterior(artifact,inputs,draws,seed):
    """Public-only input. No label/truth argument; all four outputs retained."""
    config=read('configs/compact_proxy_gl4_v3.json');CheckedLikelihood(inputs,config)
    rep=arrays(artifact/'representation.npz');p=arrays(artifact/'neural/best.npz');y=inputs['features']['measurements'].values_jy.astype(float)
    start=time.perf_counter();x=((y-rep['raw_mean'])@rep['compression']-rep['mean']@(rep['basis']/rep['scale']))[None]
    support=ProbitSupport(config);mu,a,_=predict(x,p);var=np.diagonal(covariance(a),axis1=1,axis2=2);kl=.5*(mu*mu+var-1-np.log(var));lo=support.decode(mu-1.95996398454*np.sqrt(var));hi=support.decode(mu+1.95996398454*np.sqrt(var));priorlo=support.decode(np.full((1,4),-1.95996398454));priorhi=support.decode(np.full((1,4),1.95996398454));ratio=(hi-lo)/(priorhi-priorlo);flags=(kl<.1)&(ratio>.8)
    values=support.decode(sample(x,p,draws,np.random.default_rng(seed)))[0];median=support.decode(mu)[0];wall=time.perf_counter()-start
    return values,dict(targets=TARGETS,units=UNITS,posterior_median=median.tolist(),central95=[lo[0].tolist(),hi[0].tolist()],prior_KL_nats=kl[0].tolist(),prior95_width_ratio=ratio[0].tolist(),prior_dominated_warning=flags[0].tolist(),temperature_claim='Abstain from data-constrained temperature precision. This warning is heuristic, does not fix/drop Te, and unflagged neural Te precision is unvalidated.',model_status='Failed validation remedy, not promoted or certified for real data',wall_seconds_raw_affine_transform_and_posterior_draws=wall,draws=draws,seed=seed,checkpoint_sha256=digest(artifact/'neural/best.npz'),timing_excludes='disk/model loading and input validation; no conventional nuisance-conditional solve',bookkeeping=inputs['bookkeeping'])

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('stage',choices=['prepare','neural','old32','noise128','half']);r.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('predict');p.add_argument('--artifact',type=Path,default=Path('output/ml_initial_20261023/repair_02'));p.add_argument('--dataset',type=Path,default=Path('output/training_readiness_20261021/candidate_gl4_512_01'));p.add_argument('--validation-index',type=int,default=0);p.add_argument('--out',type=Path,required=True);p.add_argument('--draws',type=int,default=2048);p.add_argument('--seed',type=int,default=231040)
    args=ap.parse_args()
    if args.command=='run':
        import repair_campaign as campaign
        campaign.OUT=args.out;sys.argv=[sys.argv[0],args.stage];campaign.main()
    else:
        args.out.mkdir(exist_ok=False);inp=NPEInputDataset(args.dataset/'public',split='validation')[args.validation_index];draws,report=posterior(args.artifact,inp,args.draws,args.seed);write_npz(args.out/'posterior_draws.npz',physical_targets=draws);atomic_json(args.out/'posterior.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
