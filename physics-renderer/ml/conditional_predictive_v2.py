"""Coherent nuisance extension of a frozen four-target neural posterior.

Sample q_NN(theta|y), then p(nuisance|theta,y) using the ORIGINAL physics law.
This preserves target/nuisance dependence without using nuisance truths or
reweighting/filtering neural theta draws. It costs conditional Bayesian solves;
it is not a fully amortized joint neural model or a cure for inaccurate q_NN.
"""
from collections import OrderedDict
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares
from diagnostics_v2 import diagnostic_report
from hii_sim.io import atomic_json, write_npz
from run_initial import original_to_unit

NUISANCE=['pa_branch','east_prior_unit','north_prior_unit','common_gain_z','slope_z']


class ConditionalLikelihood:
    def __init__(self,like,targets):
        self.like=like
        self.fixed=original_to_unit(targets,like.config)
        if not np.isfinite(self.fixed).all() or np.any((self.fixed<0)|(self.fixed>1)):
            raise ValueError('Neural target draw outside declared support')
        self.white_y=like.whiten(like.y)
        self.cache=OrderedDict()
        self.forward_calls=0
        self.logp_calls=0
        self.nonfinite_proposals=0

    def basis(self,geometry):
        key=tuple(geometry)
        if key not in self.cache:
            t=np.r_[self.fixed,geometry[0]%1,geometry[1:3],0.,0.]
            base=self.like.vector(t)
            # base already includes exp(-s^2/2); variable gains below are
            # exp(s*z_common + slope_sigma*z_slope*log_frequency).
            matrix=np.zeros((len(base),len(self.like.o.frequencies_ghz)))
            matrix[np.arange(len(base)),self.like.m.frequency_index]=base
            white=np.empty_like(matrix)
            for idx,inverse in self.like.buckets:
                white[idx]=np.einsum('bij,bjk->bik',inverse,matrix[idx])
            self.cache[key]=white
            self.forward_calls+=1
            if len(self.cache)>32:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def residual(self,n):
        with np.errstate(over='ignore',invalid='ignore'):
            gains=np.exp(self.like.s*n[3]+self.like.o.calibration.spectral_log_slope_std*n[4]*self.like.logf)
            return np.r_[self.basis(n[:3])@gains-self.white_y,n[3:5]]

    def logp(self,n,branch_center):
        self.logp_calls+=1
        if not np.isfinite(n).all():
            self.nonfinite_proposals+=1
            return -np.inf
        if abs(n[0]-branch_center)>.5 or np.any((n[1:3]<0)|(n[1:3]>1)):
            return -np.inf
        r=self.residual(n)
        if not np.isfinite(r).all():
            self.nonfinite_proposals+=1
            return -np.inf
        return -.5*float(r@r)


def conditional_diagnostics(chains):
    angle=2*np.pi*chains[:,:,0]
    a=np.concatenate((chains,np.sin(angle)[...,None],np.cos(angle)[...,None]),axis=2)
    return diagnostic_report(a,NUISANCE+['pa_sin','pa_cos'])


def run_conditional(conditional,settings,seed,out,tag):
    required=('warmup','max_draws','diagnostic_every','wall_seconds')
    if (any(not np.isfinite(settings[k]) for k in required) or settings['warmup']<0 or
        settings['max_draws']<20 or settings['diagnostic_every']<20 or settings['wall_seconds']<=0 or
        any(int(settings[k])!=settings[k] for k in required[:3]) or
        not np.isfinite(settings['calibration_step']) or settings['calibration_step']<=0 or
        not isinstance(settings['map_max_nfev'],int) or settings['map_max_nfev']<=0):
        raise ValueError('Invalid bounded conditional sampler settings')
    if (out/f'{tag}_sampling.json').exists() or (out/f'{tag}_chains.npz').exists():
        raise FileExistsError('Preserve conditional attempts; use a new versioned output')
    started,cpu=time.perf_counter(),time.process_time()
    rngs=[np.random.default_rng(seed+1009*c) for c in range(4)]
    fits=[]
    # Gaussian calibration priors are UNBOUNDED. Bounds here apply only to
    # uniform PA/position coordinates; residual includes both N(0,1) priors.
    for c in range(4):
        fit=least_squares(conditional.residual,[(c+.5)/4,.5,.5,0.,0.],
                          bounds=([0,0,0,-np.inf,-np.inf],[1,1,1,np.inf,np.inf]),
                          max_nfev=settings['map_max_nfev'],diff_step=1e-4)
        fits.append(fit)
        if time.perf_counter()-started>settings['wall_seconds']:
            break
    best=min(fits,key=lambda f:f.cost)
    center=best.x.copy()
    branch_center=float(center[0])
    h=best.jac.T@best.jac
    vals,vecs=np.linalg.eigh(h)
    chol=cholesky((vecs*(1/np.maximum(vals,1.)))@vecs.T,lower=True)
    ind_chol=chol*np.sqrt(1.5)
    inv=solve_triangular(ind_chol,np.eye(5),lower=True)
    def logq(n):
        v=inv@(n-center)
        return -5*np.log1p(v@v/5)  # df=5, dimension=5; normalizer cancels
    states=[]
    for c in range(4):
        state=fits[min(c,len(fits)-1)].x.copy()
        state[0]=branch_center+(state[0]-branch_center+.5)%1-.5
        states.append(state)
    states=np.array(states)
    lps=np.array([conditional.logp(n,branch_center) for n in states])
    retained=[]
    accepted=np.zeros((4,3),int)
    attempts=np.zeros((4,3),int)
    reason='draw_budget'
    step=-1
    completed=0
    def save():
        chain=np.array(retained).transpose(1,0,2) if retained else np.empty((4,0,5))
        diag=conditional_diagnostics(chain) if chain.shape[1]>=20 else {'passed':False,'variables':{},'failure':'insufficient_retained_draws'}
        write_npz(out/f'{tag}_chains.npz',coordinates=chain,states=states,logp=lps,
                  fixed_prior_units=conditional.fixed,proposal_center=center,proposal_cholesky=chol)
        status={'converged':bool(diag['passed']),'diagnostics':diag,'stop_reason':reason,
                'retained_per_chain':chain.shape[1],'steps_including_warmup':completed,
                'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,
                'forward_calls':conditional.forward_calls,'logp_calls':conditional.logp_calls,
                'nonfinite_proposal_rejections':conditional.nonfinite_proposals,
                'acceptance_independence_joint_calibration':(accepted/np.maximum(attempts,1)).tolist(),
                'seed':seed,'rng_end_states':[r.bit_generator.state for r in rngs],
                'initializations':[{'cost':float(f.cost),'success':bool(f.success),'nfev':int(f.nfev),'coordinates':f.x.tolist()} for f in fits],
                'branch_center':branch_center,
                'prior':'PA uniform over one full period, position uniform prior units [0,1], unbounded independent standard-normal calibration coordinates',
                'resume_policy':'Latest arrays/RNG states retained; this CLI does not automatically resume an incomplete conditional solve. New attempts require new versioned output.'}
        atomic_json(out/f'{tag}_sampling.json',status)
        return chain,status
    for step in range(settings['warmup']+settings['max_draws']):
        if time.perf_counter()-started>settings['wall_seconds']:
            reason='wall_budget'
            break
        for c,rng in enumerate(rngs):
            p=rng.uniform()
            if p<.4:
                kind=0
                proposal=center+ind_chol@rng.normal(size=5)/np.sqrt(rng.chisquare(5)/5)
                correction=logq(states[c])-logq(proposal)
            elif p<.75:
                kind=1
                proposal=states[c]+.7*chol@rng.normal(size=5)
                correction=0.
            else:
                kind=2
                proposal=states[c].copy()
                proposal[3:]+=rng.normal(size=2)*settings['calibration_step']
                correction=0.
            lp=conditional.logp(proposal,branch_center)
            attempts[c,kind]+=1
            if np.log(rng.uniform())<lp-lps[c]+correction:
                states[c],lps[c]=proposal,lp
                accepted[c,kind]+=1
        if step>=settings['warmup']:
            retained.append(states.copy())
        completed=step+1
        if retained and len(retained)%settings['diagnostic_every']==0:
            chain,status=save()
            print(f'{tag}: {len(retained)} draws/chain; pass={status["converged"]}; {status["wall_seconds"]:.1f}s',flush=True)
            if status['converged']:
                reason='convergence_gate'
                break
    chain,status=save()  # Also preserves failures/stops before first milestone.
    return chain,status
