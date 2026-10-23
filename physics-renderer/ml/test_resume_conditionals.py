import json
import numpy as np
from numpy.testing import assert_array_equal,assert_allclose
from resume_conditionals import advance


class GaussianConditional:
    def logp(self,n,branch):
        if abs(n[0]-branch)>.5 or np.any((n[1:3]<0)|(n[1:3]>1)):return -np.inf
        return -.5*np.sum(((n-np.r_[.5,.5,.5,0.,0.])/np.r_[.04,.03,.03,.3,.3])**2)


def start():
    states=np.tile(np.r_[.5,.5,.5,0.,0.],(4,1))
    like=GaussianConditional()
    snapshot={'states':states,'logp':np.array([like.logp(s,.5) for s in states]),
              'proposal_center':states[0].copy(),'proposal_cholesky':np.diag([.04,.03,.03,.3,.3])}
    rng_states=[np.random.default_rng(30+c).bit_generator.state for c in range(4)]
    return like,snapshot,rng_states


def test_segmented_resume_has_exact_same_transition_path():
    like,snapshot,states=start()
    whole,final,rng,_=advance(like,snapshot,states,.5,.15,120,30)
    a,mid,rng_mid,_=advance(like,snapshot,states,.5,.15,47,30)
    b,final_split,rng_split,_=advance(like,mid,rng_mid,.5,.15,73,30)
    assert_array_equal(np.concatenate((a,b),axis=1),whole)
    assert_array_equal(final['states'],final_split['states'])
    assert json.dumps(rng,sort_keys=True)==json.dumps(rng_split,sort_keys=True)


def test_resumed_kernel_recovers_known_posterior_and_zero_budget_preserves_state():
    like,snapshot,states=start()
    chunk,unchanged,unadvanced,_=advance(like,snapshot,states,.5,.15,100,0)
    assert chunk.shape==(4,0,5)
    assert_array_equal(snapshot['states'],unchanged['states'])
    assert states==unadvanced
    chain,_,_,_=advance(like,snapshot,states,.5,.15,5000,30)
    assert_allclose((chain[:,1000:].mean(axis=(0,1))-snapshot['proposal_center'])/np.diag(snapshot['proposal_cholesky']),np.zeros(5),atol=.08)
