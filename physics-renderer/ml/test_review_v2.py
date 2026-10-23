import copy
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose
from diagnostics_v2 import diagnostic_report,autocovariance,ess
from review_contract import CheckedLikelihood,validate_config,validate_case_set,full_diagnostics
from conditional_predictive_v2 import ConditionalLikelihood,run_conditional
from hii_sim.loaders import NPEInputDataset


@pytest.fixture(scope='module')
def fixture():
    config=json.loads(Path('configs/compact_proxy_gl4_v3.json').read_text())
    public=Path('output/training_readiness_20261021/candidate_gl4_512_01/public')
    inp=NPEInputDataset(public,split='validation')[0]
    return config,inp


def test_every_variable_must_be_named_and_nonfinite_fails_closed():
    x=np.random.default_rng(31).normal(size=(4,1000,2))
    with pytest.raises(ValueError,match='EVERY'):
        diagnostic_report(x,['one'])
    with pytest.raises(ValueError):
        diagnostic_report(x,['duplicate','duplicate'])
    x[0,1,1]=np.nan
    report=diagnostic_report(x,['first','bad'])
    assert not report['passed'] and report['variables']['bad']['failure']=='nonfinite_chain_values'
    json.dumps(report,allow_nan=False)


def test_stuck_or_chain_specific_constant_failures_are_serializable():
    for x in (np.ones((4,100,1)),np.broadcast_to(np.arange(4)[:,None,None],(4,100,1))):
        report=diagnostic_report(x,['stuck'])
        assert not report['passed']
        json.dumps(report,allow_nan=False)


def test_odd_fft_padding_matches_independent_direct_autocovariance():
    rng=np.random.default_rng(42)
    # n=31 -> next_fast_len(62)=63 (odd), the historical inverse-length defect.
    for n in (31,32,63,64,169):
        x=rng.normal(size=(4,n))
        c=x-x.mean(axis=1,keepdims=True)
        direct=np.array([[np.dot(row[:n-lag],row[lag:])/n for lag in range(n)] for row in c])
        assert_allclose(autocovariance(x),direct,atol=1e-14,rtol=1e-12)


def test_rank_folded_diagnostics_detect_different_modes_and_scales():
    x=np.random.default_rng(4).normal(size=(4,5000,2))
    assert diagnostic_report(x,['a','b'])['passed']
    x[0,:,1]+=4
    assert not diagnostic_report(x,['a','b'])['passed']
    x[0,:,1]-=4
    x[0,:,1]*=4
    assert not diagnostic_report(x,['a','b'])['passed']


@pytest.mark.parametrize('mutation',['pa','offset','target_transform','grid','noise','features'])
def test_prior_units_and_feature_contract_reject_mismatch(fixture,mutation):
    config,inp=fixture
    config,inp=copy.deepcopy(config),copy.deepcopy(inp)
    if mutation=='pa':config['sampling']['ranges']['pa_deg'][1]=360
    elif mutation=='offset':config['sampling']['offset_limit_arcsec']=20
    elif mutation=='target_transform':config['sampling']['ranges']['em0_pc_cm6'][2]='linear'
    elif mutation=='grid':config['grid']['subpixels']=2
    elif mutation=='noise':config['instrument']['calibration_fraction']=.1
    else:inp['features']['true_parameters']=[1,2,3,4]
    with pytest.raises(ValueError):validate_config(config,inp)


def test_cached_conditional_likelihood_equals_full_correlated_likelihood(fixture):
    config,inp=fixture
    like=CheckedLikelihood(inp,config)
    targets=like.unpack(np.r_[np.full(7,.5),0.,0.])[0]
    conditional=ConditionalLikelihood(like,targets)
    rng=np.random.default_rng(9)
    for _ in range(5):
        geometry=rng.uniform(.05,.95,3)
        for _ in range(3):
            n=np.r_[geometry,rng.normal(size=2)]
            direct=like.residual(np.r_[conditional.fixed,n])
            assert_allclose(conditional.residual(n),direct,rtol=1e-12,atol=1e-10)
    before=conditional.forward_calls
    n[3:]+=1
    conditional.residual(n)
    assert conditional.forward_calls==before


def test_case_set_is_exact_not_zip_truncated():
    expected=[('a','a1'),('b','b1')]
    good=[{'source_id':a,'realization_id':b} for a,b in expected]
    validate_case_set(good,expected)
    for bad in (good[:1],good+[good[0]],[good[0],good[0]],good+[{'source_id':'c','realization_id':'c1'}]):
        with pytest.raises(ValueError):validate_case_set(bad,expected)


class KnownConditional:
    def __init__(self,theta):
        self.mean=np.r_[.5,.5,.5,9*theta,.3*theta]
        self.scale=np.array([.04,.03,.03,.2,.3])
        self.fixed=np.array([theta])
        self.forward_calls=self.logp_calls=self.nonfinite_proposals=0
    def residual(self,n):
        physical=n.copy()
        physical[0]%=1
        return (physical-self.mean)/self.scale
    def logp(self,n,branch_center):
        self.logp_calls+=1
        if abs(n[0]-branch_center)>.5 or np.any((n[1:3]<0)|(n[1:3]>1)):
            return -np.inf
        r=self.residual(n)
        return -.5*r@r


def test_conditional_sampler_recovers_dependency_and_unbounded_calibration(tmp_path):
    settings={'warmup':300,'max_draws':4000,'diagnostic_every':2000,'wall_seconds':30,
              'map_max_nfev':50,'calibration_step':.15}
    means=[]
    for theta in (-1,1):
        known=KnownConditional(theta)
        chain,status=run_conditional(known,settings,100+theta,tmp_path,'known_'+str(theta))
        physical=chain.copy()
        physical[:,:,0]%=1
        assert status['converged']
        assert_allclose((physical.mean(axis=(0,1))-known.mean)/known.scale,np.zeros(5),atol=.12)
        assert_allclose(physical.std(axis=(0,1))/known.scale,np.ones(5),atol=.12)
        means.append(physical[:,:,3].mean())
    assert means[1]-means[0]>17.9  # Correct dependence; no +/-7 clipping or true nuisance substitution.


def test_conditional_budget_stop_before_first_checkpoint_is_saved(tmp_path):
    settings={'warmup':500,'max_draws':999,'diagnostic_every':1000,'wall_seconds':.000001,
              'map_max_nfev':2,'calibration_step':.1}
    chain,status=run_conditional(KnownConditional(0),settings,3,tmp_path,'stopped')
    assert not status['converged'] and status['stop_reason']=='wall_budget'
    assert chain.shape==(4,0,5) and status['steps_including_warmup']==0
    assert (tmp_path/'stopped_sampling.json').exists()
    json.dumps(status,allow_nan=False)
    settings['wall_seconds']=-1
    with pytest.raises(ValueError):run_conditional(KnownConditional(0),settings,3,tmp_path,'bad')


def test_holdout_guard_allows_hashing_but_forbids_decoding(tmp_path):
    (tmp_path/'public').mkdir()
    (tmp_path/'protected/labels').mkdir(parents=True)
    (tmp_path/'public/manifest.json').write_text(json.dumps({'records':[{'source_id':'heldout','split':'test'}]}))
    label=tmp_path/'protected/labels/heldout.json'
    label.write_text('{"toy_label": 42}')
    code='''
from pathlib import Path
import sys
from hii_sim.io import digest
from review_inference_v2 import seal_holdout_labels
root=Path(sys.argv[1]);seal_holdout_labels(root)
path=root/'protected/labels/heldout.json'
assert len(digest(path))==64
try:
    path.read_text()
except PermissionError:
    print('hash-only PASS')
else:
    raise AssertionError('Holdout decoding permitted')
'''
    result=subprocess.run([sys.executable,'-B','-c',code,str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'hash-only PASS' in result.stdout
