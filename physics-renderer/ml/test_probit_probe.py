import json
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import ndtri
from model import Support,initialize,predict,LOGIT_SCALE
from probit_probe import ProbitSupport,probit_prior_unit_nll
from diagnose_model import prior_unit_nll


def test_probit_support_roundtrip_and_exact_prior_density():
    config=json.loads(Path('configs/compact_proxy_gl4_v3.json').read_text())
    support=ProbitSupport(config)
    u=np.random.default_rng(6).uniform(.0001,.9999,size=(1000,4))
    physical=support.decode(ndtri(u))
    assert_allclose(support.unit(physical),u,atol=1e-14)
    assert_allclose(support.encode(physical),ndtri(u),atol=1e-12)
    zero=np.zeros_like(u);identity=np.tile(np.eye(4),(len(u),1,1))
    joint,marginal=probit_prior_unit_nll(physical,zero,identity,support)
    assert_allclose(joint,0,atol=1e-12);assert_allclose(marginal,0,atol=1e-12)


def test_common_prior_unit_scores_include_correct_jacobian():
    config=json.loads(Path('configs/compact_proxy_gl4_v3.json').read_text())
    old=Support(config);new=ProbitSupport(config)
    rng=np.random.default_rng(7);p=initialize(3,5,rng);x=rng.normal(size=(20,3))
    mu,a,_=predict(x,p);physical=new.decode(rng.normal(size=(20,4)))
    old_score,_=prior_unit_nll(physical,mu,a,old)
    new_score,_=probit_prior_unit_nll(physical,mu,a,new)
    # Independent multivariate density + analytical Jacobian in u coordinates.
    u=new.unit(physical)
    for j,(support,score) in enumerate([(old,old_score),(new,new_score)]):
        z=support.encode(physical);e=z-mu
        log_gaussian=-.5*np.sum(np.einsum('bij,bj->bi',a,e)**2,axis=1)+np.log(np.diagonal(a,axis1=1,axis2=2)).sum(axis=1)-2*np.log(2*np.pi)
        logjac=(-np.log(LOGIT_SCALE)-np.log(u)-np.log1p(-u)).sum(axis=1) if j==0 else (.5*z*z+.5*np.log(2*np.pi)).sum(axis=1)
        assert_allclose(score,-log_gaussian-logjac,atol=1e-12)
