from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_allclose
from diagnose_model import local_information


def fixture():
    rng=np.random.default_rng(17);d=12
    raw=rng.normal(size=(d,d));cov=raw@raw.T+np.eye(d)
    chol=np.linalg.cholesky(cov);inverse=np.linalg.inv(chol)
    jac=rng.normal(size=(d,9))
    like=SimpleNamespace(vector=lambda t:jac@t,whiten=lambda x:inverse@x,
                         buckets=[(np.arange(d)[None,:],inverse[None,:,:])])
    transform=SimpleNamespace(feature_scale_jy=np.ones(d))
    return like,transform,jac,cov


def test_identity_compression_retains_all_parameter_information():
    like,transform,jac,cov=fixture()
    result=local_information(like,np.r_[np.full(7,.5),0.,0.],transform,
                             {'basis':np.eye(12),'scale':np.ones(12)},1e-4)
    assert_allclose(result['conditional_information_retention'],1,atol=1e-10)
    assert_allclose(result['nuisance_projected_information_retention'],1,atol=1e-10)
    full=jac.T@np.linalg.solve(cov,jac);full[7,7]+=1;full[8,8]+=1
    expected=full[:4,:4]-full[:4,4:]@np.linalg.solve(full[4:,4:],full[4:,:4])
    assert_allclose(result['full_nuisance_projected_information'],expected,atol=1e-9)


def test_lossy_compression_uses_transformed_correlated_covariance():
    like,transform,jac,cov=fixture()
    basis=np.eye(12)[:,:10]
    result=local_information(like,np.r_[np.full(7,.5),0.,0.],transform,
                             {'basis':basis,'scale':np.ones(10)},1e-4)
    fj=jac.T@basis@np.linalg.solve(basis.T@cov@basis,basis.T@jac)
    fj[7,7]+=1;fj[8,8]+=1
    expected=fj[:4,:4]-fj[:4,4:]@np.linalg.solve(fj[4:,4:],fj[4:,:4])
    assert_allclose(result['compressed_nuisance_projected_information'],expected,atol=1e-9)
    assert np.all(np.array(result['conditional_information_retention'])<=1+1e-10)
