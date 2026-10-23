import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_t
from angle_refresh import global_parameters,global_proposal,global_logq


def test_global_PA_proposal_has_exact_student_t_Hastings_density():
    rng=np.random.default_rng(8)
    center=np.r_[.5,.5,.5,0.,0.]
    chol=np.diag([.08,.03,.03,.2,.3]);chol[1,0]=.02;chol[3,0]=.1
    beta,factor,inverse=global_parameters(center,chol)
    differences=[]
    for _ in range(100):
        draw=global_proposal(rng,center,.5,beta,factor)
        assert 0<=draw[0]<=1
        location=center[1:]+beta*(draw[0]-center[0])
        reference=multivariate_t.logpdf(draw[1:],loc=location,shape=factor@factor.T,df=5)
        differences.append(reference-global_logq(draw,center,beta,inverse))
    assert_allclose(differences,np.full(100,differences[0]),atol=1e-12)


def test_global_PA_proposal_covers_whole_period_and_conditional_dependence():
    rng=np.random.default_rng(9);center=np.r_[.5,.5,.5,0.,0.]
    chol=np.diag([.1,.01,.01,.1,.1]);chol[3,0]=.2
    beta,factor,inverse=global_parameters(center,chol)
    draws=np.array([global_proposal(rng,center,.5,beta,factor) for _ in range(20000)])
    assert_allclose(draws[:,0].mean(),.5,atol=.01)
    assert_allclose(draws[:,0].var(),1/12,atol=.003)
    # Law of total covariance: Cov(PA,gain_z)=beta_gain * Var(uniform PA).
    assert_allclose(np.cov(draws[:,0],draws[:,3])[0,1],beta[2]/12,atol=.006)
