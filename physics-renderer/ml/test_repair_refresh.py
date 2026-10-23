import numpy as np
from scipy.special import ndtr
from scipy.stats import norm,multivariate_t
from repair_sampler_refresh import proposal_parameters,angle_logq,angle_proposal

def test_angle_refresh_hastings_density_matches_independent_reference():
    rng=np.random.default_rng(201);a=rng.normal(size=(9,9));pilot=(rng.normal(size=(4,1000,9))@a.T)*.2
    mean,chol,others,beta,conditional=proposal_parameters(pilot,1.5);z1=rng.normal(size=9);z2=rng.normal(size=9)
    def reference(z):return norm.logpdf(z[4])+multivariate_t.logpdf(z[others],loc=mean[others]+beta*(z[4]-mean[4]),shape=conditional@conditional.T,df=5)
    np.testing.assert_allclose(angle_logq(z1,mean,others,beta,conditional,5)-angle_logq(z2,mean,others,beta,conditional,5),reference(z1)-reference(z2),atol=1e-12)

def test_angle_refresh_is_uniform_in_physical_prior_angle():
    rng=np.random.default_rng(202);pilot=rng.normal(size=(4,500,9));mean,chol,others,beta,conditional=proposal_parameters(pilot,1.5)
    draws=np.array([angle_proposal(rng,mean,others,beta,conditional,5) for _ in range(10000)]);u=ndtr(draws[:,4])
    assert abs(u.mean()-.5)<.01
    assert abs(u.var()-1/12)<.003
    assert np.isfinite(draws).all()
