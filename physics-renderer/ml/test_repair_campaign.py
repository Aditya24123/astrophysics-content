"""Independent mathematical checks for the compressed posterior repair."""
import numpy as np
from scipy.special import ndtr, ndtri
from repair_campaign import whiten_rows, compression_matrix, compressed_covariance, CompressedLikelihood, latent_to_unit

def fixture():
    rng=np.random.default_rng(7);a=rng.normal(size=(6,6));sigma=a@a.T+np.eye(6);l=np.linalg.cholesky(sigma);w=np.linalg.inv(l)
    buckets=[(np.arange(6)[None,:],w[None,:,:])]
    q,_=np.linalg.qr(rng.normal(size=(6,3)));scale=np.array([2.,3.,4.]);pca=dict(basis=q,scale=scale)
    return rng,sigma,w,buckets,pca

def test_exact_compressed_covariance_and_operator():
    rng,sigma,w,buckets,pca=fixture();c=compression_matrix(pca,buckets);x=rng.normal(size=(10,6))
    np.testing.assert_allclose(whiten_rows(x,buckets),x@w.T,atol=1e-14)
    np.testing.assert_allclose(c,w.T@pca['basis']/pca['scale'],atol=1e-14)
    np.testing.assert_allclose(compressed_covariance(c,buckets),c.T@sigma@c,atol=1e-14)
    np.testing.assert_allclose(c.T@sigma@c,np.diag(1/pca['scale']**2),atol=1e-14)

def test_compressed_gaussian_energy_is_exact_projection():
    rng,sigma,w,buckets,pca=fixture();c=compression_matrix(pca,buckets);r=rng.normal(size=6);v=c.T@r
    energy=v@np.linalg.solve(c.T@sigma@c,v);q=pca['basis'];white=w@r
    np.testing.assert_allclose(energy,np.sum((q.T@white)**2),atol=1e-13)
    np.testing.assert_allclose(white@white,energy+np.sum((white-q@(q.T@white))**2),atol=1e-13)
    rotation=rng.normal(size=(3,3));vr=rotation.T@v;cr=rotation.T@(c.T@sigma@c)@rotation
    np.testing.assert_allclose(energy,vr@np.linalg.solve(cr,vr),rtol=1e-10)

def test_probit_uniform_prior_change_of_variables_and_nuisance():
    z=np.array([-.8,.2,1.,-.3,1.2,.4,-1.5,.7,-.6]);t=latent_to_unit(z)
    np.testing.assert_allclose(ndtri(t[:7]),z[:7]);np.testing.assert_array_equal(t[7:],z[7:])
    # Uniform-u log prior 0 + log|du/dz| + normal-calibration prior.
    logjac=np.sum(-.5*z[:7]**2-.5*np.log(2*np.pi));cal=-.5*np.sum(z[7:]**2)-np.log(2*np.pi)
    np.testing.assert_allclose(logjac+cal,-.5*z@z-4.5*np.log(2*np.pi))

def test_compressed_target_keeps_exact_prior_and_mean_cancels():
    rng,sigma,w,buckets,pca=fixture();c=compression_matrix(pca,buckets)
    class Base:
        y=np.arange(6.)
        def vector(self,t):return np.arange(6.)+t[0]
    base=Base();base.buckets=buckets;like=CompressedLikelihood(base,c);z=np.linspace(-.7,.7,9)
    r=c.T@(base.vector(latent_to_unit(z))-base.y)
    expected=-.5*r@np.linalg.solve(c.T@sigma@c,r)-.5*z@z
    np.testing.assert_allclose(like.logp(z),expected,atol=1e-13)
    offset=rng.normal(size=6);np.testing.assert_allclose(c.T@(base.vector(latent_to_unit(z))-offset)-c.T@(base.y-offset),r,atol=1e-14)

def test_fixed_hastings_sampler_recovers_known_gaussian(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from repair_campaign import sample_case
    import repair_campaign
    original_read=repair_campaign.read
    def no_science_deadline_in_numerical_fixture(path):
        value=original_read(path)
        if Path(path)==repair_campaign.PLAN:value=dict(value,science_stop_utc='2100-01-01T00:00:00+00:00')
        return value
    monkeypatch.setattr(repair_campaign,'read',no_science_deadline_in_numerical_fixture)
    center=np.linspace(-.3,.4,9);strength=np.linspace(.5,1.5,9)
    class Gaussian:
        def residual(self,z):return np.r_[strength*(z-center),z]
        def logp(self,z):
            r=self.residual(z);return -.5*r@r
    settings=dict(case_wall_seconds=30,map_max_nfev=20,proposal_covariance_inflation=1.5,student_df=5,independence_probability=.8,local_scale=.6,warmup=1000,max_draws_per_chain=3000,diagnostics_every=1000)
    config=json.loads(Path('configs/compact_proxy_gl4_v3.json').read_text());dest=tmp_path/'known'
    result=sample_case(Gaussian(),config,settings,571,dest)
    with np.load(dest.with_suffix('.chains.npz')) as saved:chain=saved['latent_coordinates'].reshape(-1,9)
    expected=strength**2*center/(1+strength**2);variance=1/(1+strength**2)
    np.testing.assert_allclose(chain.mean(axis=0),expected,atol=.065)
    np.testing.assert_allclose(chain.var(axis=0),variance,rtol=.12,atol=.015)
    assert result['passed']
