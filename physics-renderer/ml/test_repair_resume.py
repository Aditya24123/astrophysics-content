"""A split fixed refresh must replay the exact unsplit transition path."""
import json
from pathlib import Path
import numpy as np
from hii_sim.io import atomic_json,write_npz
from repair_sampler_refresh import refresh
from repair_refresh_resume import resume

def test_refresh_resume_has_identical_transition_path(tmp_path,monkeypatch):
    import repair_sampler_refresh as original
    import repair_refresh_resume as continuation
    for module in [original,continuation]:
        actual=module.read
        def future(path,actual=actual):
            value=actual(path)
            if Path(path)==original.PLAN:value=dict(value,science_stop_utc='2100-01-01T00:00:00+00:00')
            return value
        monkeypatch.setattr(module,'read',future)
    class Gaussian:
        def logp(self,z):return -.5*float(z@z)-.25*float(np.sum((z-.2)**2))
    config=json.loads(Path('configs/compact_proxy_gl4_v3.json').read_text());rng=np.random.default_rng(3);pilot=rng.normal(0,.8,(4,200,9));parent=tmp_path/'pilot';write_npz(parent.with_suffix('.chains.npz'),latent_coordinates=pilot);atomic_json(parent.with_suffix('.json'),dict(passed=False,wall_seconds=.5))
    settings=dict(proposal_covariance_inflation=1.5,student_df=5,angle_probability=.25,independence_probability=.8,local_scale=.6,warmup=10,max_draws=40,diagnostic_every=1000,case_wall_seconds=30)
    split=tmp_path/'split';whole=tmp_path/'whole';extended=tmp_path/'extended';refresh(Gaussian(),config,parent,split,settings,701);refresh(Gaussian(),config,parent,whole,dict(settings,max_draws=100),701);resume(Gaussian(),config,split,extended,max_additional=60,wall_seconds=30)
    with np.load(whole.with_suffix('.chains.npz')) as a,np.load(extended.with_suffix('.chains.npz')) as b:
        for name in ['latent_coordinates','coordinates','log_likelihood','states']:np.testing.assert_array_equal(a[name],b[name])
    assert json.loads(whole.with_suffix('.json').read_text())['rng_states']==json.loads(extended.with_suffix('.json').read_text())['rng_states']
