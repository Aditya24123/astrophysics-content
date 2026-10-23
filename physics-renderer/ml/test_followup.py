import json
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from diagnostics import diagnostic_report, ess
from bayes import PhysicsLikelihood
from bayes_followup import GroupedLikelihood, sample_followup
from hii_sim.loaders import NPEInputDataset


def test_rank_diagnostics_detect_shift_scale_stuck_and_autocorrelation():
    rng = np.random.default_rng(35)
    x = rng.normal(size=(4, 4000, 1))
    assert diagnostic_report(x, ['x'])['passed']
    for changed in (x + np.array([0, 0, 0, 2])[:, None, None],
                    x * np.array([1, 1, 1, 4])[:, None, None], np.ones_like(x)):
        assert not diagnostic_report(changed, ['x'])['passed']
    ar = rng.normal(size=(4, 20000))
    for j in range(1, ar.shape[1]):
        ar[:, j] += .9*ar[:, j-1]
    # AR(1) theoretical ESS=N*(1-rho)/(1+rho); finite estimate is noisy.
    assert .6 < ess(ar)/(ar.size*.1/1.9) < 1.4


def test_grouped_whitening_is_identical_to_original():
    config = json.loads(Path('configs/compact_proxy_gl4_v3.json').read_text())
    root = Path('output/training_readiness_20261021/candidate_gl4_512_01/public')
    inp = NPEInputDataset(root, split='validation')[0]
    original, grouped = PhysicsLikelihood(inp, config), GroupedLikelihood(inp, config)
    rng = np.random.default_rng(19)
    for _ in range(4):
        delta = rng.normal(size=len(original.y))
        assert_allclose(original.whiten(delta), grouped.whiten(delta), rtol=1e-14, atol=1e-12)
        t = np.r_[rng.uniform(size=7), rng.normal(size=2)]
        assert_allclose(original.logp(t), grouped.logp(t), rtol=1e-13)


def test_fixed_mixture_sampler_recovers_known_joint_density(tmp_path):
    class KnownGaussian:
        config = {'sampling': {'ranges': {n: [0., 1., 'linear'] for n in
                  ['em0_pc_cm6', 'te_k', 'major_fwhm_arcsec', 'axis_ratio']}}}
        center = np.r_[np.full(7, .5), 0., 0.]
        scale = np.r_[np.full(7, .03), 1., 1.]
        def residual(self, t):
            return (t-self.center)/self.scale
        def logp(self, t):
            return -.5*sum(self.residual(t)**2)
    like = KnownGaussian()
    pilot = like.center+like.scale*np.random.default_rng(10).normal(size=(4, 1000, 9))
    # Deliberately offset/underscale proposal to test the Hastings correction.
    pilot += like.scale*.5
    pilot = like.center + .8*(pilot-like.center)
    settings = json.loads(Path('ml/followup_plan_v1.json').read_text())
    settings.update(warmup=500, max_draws_per_chain=4000, diagnostic_every=4000)
    chains, report = sample_followup(like, pilot, settings, 398, tmp_path, 'known')
    physical = chains.copy()
    physical[:, :, 4] %= 1
    z = (physical-like.center)/like.scale
    assert_allclose(z.mean(axis=(0, 1)), np.zeros(9), atol=.12)
    assert_allclose(z.std(axis=(0, 1)), np.ones(9), atol=.12)
    assert report['diagnostics']['passed']
