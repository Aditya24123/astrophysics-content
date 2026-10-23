import json
import numpy as np
from numpy.testing import assert_allclose
from model import Support, initialize, loss_grad, predict, sample, covariance, fit_pca, project
from bayes import PhysicsLikelihood, diagnostics
from hii_sim.loaders import NPEInputDataset
from pathlib import Path


def test_resource_monitor_does_not_block_training():
    from run_initial import memory
    result = memory()
    assert isinstance(result, dict) and result


def test_network_gradient_central_difference():
    rng = np.random.default_rng(12)
    x, y = rng.normal(size=(7, 3)), rng.normal(size=(7, 4))
    p = initialize(3, 5, rng)
    _, grad = loss_grad(x, y, p, .003)
    for name in p:
        for index in np.ndindex(p[name].shape):
            old = p[name][index]
            p[name][index] = old+1e-5
            plus = loss_grad(x, y, p, .003)[0]
            p[name][index] = old-1e-5
            minus = loss_grad(x, y, p, .003)[0]
            p[name][index] = old
            assert_allclose(grad[name][index], (plus-minus)/2e-5, atol=1e-7, rtol=1e-5)


def test_sampling_covariance_and_support():
    rng = np.random.default_rng(7)
    p = initialize(3, 5, rng)
    p['b2'][5:8] += .4
    x = np.ones((1, 3))
    mu, a, _ = predict(x, p)
    draws = sample(x, p, 100000, rng)[0]
    assert_allclose(draws.mean(axis=0), mu[0], atol=.025)
    assert_allclose(np.cov(draws.T), covariance(a)[0], atol=.025)
    support = Support(json.loads(Path('configs/compact_proxy_gl4_v3.json').read_text()))
    assert_allclose(support.encode(support.decode(draws[:100])), draws[:100], atol=1e-10)
    values = support.decode(draws)
    assert np.all((values[:, 1] >= 5000) & (values[:, 1] <= 12000))
    assert np.all((values[:, 3] >= .5) & (values[:, 3] <= 1))


def test_pca_has_no_label_argument_and_repeatable():
    x = np.random.default_rng(2).normal(size=(40, 60))
    a, b = fit_pca(x, 5), fit_pca(x.copy(), 5)
    assert_allclose(project(x, a), project(x, b))
    assert_allclose(project(x, a).mean(axis=0), 0, atol=1e-12)


def test_accelerated_forward_matches_original_over_prior():
    config = json.loads(Path('configs/compact_proxy_gl4_v3.json').read_text())
    public = Path('output/training_readiness_20261021/candidate_gl4_512_01/public')
    inp = NPEInputDataset(public, split='validation')[0]
    like = PhysicsLikelihood(inp, config)
    rng = np.random.default_rng(9)
    points = [np.r_[np.full(7, .5), 0, 0], np.r_[[1, 0, 0, 0, .9, 1, 0], 2, -2]]
    points += [np.r_[rng.uniform(size=7), rng.normal(size=2)] for _ in range(6)]
    for p in points:
        accelerated, reference = like.vector(p), like.vector(p, reference=True)
        assert np.linalg.norm(like.whiten(accelerated-reference)) < 1e-5


def test_thermal_whitening_and_mcmc_diagnostic():
    config = json.loads(Path('configs/compact_proxy_gl4_v3.json').read_text())
    public = Path('output/training_readiness_20261021/candidate_gl4_512_01/public')
    like = PhysicsLikelihood(NPEInputDataset(public, split='validation')[0], config)
    for idx, inv in like.groups:
        cov = like.o.thermal.covariance(like.m.frequency_index[idx],
                                     self_conjugate=bool(like.m.self_conjugate[idx[0]]))
        assert_allclose(inv @ cov @ inv.T, np.eye(len(idx)), atol=1e-12)
    chains = np.random.default_rng(1).normal(size=(4, 1000, 4))
    rhat, ess = diagnostics(chains)
    assert np.all(rhat < 1.02) and np.all(ess > 200)
    chains[0] += 4
    assert np.all(diagnostics(chains)[0] > 1.2)
