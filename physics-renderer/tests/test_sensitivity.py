from dataclasses import replace
import numpy as np
from hii_sim.schema import Grid, Instrument, Source, Sky
from hii_sim.instrument import operator, independent_fourier_cells, noise_sigma_real, observe
from hii_sim.sensitivity import whiten_columns, local_diagnostic, scaled_measurement_jacobian


def test_conjugates_count_once_and_self_cell_variance():
    n = 16
    values = np.ones((1, n, n, 1), dtype=complex)
    weights = np.ones((1, n, n))
    unique, self_cells = independent_fourier_cells(n)
    white = whiten_columns(values, weights, [1.], 0.)
    # Real+imag Hermitian field has n*n independent real degrees of freedom.
    assert len(white) == n*n
    assert (unique & self_cells).sum() == 4
    # For a real constant value at each uv, information = n*n/2, not n*n.
    assert np.isclose((white*white).sum(), n*n/2)


def test_cross_frequency_whitening_matches_explicit_covariance():
    values = np.zeros((2, 16, 16, 2), complex)
    weights = np.zeros((2, 16, 16))
    weights[:, 1, 2] = weights[:, -1, -2] = 1
    signal = np.array([[1+2j, 4+1j], [3-1j, 2+5j]])
    values[:, 1, 2] = signal
    values[:, -1, -2] = signal.conj()
    covariance = np.array([[4., .3*2*3], [.3*2*3, 9.]])
    white = whiten_columns(values, weights, [2., 3.], .3)
    explicit = signal.real.T@np.linalg.solve(covariance, signal.real)+signal.imag.T@np.linalg.solve(covariance, signal.imag)
    assert np.allclose(white.T@white, explicit)


def test_generated_noise_channel_covariance():
    grid = Grid(16, 20, 1)
    inst = Instrument(frequencies_ghz=(1.1, 1.8), bandwidths_mhz=(8, 8), sampling_fraction=1,
                      calibration_fraction=0, calibration_slope_std=0)
    w, _ = operator(grid, inst)
    unique, self_cells = independent_fourier_cells(grid.npix)
    y, x = np.argwhere((w > 0).all(axis=0) & unique & ~self_cells)[0]
    sigma = noise_sigma_real(grid, inst)
    values = np.array([observe(np.zeros_like(w), grid, inst, seed)[0]["vis_jy"][:, y, x].real/sigma
                       for seed in range(1500)])
    assert np.max(abs(np.cov(values.T)-np.array([[1, .25], [.25, 1]]))) < .09


def test_calibration_reduces_information_and_circular_pa_vanishes():
    grid = Grid(32, 7.5, 2)
    inst = Instrument()
    d = local_diagnostic(Source(1.2e5, 8000, 32, 1, 37), grid, Sky(), inst)
    thermal = np.array(d["thermal_information"])
    marginal = np.array(d["calibration_marginal_information"])
    assert np.linalg.eigvalsh(thermal-marginal).min() > -1e-5
    assert d["scaled_response_snr_for_delta_0p01"][-1] < 1e-8


def test_log_coordinates_are_invariant_to_em_unit_rescaling():
    from hii_sim.instrument import expected_measurement
    from hii_sim.physics import render
    grid, inst, sky = Grid(32, 7.5, 2), Instrument(), Sky()
    source = Source(1.2e5, 8000, 32, .72, 37)
    _, jac, _ = scaled_measurement_jacobian(source, grid, sky, inst)
    # Re-express EM as millions of pc cm^-6, differentiate linearly in that
    # coordinate, then convert to its logarithm. Old raw-unit columns fail.
    em_million = source.em0_pc_cm6/1e6
    delta = em_million*1e-4
    def value(x):
        cube = render(replace(source, em0_pc_cm6=x*1e6), grid, sky, inst.frequencies_ghz, inst.bandwidths_mhz)
        return expected_measurement(cube, grid, inst)
    rescaled = (value(em_million+delta)-value(em_million-delta))/(2*delta)*em_million
    assert np.linalg.norm(jac[..., 0]-rescaled)/np.linalg.norm(rescaled) < 1e-6
