from dataclasses import replace
import numpy as np
import pytest
from hii_sim.schema import Grid, Instrument, Sky, Source
from hii_sim.instrument import operator, coverage_diagnostics, expected_measurement
from hii_sim.instrument_validation import direct_pixel_visibilities, bandwidth_reference
from hii_sim.physics import render


def test_report_actual_baselines_instead_of_requested_lower_cut():
    grid, inst = Grid(), Instrument()
    report = coverage_diagnostics(grid, inst)
    assert np.isclose(report["fourier_spacing_wavelengths"], 859.4366927)
    assert report["channels"][0]["actual_min_nonzero_baseline_m"] > 200
    assert not report["channels"][0]["dc_measured"]
    assert report["channels"][0]["requested_min_baseline_m"] == 35
    with pytest.raises(ValueError, match="misses required short spacings"):
        operator(grid, replace(inst, max_short_spacing_ratio=1.5))
    operator(grid, replace(inst, min_baseline_m=250, max_short_spacing_ratio=1.5))


def test_direct_dft_matches_fft_phase_flux_and_units():
    grid = Grid(32, 7.5, 2)
    inst = Instrument(frequencies_ghz=(1.44,), bandwidths_mhz=(8,), sampling_fraction=1)
    source = Source(1.2e5, 8000, 32, .7, 37)
    cube = render(source, grid, Sky(east_arcsec=11), [1.44], [0], 1)
    weights, pb = operator(grid, inst)
    cells = np.argwhere(weights[0] > 0)[:15]
    uv = np.fft.fftfreq(grid.npix, np.deg2rad(grid.pixel_arcsec/3600))
    direct = direct_pixel_visibilities(cube[0]*pb[0], grid, np.column_stack((uv[cells[:,1]], uv[cells[:,0]])))
    fft = expected_measurement(cube, grid, inst)[0][tuple(cells.T)]
    assert np.allclose(direct, fft, rtol=1e-10, atol=1e-13)


def test_bandwidth_reference_converges_and_zero_width_limit():
    grid = Grid(32, 7.5, 2)
    inst = Instrument(frequencies_ghz=(1.44,), bandwidths_mhz=(128,))
    source, sky = Source(1.2e5, 8000, 32, .7, 37), Sky(east_arcsec=10)
    uv = np.array([[1000, 2000], [3000, 1000]])
    ref3 = bandwidth_reference(source, grid, sky, inst, 0, uv, 3)
    ref7 = bandwidth_reference(source, grid, sky, inst, 0, uv, 7)
    assert np.linalg.norm(ref3-ref7)/np.linalg.norm(ref7) < 1e-5
    narrow = bandwidth_reference(source, grid, sky, replace(inst, bandwidths_mhz=(1e-6,)), 0, uv, 3)
    _, pb = operator(grid, inst)
    plane = render(source, grid, sky, [1.44], [0], 1)[0]
    monochromatic = direct_pixel_visibilities(plane*pb[0], grid, uv)
    assert np.allclose(narrow, monochromatic, rtol=1e-10, atol=1e-14)
