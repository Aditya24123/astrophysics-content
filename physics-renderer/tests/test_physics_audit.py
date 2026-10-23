import numpy as np
import pytest
from astropy import units as u
from hii_sim.physics import optical_depth, intensity, render, RJ_JY_SR
from hii_sim.schema import Source, Sky, Grid


def test_units_thin_and_thick_limits():
    with pytest.raises(u.UnitConversionError):
        optical_depth(1*u.m, 8000*u.K, 1.44*u.GHz)
    em = np.array([1e-8, 1e12])*u.pc/u.cm**6
    tau = optical_depth(em, 8000*u.K, 1.44*u.GHz)
    tb = intensity(em, 8000*u.K, 1.44*u.GHz).value/(RJ_JY_SR*(1.44e9)**2)
    assert tb[0]/(8000*tau[0]) == pytest.approx(1., rel=1e-10)
    assert tb[1]/8000 == pytest.approx(1., rel=1e-14)


def test_compact_non_gaussian_subpixel_refinement():
    s = Source(1e7, 8000, 12, .3, 113, "ellipsoid_em")
    sky = Sky(east_arcsec=1, north_arcsec=-.75)
    planes = {n: render(s, Grid(64, 2.5, n), sky, [1.44], [0], 1)[0] for n in (2,4,8,16)}
    errors = {n: np.linalg.norm(planes[n]-planes[16])/np.linalg.norm(planes[16]) for n in (2,4,8)}
    assert errors[8] < errors[4] < errors[2]
    assert errors[8] < .012
    assert np.isfinite(planes[16]).all() and planes[16].min() >= 0


def test_validation_records_observed_error_and_code(monkeypatch, tmp_path):
    import hii_sim.validation as validation
    monkeypatch.setattr(validation, "representative_sensitivity", lambda *args: {})
    report = validation.validate("configs/base.json", tmp_path)
    checks = report["checks"]
    measured = max(x["relative_error"] for x in checks["transfer_ode_errors"])
    assert checks["transfer_ode_max_relative_error"] == measured
    assert measured != 2e-10
    assert len(report["provenance"]["code_sha256"]) == 64
    assert report["provenance"]["configuration"]["grid"]["subpixels"] == 2
