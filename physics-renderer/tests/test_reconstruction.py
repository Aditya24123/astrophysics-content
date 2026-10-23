"""Regression checks for measurement-only reconstruction and flux accounting."""
import inspect
import numpy as np
import pytest

from hii_sim.reconstruction import (aperture_flux_diagnostic, dirty_image,
    fit_clean_beam, hogbom, point_spread_function, reconstruct_measurements)
from hii_sim.reconstruction_validation import _fixture, _legacy_weights


def test_psf_derived_beam_recovers_elliptical_width():
    y,x=np.indices((64,64))-32
    psf=np.exp(-.5*((x/4.)**2+(y/6.)**2))
    beam,metadata=fit_clean_beam(psf,pixel_arcsec=2.)
    np.testing.assert_allclose(beam,psf,rtol=1e-11,atol=1e-15)
    assert metadata["major_fwhm_arcsec"]==pytest.approx(np.sqrt(8*np.log(2))*6*2)
    assert metadata["minor_fwhm_arcsec"]==pytest.approx(np.sqrt(8*np.log(2))*4*2)


def test_offset_point_clean_conserves_flux_and_residual_measurement_closure():
    weights=_legacy_weights()[None]
    sky=np.zeros((1,96,96)); sky[0,39,57]=.01
    vis=np.fft.fft2(np.fft.ifftshift(sky,axes=(-2,-1)),axes=(-2,-1))
    vis=np.where(weights>0,vis,0.)
    result=reconstruct_measurements(vis,weights,2.5,rms_jy_dirty_beam=1e-9,
                                    threshold_sigma=1,niter=300,gain=.2)
    clean=result["channels"][0]
    assert clean["stop_reason"]=="threshold"
    assert clean["components_jy"].sum()==pytest.approx(.01,abs=2e-9)
    assert np.unravel_index(clean["components_jy"].argmax(),(96,96))==(39,57)
    assert clean["restored_jy_clean_beam"].sum()/clean["beam"]["area_pixels"]==pytest.approx(clean["components_jy"].sum())
    prediction=np.fft.fft2(np.fft.ifftshift(clean["components_jy"]))
    residual=dirty_image(vis-prediction[None],weights)[0]
    np.testing.assert_allclose(clean["residual_jy_dirty_beam"],residual,atol=2e-17)
    # The public entry point takes only measurements, observing metadata, and
    # imaging controls; it cannot accept an unnoticed source/label argument.
    assert "source" not in inspect.signature(reconstruct_measurements).parameters
    with pytest.raises(TypeError):
        reconstruct_measurements(vis,weights,2.5,rms_jy_dirty_beam=1e-9,source={"em0":1e5})


def test_extended_flux_includes_residual_when_zero_spacing_measured():
    weights=_legacy_weights(include_dc=True)[None]
    sky=_fixture(96,2.5,"gaussian_90")
    vis=np.fft.fft2(np.fft.ifftshift(sky))[None]
    psf=point_spread_function(weights)[0]
    result=hogbom(dirty_image(vis,weights)[0],psf,gain=.2,niter=50,
                   threshold=0.,pixel_arcsec=2.5)
    # Deliberately early stopping leaves most extended flux in the residual.
    component_flux=result["components_jy"].sum()
    residual_flux=result["residual_jy_dirty_beam"].sum()/psf.sum()
    assert component_flux < .008
    assert residual_flux > .002
    assert component_flux+residual_flux==pytest.approx(.01,abs=1e-14)


def test_missing_zero_spacing_has_exact_total_flux_null_space():
    weights=_legacy_weights()
    sky=_fixture(96,2.5,"gaussian_90")
    changed=sky+.005/sky.size
    v1=np.fft.fft2(np.fft.ifftshift(sky));v2=np.fft.fft2(np.fft.ifftshift(changed))
    np.testing.assert_allclose(v1[weights>0],v2[weights>0],atol=2e-18)
    assert changed.sum()-sky.sum()==pytest.approx(.005)
    psf=point_spread_function(weights[None])[0]
    assert abs(psf.sum())<1e-12


def test_legacy_extended_component_result_and_residual_are_separate():
    weights=_legacy_weights()[None]
    sky=_fixture(96,2.5,"gaussian_90")
    y,x=(np.indices((96,96))-48)*2.5
    sky*=np.exp(-4*np.log(2)*(x*x+y*y)/(42*60/1.06)**2)
    vis=np.fft.fft2(np.fft.ifftshift(sky))[None]
    psf=point_spread_function(weights)[0]
    result=hogbom(dirty_image(vis,weights)[0],psf,gain=.2,niter=3000,
                   threshold=1e-6,pixel_arcsec=2.5)
    assert result["components_jy"].sum()==pytest.approx(.002423,abs=1e-6)
    aperture=np.hypot(x,y)<=100
    photometry=aperture_flux_diagnostic(result,psf,aperture,10)
    assert photometry["residual_aperture_estimate_jy"] is not None
    assert photometry["component_plus_residual_estimate_jy"] != photometry["component_flux_jy"]
    assert "no missing-spacing correction" in photometry["flux_interpretation"]


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        dirty_image(np.ones((2,4,4)),np.zeros((2,4,4)))
    with pytest.raises(ValueError):
        dirty_image(np.ones((4,4)),np.ones((4,4)))
    with pytest.raises(ValueError):
        fit_clean_beam(np.eye(20))
