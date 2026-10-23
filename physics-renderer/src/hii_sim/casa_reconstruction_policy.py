"""Proposed measurement-only CASA multiscale controls; never runs CASA.

This is an experimental frozen comparison policy, not a production default.
Inputs are observed beam/grid/uv geometry and an observed noise proxy only.
"""
import math


def multiscale_policy(*, beam_major_arcsec, cell_arcsec, npix,
                      min_usable_uv_lambda, observed_border_rms_jy_beam):
    values = (beam_major_arcsec, cell_arcsec, min_usable_uv_lambda,
              observed_border_rms_jy_beam)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Policy requires positive finite measured beam, cell, shortest usable uv radius and noise proxy")
    if isinstance(npix, bool) or not isinstance(npix, int) or npix < 16:
        raise ValueError("Policy requires integer square-image npix >= 16")
    beam_pixels = beam_major_arcsec/cell_arcsec
    pixel_radians = math.radians(cell_arcsec/3600.)
    # The half reciprocal-baseline angle is a conservative policy bound, not
    # a measured largest recoverable scale or a guarantee of recovered flux.
    cap_pixels = min(npix/4., 0.5/(min_usable_uv_lambda*pixel_radians))
    scales = sorted({0, *(max(1, int(math.floor(factor*beam_pixels+0.5)))
                          for factor in (1., 3., 9.)
                          if max(1, int(math.floor(factor*beam_pixels+0.5))) <= cap_pixels)})
    if len(scales) < 2:
        raise ValueError("No nonzero beam-scale component fits field/short-spacing bounds")
    return {"policy_id": "observable_multiscale_comparison_v1",
            "status": "proposed_unvalidated_not_production",
            "inputs": {"beam_major_arcsec": float(beam_major_arcsec), "cell_arcsec": float(cell_arcsec),
                       "npix": npix, "min_usable_uv_lambda": float(min_usable_uv_lambda),
                       "observed_border_rms_jy_beam": float(observed_border_rms_jy_beam)},
            "scale_diameter_cap_pixels": cap_pixels,
            "scale_rule": "round half up of 0,1,3,9 times measured major-beam FWHM/cell; retain scales <= min(npix/4,0.5/(u_min*pixel_radians)); CASA sizes are kernel diameters, not source FWHM labels",
            "tclean_overrides": {"deconvolver": "multiscale", "scales": scales, "smallscalebias": 0.0,
                                 "gain": 0.1, "niter": 5000,
                                 "threshold": "%.12gJy" % (3.*observed_border_rms_jy_beam),
                                 "nsigma": 0.0, "fullsummary": True, "usemask": "user", "mask": "",
                                 "savemodel": "none", "restart": False},
            "scope": "Same noisy DATA, natural weights, grid, PB handling and full-field component-center mask as baseline; fresh output required. Border MAD can contain sidelobes; not calibrated thermal noise. No truth, density/EM/temperature/size labels, startmodel or source mask."}


def configured_multiscale_policy(*, noise_multiplier=3.0, **observables):
    """Supported coarse development policy with explicit noise multiplier.

    The frozen v1 function and artifacts remain reproducible. This wrapper
    exposes configuration without changing its coarse geometry algorithm.
    """
    if (isinstance(noise_multiplier, bool) or not isinstance(noise_multiplier, (int, float))
            or not math.isfinite(noise_multiplier) or noise_multiplier <= 0):
        raise ValueError("noise_multiplier must be finite and positive")
    result = multiscale_policy(**observables)
    result.update(policy_id="observable_coarse_multiscale_v2",
                  status="development_supported_fidelity_not_certified",
                  noise_multiplier=float(noise_multiplier))
    result["tclean_overrides"]["threshold"] = "%.12gJy" % (
        noise_multiplier * observables["observed_border_rms_jy_beam"])
    return result
