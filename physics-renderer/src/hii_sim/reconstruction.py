"""Measurement-only dirty imaging and a transparent Högbom CLEAN baseline.

All convolutions use the same periodic grid as the fast measurement operator.
No source parameters, true images, simulation seeds, or labels are accepted.
"""
import numpy as np
from scipy.ndimage import label


def dirty_image(visibilities, weights):
    """Unit-peak PSF normalization. vis/weights have [frequency,y,x] shape."""
    visibilities, weights = np.asarray(visibilities), np.asarray(weights)
    if (visibilities.ndim != 3 or visibilities.shape != weights.shape
            or np.any(weights < 0) or not np.isfinite(weights).all()
            or not np.isfinite(visibilities).all()):
        raise ValueError("Measurement/weight mismatch")
    norm = weights.mean(axis=(-2,-1))[:,None,None]
    if np.any(norm <= 0):
        raise ValueError("No supported measurements")
    return np.fft.fftshift(np.fft.ifft2(visibilities*weights, axes=(-2,-1)),
                           axes=(-2,-1)).real/norm


def point_spread_function(weights):
    return dirty_image(np.ones_like(weights, dtype=complex), weights)


def _circular_convolve(model, kernel):
    return np.fft.ifft2(np.fft.fft2(model)*np.fft.fft2(np.fft.ifftshift(kernel))).real


def fit_clean_beam(psf, pixel_arcsec=1.):
    """Fit a unit-peak elliptical Gaussian to the connected PSF core >= 0.5.

    This is a deterministic small-field baseline, not CASA's beam fitter. A
    poorly sampled/non-Gaussian central lobe is rejected rather than replaced
    with an undocumented width. PA is measured north through east; x is west.
    """
    psf = np.asarray(psf, dtype=float)
    if psf.ndim != 2 or not np.isfinite(psf).all() or pixel_arcsec <= 0:
        raise ValueError("Finite 2-D PSF and positive pixel size required")
    cy, cx = np.array(psf.shape)//2
    if not np.isclose(psf[cy, cx], 1., rtol=1e-6, atol=1e-8):
        raise ValueError("PSF must have unit central peak")
    connected, _ = label(psf >= .5)
    selected = connected == connected[cy, cx]
    y, x = np.indices(psf.shape, dtype=float)
    x -= cx; y -= cy
    selected[cy, cx] = False
    design = np.column_stack((x[selected]**2, 2*x[selected]*y[selected], y[selected]**2))
    if len(design) < 5 or np.linalg.matrix_rank(design) < 3:
        raise ValueError("PSF core is undersampled; use smaller image pixels")
    a, b, d = np.linalg.lstsq(design, -2*np.log(psf[selected]), rcond=None)[0]
    precision = np.array([[a,b],[b,d]])
    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    if np.min(eigenvalues) <= 0:
        raise ValueError("PSF core does not support a positive Gaussian beam fit")
    beam = np.exp(-.5*(a*x*x+2*b*x*y+d*y*y))
    fwhm = np.sqrt(8*np.log(2)/eigenvalues)*pixel_arcsec
    major_vector = eigenvectors[:, 0]
    pa = float(np.degrees(np.arctan2(-major_vector[0], major_vector[1])) % 180)
    return beam, {"major_fwhm_arcsec": float(fwhm[0]),
                  "minor_fwhm_arcsec": float(fwhm[1]), "pa_deg": pa,
                  "area_pixels": float(beam.sum()),
                  "core_fit_relative_l2": float(np.linalg.norm((beam-psf)[selected])/
                                                 np.linalg.norm(psf[selected])),
                  "fit_definition": "least-squares log Gaussian on connected PSF >= 0.5"}


def hogbom(dirty, psf, gain=.1, niter=1000, threshold=0., restore_sigma_pixels=None,
           clean_mask=None, pixel_arcsec=1.):
    """Signed point-component CLEAN, with beam derived from the measured PSF.

    Components are Jy/pixel; the model restoration is Jy/clean beam and the
    residual is Jy/dirty beam. The mixed image is retained for display only.
    CLEAN masks/thresholds must be chosen from observations or a predeclared
    imaging policy, never the hidden source. A supplied sigma is explicit
    legacy compatibility, recorded in the beam metadata. No flux extrapolation
    or short-spacing correction is performed.
    """
    dirty, psf = np.asarray(dirty, dtype=float), np.asarray(psf, dtype=float)
    if (dirty.ndim != 2 or dirty.shape != psf.shape or not np.isfinite(dirty).all()
            or not np.isfinite(psf).all() or not 0 < gain <= 1 or niter < 0
            or int(niter) != niter or threshold < 0 or not np.isfinite(threshold)):
        raise ValueError("Invalid dirty image, PSF, or CLEAN controls")
    center = np.array(psf.shape)//2
    if not np.isclose(psf[tuple(center)], 1., rtol=1e-6, atol=1e-8):
        raise ValueError("PSF must have unit central peak")
    mask = np.ones(dirty.shape, dtype=bool) if clean_mask is None else np.asarray(clean_mask, dtype=bool)
    if mask.shape != dirty.shape or not mask.any():
        raise ValueError("CLEAN mask must match image and contain eligible pixels")
    residual = dirty.copy(); model = np.zeros_like(dirty)
    iterations = 0; reason = "iteration_limit"
    for _ in range(int(niter)):
        pos = np.unravel_index(np.where(mask, np.abs(residual), -np.inf).argmax(), residual.shape)
        if abs(residual[pos]) <= threshold:
            reason = "threshold"; break
        component = gain*residual[pos]
        model[pos] += component
        residual -= component*np.roll(psf, np.array(pos)-center, axis=(0,1))
        iterations += 1
    # Recompute the residual from the measurement-domain equivalent convolution,
    # avoiding iteration-roundoff accumulation and making closure inspectable.
    residual = dirty-_circular_convolve(model, psf)
    if restore_sigma_pixels is None:
        beam, beam_metadata = fit_clean_beam(psf, pixel_arcsec)
    else:
        if restore_sigma_pixels <= 0:
            raise ValueError("Restoring sigma must be positive")
        yy,xx = np.indices(psf.shape)-center[:,None,None]
        beam = np.exp(-(xx*xx+yy*yy)/(2*restore_sigma_pixels**2))
        beam_metadata = {"major_fwhm_arcsec": float(np.sqrt(8*np.log(2))*restore_sigma_pixels*pixel_arcsec),
                         "minor_fwhm_arcsec": float(np.sqrt(8*np.log(2))*restore_sigma_pixels*pixel_arcsec),
                         "pa_deg": 0., "area_pixels": float(beam.sum()),
                         "fit_definition": "explicit user-supplied circular sigma"}
    restored = _circular_convolve(model, beam)
    return {"components_jy": model, "residual_jy_dirty_beam": residual,
            "restored_jy_clean_beam": restored, "image_mixed_beams": restored+residual,
            "clean_beam": beam, "beam": beam_metadata,
            "iterations": iterations, "stop_reason": reason,
            "residual_peak_jy_dirty_beam": float(np.max(np.abs(residual))),
            "residual_rms_jy_dirty_beam": float(np.sqrt(np.mean(residual**2)))}


def aperture_flux_diagnostic(result, psf, aperture, dirty_beam_radius_pixels):
    """Explicit finite-aperture component and residual bookkeeping.

    Residual flux = sum(residual in aperture) / sum(PSF in central disk). The
    finite-disk PSF area is a convention when DC is absent, not an independent
    measurement of missing total flux. Vary this radius in scientific checks.
    No combined restored image is implicitly assumed to have one beam unit.
    """
    aperture = np.asarray(aperture, dtype=bool)
    if aperture.shape != psf.shape or dirty_beam_radius_pixels <= 0:
        raise ValueError("Matching aperture and positive beam integration radius required")
    yy, xx = np.indices(psf.shape)-np.array(psf.shape)[:, None, None]//2
    beam_area = float(psf[np.hypot(xx, yy) <= dirty_beam_radius_pixels].sum())
    residual_sum = float(result["residual_jy_dirty_beam"][aperture].sum())
    components = float(result["components_jy"][aperture].sum())
    valid = beam_area > 1e-10
    residual = residual_sum/beam_area if valid else None
    return {"component_flux_jy": components,
            "residual_pixel_sum_jy_dirty_beam": residual_sum,
            "dirty_beam_area_pixels": beam_area,
            "dirty_beam_radius_pixels": float(dirty_beam_radius_pixels),
            "residual_aperture_estimate_jy": residual,
            "component_plus_residual_estimate_jy": components+residual if valid else None,
            "flux_interpretation": "finite-aperture beam-area convention; no missing-spacing correction"}


def reconstruct_measurements(visibilities, weights, pixel_arcsec, *, rms_jy_dirty_beam,
                             gain=.1, niter=3000, threshold_sigma=3., clean_mask=None):
    """Reconstruct each band from measured visibilities and observing metadata.

    The default stopping threshold is 3 times the declared thermal dirty-image
    RMS. Calibration artifacts are not thermal noise; residuals must still be
    inspected. Images are apparent sky (PB uncorrected). Outputs preserve
    components, restoring beam, and residual independently for honest analysis.
    """
    if rms_jy_dirty_beam < 0 or threshold_sigma < 0:
        raise ValueError("Nonnegative RMS and threshold multiplier required")
    dirty = dirty_image(visibilities, weights)
    psf = point_spread_function(weights)
    channels = [hogbom(d, p, gain=gain, niter=niter,
                       threshold=rms_jy_dirty_beam*threshold_sigma,
                       clean_mask=clean_mask, pixel_arcsec=pixel_arcsec)
                for d,p in zip(dirty, psf)]
    return {"dirty_jy_dirty_beam": dirty, "psf": psf, "channels": channels,
            "policy": {"algorithm": "signed_hogbom_periodic_v2", "gain": gain,
                       "niter": niter, "threshold_sigma": threshold_sigma,
                       "threshold_jy_dirty_beam": rms_jy_dirty_beam*threshold_sigma,
                       "primary_beam_corrected": False,
                       "requires_truth": False}}
