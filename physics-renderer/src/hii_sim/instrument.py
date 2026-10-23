"""Fast *artificial gridded Fourier* telescope. This is not a VLA simulator."""
from dataclasses import asdict
from functools import lru_cache
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.constants import c
from .schema import Grid, Instrument
from .reconstruction import dirty_image, point_spread_function


def primary_beam(grid, frequency_ghz, coefficient=42.):
    east, north = grid.coordinates()
    return np.exp(-4*np.log(2)*(east**2+north**2)/(coefficient*60/frequency_ghz)**2)


def _hermitian_partner(a):
    ids = (-np.arange(a.shape[-1])) % a.shape[-1]
    return a[..., ids[:,None], ids[None,:]]


@lru_cache(maxsize=8)
def _operator(grid_json, instrument_json):
    grid = Grid(**json.loads(grid_json)); inst = Instrument(**json.loads(instrument_json))
    n = grid.npix
    uv = np.fft.fftfreq(n, np.deg2rad(grid.pixel_arcsec/3600))
    v, uu = np.meshgrid(uv, uv, indexing="ij")
    radius = np.hypot(uu, v)
    sigma = np.deg2rad(inst.taper_fwhm_arcsec/3600)/np.sqrt(8*np.log(2))
    taper = np.exp(-2*np.pi**2*sigma**2*radius**2)
    rng = np.random.default_rng(inst.sampling_seed)
    # Shared synthetic pattern in wavelength cells, not antenna tracks.
    uniform = rng.uniform(size=(n,n))
    uniform = np.minimum(uniform, _hermitian_partner(uniform))
    # min(U,V) CDF is 1-(1-x)^2; ensures requested fraction for nonself cells.
    mask = uniform < 1-np.sqrt(1-inst.sampling_fraction)
    weights, beams = [], []
    for f in inst.frequencies_ghz:
        wavelength = c.value/(f*1e9)
        support = (radius >= inst.min_baseline_m/wavelength) & (radius <= inst.max_baseline_m/wavelength)
        w = np.where(mask & support, taper, 0.)
        if not np.any(w):
            raise ValueError("Empty uv coverage; increase field/grid or baseline range")
        actual_min = float(radius[(w > 0) & (radius > 0)].min()*wavelength)
        if inst.max_short_spacing_ratio is not None and actual_min > inst.min_baseline_m*inst.max_short_spacing_ratio:
            raise ValueError(f"Fourier grid misses required short spacings at {f} GHz: "
                             f"actual minimum {actual_min:.3f} m, requested cut {inst.min_baseline_m} m. "
                             "Increase field of view or explicitly select a compact-source scope.")
        weights.append(w)
        beams.append(primary_beam(grid, f, inst.pb_fwhm_arcmin_ghz))
    w = np.asarray(weights); pb = np.asarray(beams)
    w.setflags(write=False); pb.setflags(write=False)
    return w, pb


def operator(grid: Grid, inst: Instrument):
    return _operator(json.dumps(asdict(grid),sort_keys=True), json.dumps(asdict(inst),sort_keys=True))


def independent_fourier_cells(n):
    """One representative of each Hermitian pair; include real-only self cells."""
    y, x = np.indices((n, n))
    index = y*n+x
    partner = ((-y) % n)*n+((-x) % n)
    return index <= partner, index == partner


def coverage_diagnostics(grid, inst):
    """Describe actual sampled cells, never equate a baseline cut with coverage."""
    w, _ = operator(grid, inst)
    uv = np.fft.fftfreq(grid.npix, np.deg2rad(grid.pixel_arcsec/3600))
    radius = np.hypot(uv[:, None], uv[None, :])
    unique, self_cells = independent_fourier_cells(grid.npix)
    channels = []
    for f, weight in zip(inst.frequencies_ghz, w):
        sampled = weight > 0
        nonzero = sampled & (radius > 0)
        wavelength = c.value/(f*1e9)
        minimum = float(radius[nonzero].min()*wavelength)
        channels.append({"frequency_ghz": float(f),
                         "actual_min_nonzero_baseline_m": minimum,
                         "actual_max_baseline_m": float(radius[sampled].max()*wavelength),
                         "requested_min_baseline_m": inst.min_baseline_m,
                         "min_ratio_to_request": minimum/inst.min_baseline_m if inst.min_baseline_m else None,
                         "independent_complex_cells": int((sampled & unique & ~self_cells).sum()),
                         "independent_real_only_cells": int((sampled & unique & self_cells).sum()),
                         "dc_measured": bool(sampled[0, 0]),
                         "inverse_min_uv_arcsec": float(np.rad2deg(1/radius[nonzero].min())*3600)})
    return {"model": "artificial_Hermitian_Fourier_lattice_not_antenna_tracks",
            "field_arcsec": grid.npix*grid.pixel_arcsec,
            "fourier_spacing_wavelengths": float(uv[1]),
            "inverse_min_uv_note": "geometric scale 1/u_min, not a calibrated maximum recoverable scale",
            "channels": channels}


def noise_sigma_real(grid, inst):
    """Per-component sigma for non-self cells. Self cells have twice this variance."""
    w, _ = operator(grid, inst)
    factor = inst.rms_jy_dirty_beam*w.mean(axis=(-2,-1))/np.sqrt((w*w).mean(axis=(-2,-1)))
    return factor*grid.npix/np.sqrt(2)


def expected_measurement(sky_jy_sr, grid, inst):
    w, pb = operator(grid, inst)
    apparent_jy_pixel = sky_jy_sr * pb * grid.omega_sr
    vis = np.fft.fft2(np.fft.ifftshift(apparent_jy_pixel, axes=(-2,-1)), axes=(-2,-1))
    return np.where(w > 0, vis, 0.)


def observe(sky_jy_sr, grid: Grid, inst: Instrument, seed):
    """Return noisy/calibrated Fourier measurements and protected corruption draws."""
    w, pb = operator(grid, inst)
    vis = expected_measurement(sky_jy_sr, grid, inst)
    rng = np.random.default_rng(seed)
    # Positive lognormal common amplitude, mean unity; spectral slope also shared.
    s = np.sqrt(np.log1p(inst.calibration_fraction**2))
    common_gain = np.exp(rng.normal(0,s)-s*s/2)
    slope = rng.normal(0,inst.calibration_slope_std)
    f = np.asarray(inst.frequencies_ghz)
    gains = common_gain*(f/np.exp(np.log(f).mean()))**slope
    shared = rng.normal(size=(grid.npix, grid.npix))
    white = np.sqrt(inst.noise_frequency_correlation)*shared + np.sqrt(1-inst.noise_frequency_correlation)*rng.normal(size=w.shape)
    norm = w.mean(axis=(-2,-1))
    factor = inst.rms_jy_dirty_beam*norm/np.sqrt((w*w).mean(axis=(-2,-1)))
    noise = np.fft.fft2(white*factor[:,None,None], axes=(-2,-1))
    measured = np.where(w > 0, vis*gains[:,None,None]+noise, 0.)
    sigma_real = factor*grid.npix/np.sqrt(2)
    return {"vis_jy": measured, "sigma_real_jy": sigma_real}, {
        "common_gain": float(common_gain), "spectral_slope": float(slope), "seed": int(seed)}


def forward(sky_jy_sr, grid, inst, seed):
    measurement, hidden = observe(sky_jy_sr, grid, inst, seed)
    weights, pb = operator(grid, inst)
    return {"measurement": measurement,
            "dirty_jy_beam": dirty_image(measurement["vis_jy"], weights),
            "psf": point_spread_function(weights), "primary_beam": pb,
            "protected_corruption": hidden}


def ideal_gaussian_image(sky_jy_sr, grid, inst):
    """Complete-coverage Gaussian reference for approximation failure diagnostics."""
    sigma = inst.taper_fwhm_arcsec/grid.pixel_arcsec/np.sqrt(8*np.log(2))
    area = 2*np.pi*(sigma*grid.pixel_arcsec*np.pi/(180*3600))**2
    return np.asarray([gaussian_filter(s, sigma, mode="constant")*area for s in sky_jy_sr])
