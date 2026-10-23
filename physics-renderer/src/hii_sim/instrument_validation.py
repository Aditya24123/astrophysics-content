"""Bounded independent DFT checks of the fast proxy's bandwidth approximation."""
from dataclasses import replace, asdict
import numpy as np
from .instrument import operator, expected_measurement, primary_beam, noise_sigma_real, coverage_diagnostics, independent_fourier_cells
from .physics import render
from .schema import Source


def direct_pixel_visibilities(apparent_jy_sr, grid, uv_wavelengths):
    """Direct Fourier sum of pixel fluxes at arbitrary u_west,v_north.

    Independent of FFT implementation. Pixels are delta samples of their
    integrated flux, as in the fast model; not a continuous sky/antenna model.
    """
    east, north = grid.coordinates()
    west = np.deg2rad(-east.ravel()/3600)
    north = np.deg2rad(north.ravel()/3600)
    uv = np.asarray(uv_wavelengths)
    phase = np.exp(-2j*np.pi*(uv[:, 0, None]*west+uv[:, 1, None]*north))
    return phase@(np.asarray(apparent_jy_sr).ravel()*grid.omega_sr)


def bandwidth_reference(source, grid, sky, inst, channel, uv_center, quadrature=7):
    """Fixed projected physical baselines; PB, spectrum and uv vary within band."""
    center = inst.frequencies_ghz[channel]
    width = inst.bandwidths_mhz[channel]
    nodes, coefficients = np.polynomial.legendre.leggauss(quadrature)
    result = np.zeros(len(uv_center), dtype=complex)
    for node, coefficient in zip(nodes, coefficients):
        frequency = center+node*width/2000
        plane = render(source, grid, sky, [frequency], [0.], 1)[0]
        apparent = plane*primary_beam(grid, frequency, inst.pb_fwhm_arcmin_ghz)
        result += coefficient/2*direct_pixel_visibilities(apparent, grid,
                                         np.asarray(uv_center)*frequency/center)
    return result


def validate_bandwidth(grid, inst, sky):
    """Compare selected Fourier cells, including taper-active long baselines.

    Report sampled-cell error; it is not an all-baseline error bound.
    """
    weights, _ = operator(grid, inst)
    unique, _ = independent_fourier_cells(grid.npix)
    uv = np.fft.fftfreq(grid.npix, np.deg2rad(grid.pixel_arcsec/3600))
    cases = {"compact_offset": (Source(5e5, 8000, 12, .5, 37), replace(sky, east_arcsec=10, north_arcsec=-10)),
             "resolved": (Source(1.2e5, 8000, 60, .7, 70), sky),
             "turnover": (Source(1e7, 8000, 16, .7, 20), sky),
             "extended_offset_challenge": (Source(5e5, 8000, 90, .6, 20), replace(sky, east_arcsec=50))}
    results = []
    for channel in sorted({0, len(inst.frequencies_ghz)-1}):
        candidates = np.argwhere(unique & (weights[channel] > weights[channel].max()*1e-3))
        candidates = candidates[np.argsort(np.hypot(uv[candidates[:, 0]], uv[candidates[:, 1]]))]
        chosen = candidates[np.unique(np.linspace(0, len(candidates)-1, min(24, len(candidates))).astype(int))]
        indices = tuple(chosen.T)
        uv_center = np.column_stack((uv[chosen[:, 1]], uv[chosen[:, 0]]))
        for name, (source, position) in cases.items():
            cube = render(source, grid, position, inst.frequencies_ghz, inst.bandwidths_mhz, 5)
            fast = expected_measurement(cube, grid, inst)[channel][indices]
            ref3 = bandwidth_reference(source, grid, position, inst, channel, uv_center, 3)
            ref7 = bandwidth_reference(source, grid, position, inst, channel, uv_center, 7)
            w = weights[channel][indices]
            results.append({"case": name, "channel": channel, "source": asdict(source),
                "sky": asdict(position), "sampled_cells": len(chosen),
                "weighted_relative_l2_error": float(np.linalg.norm(w*(fast-ref7))/np.linalg.norm(w*ref7)),
                "reference_3_vs_7_relative_l2": float(np.linalg.norm(ref3-ref7)/np.linalg.norm(ref7)),
                "max_complex_error_over_thermal_component_sigma": float(np.max(abs(fast-ref7))/noise_sigma_real(grid, inst)[channel])})
    return {"coverage": coverage_diagnostics(grid, inst), "bandwidth_cases": results,
            "scope": "selected Fourier cells; direct pixel sum, fixed projected baselines, chromatic Gaussian PB and source spectrum; no w-term/time averaging",
            "unsupported": ["antenna geometry and Earth rotation", "phase errors", "flagging", "gain solving/calibration processing", "time smearing", "polarization", "w-term"],
            "fast_bandwidth_policy": "source spectrum averaged, uv and PB frozen at channel center; accept only after scope-specific error checks"}
