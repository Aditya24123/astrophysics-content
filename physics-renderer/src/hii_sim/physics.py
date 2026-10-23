"""Pure-H isothermal free–free transfer; see docs/MODEL.md for equations."""
import numpy as np
from astropy import units as u
from astropy.constants import k_B, c
from .schema import Source, Grid, Sky

RJ_JY_SR = (2*k_B/c**2).value * 1e26


def gaunt_factor(te, frequency):
    t = u.Quantity(te).to_value(u.K)
    nu = u.Quantity(frequency).to_value(u.GHz)
    if np.any(np.asarray(t) <= 0) or np.any(np.asarray(nu) <= 0):
        raise ValueError("Positive temperature/frequency required")
    return np.logaddexp(5.960 - np.sqrt(3)/np.pi*np.log(nu*(t/1e4)**-1.5), 1.)


def optical_depth(em, te, frequency):
    measure = u.Quantity(em).to_value(u.pc/u.cm**6)
    if np.any(~np.isfinite(measure)) or np.any(measure < 0):
        raise ValueError("EM must be finite and nonnegative")
    t = u.Quantity(te).to_value(u.K)
    nu = u.Quantity(frequency).to_value(u.GHz)
    return 0.05468 * t**-1.5 * nu**-2 * measure * gaunt_factor(te, frequency)


def intensity(em, te, frequency, background=0*u.K, contrast=True):
    """Return specific intensity in Jy/sr; defaults to background-subtracted."""
    tau = optical_depth(em, te, frequency)
    t = u.Quantity(te).to_value(u.K)
    bg = u.Quantity(background).to_value(u.K)
    tb = (t-bg)*(-np.expm1(-tau))
    if not contrast:
        tb = tb+bg
    nu_hz = u.Quantity(frequency).to_value(u.Hz)
    return (RJ_JY_SR * nu_hz**2 * tb) * u.Jy/u.sr


def em_profile(source: Source, east, north):
    angle = np.deg2rad(source.pa_deg)
    a = north*np.cos(angle) + east*np.sin(angle)
    b = east*np.cos(angle) - north*np.sin(angle)
    r2 = (a/source.major_fwhm_arcsec)**2 + (b/(source.axis_ratio*source.major_fwhm_arcsec))**2
    if source.profile == "gaussian_em":
        return source.em0_pc_cm6*np.exp(-4*np.log(2)*r2)
    # For this challenge only, the size field is a full boundary diameter.
    return source.em0_pc_cm6*np.sqrt(np.maximum(0, 1-4*r2))


def render(source: Source, grid: Grid, sky: Sky, frequencies_ghz,
           bandwidths_mhz=None, quadrature=3):
    """Pixel/channel-averaged sky contrast [frequency,y,x] in Jy/sr.

    subpixels is samples per axis: equal-width midpoint subdivisions, or
    Gauss–Legendre order for smooth Gaussian EM. Transfer precedes integration.
    Midpoint remains the default, including its historical summation behavior.
    """
    e, n = grid.coordinates()
    if grid.spatial_rule=="gauss_legendre":
        if source.profile!="gaussian_em":
            raise ValueError("Gauss-Legendre spatial integration supports smooth Gaussian EM only")
        nodes, coefficients=np.polynomial.legendre.leggauss(grid.subpixels)
        offsets=nodes*grid.pixel_arcsec/2
        spatial_weights=np.outer(coefficients,coefficients).ravel()/4
    else:
        offsets = ((np.arange(grid.subpixels)+.5)/grid.subpixels-.5)*grid.pixel_arcsec
        spatial_weights=None
    em = np.stack([em_profile(source, e+dx-sky.east_arcsec, n+dy-sky.north_arcsec)
                   for dy in offsets for dx in offsets])
    frequencies_ghz = np.asarray(frequencies_ghz)
    bandwidths_mhz = np.zeros_like(frequencies_ghz) if bandwidths_mhz is None else np.asarray(bandwidths_mhz)
    if frequencies_ghz.shape != bandwidths_mhz.shape or quadrature < 1:
        raise ValueError("Invalid spectral grid")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    result = []
    for center, width in zip(frequencies_ghz, bandwidths_mhz):
        out = np.zeros((grid.npix, grid.npix))
        for node, weight in zip(nodes, weights):
            nu = (center+node*width/2000)*u.GHz
            samples=intensity(em*u.pc/u.cm**6, source.te_k*u.K, nu,
                              sky.background_rj_k*u.K).value
            averaged=(samples.mean(axis=0) if spatial_weights is None else
                      np.einsum("i,ijk->jk",spatial_weights,samples))
            out += weight/2 * averaged
        result.append(out)
    return np.asarray(result)
