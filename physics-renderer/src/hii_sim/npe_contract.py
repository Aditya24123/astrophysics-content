"""Measurement/uncertainty types for the declared Fourier proxy; no ML code."""
from dataclasses import dataclass
import numpy as np

from .instrument import independent_fourier_cells


def immutable_array(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CanonicalFourierData:
    """Band-major canonical cells; real then imaginary, self cells real only."""
    values_jy: np.ndarray
    frequency_index: np.ndarray
    y_index: np.ndarray
    x_index: np.ndarray
    component: np.ndarray  # 0=real, 1=imaginary; never an imaginary self mode.
    self_conjugate: np.ndarray
    unit: str = "Jy"


@dataclass(frozen=True)
class ThermalCovariance:
    """Conditional thermal law, not total calibration+thermal uncertainty."""
    sigma_real_jy: np.ndarray
    frequency_correlation: float
    provenance: str = "known_synthetic_assumption_not_real_telescope_estimate"
    family: str = "Gaussian_per_canonical_Fourier_cell"
    covariance_unit: str = "Jy^2"

    def __post_init__(self):
        sigma=np.asarray(self.sigma_real_jy)
        if (sigma.ndim!=1 or not len(sigma) or not np.isfinite(sigma).all() or np.any(sigma<=0)
                or not np.isfinite(self.frequency_correlation) or not 0<=self.frequency_correlation<1):
            raise ValueError("Invalid conditional thermal covariance parameters")
        object.__setattr__(self,"sigma_real_jy",immutable_array(sigma,dtype=float))

    def covariance(self, channels=None, *, self_conjugate=False):
        channels = np.arange(len(self.sigma_real_jy)) if channels is None else np.asarray(channels)
        if (channels.ndim != 1 or not np.issubdtype(channels.dtype, np.integer)
                or np.any(channels < 0) or np.any(channels >= len(self.sigma_real_jy))
                or len(np.unique(channels)) != len(channels)):
            raise ValueError("Covariance channels must be distinct valid integer indices")
        sigma = self.sigma_real_jy[channels]
        correlation = ((1-self.frequency_correlation)*np.eye(len(channels))
                       + self.frequency_correlation)
        return np.outer(sigma, sigma)*correlation*(2. if self_conjugate else 1.)


@dataclass(frozen=True)
class CalibrationPrior:
    """Global real gains: common lognormal amplitude and Gaussian log-slope.

    gain_moments returns dimensionless G moments. Visibility covariance is
    source dependent: F_f(k)*conj(F_g(l))*Cov(G_f,G_g); its pseudo-covariance
    uses F_f(k)*F_g(l). This is not independent additive pixel noise.
    """
    frequencies_ghz: np.ndarray
    fractional_common_amplitude_std: float
    spectral_log_slope_std: float
    provenance: str = "known_synthetic_prior_not_true_corruption_draw"
    family: str = "shared_lognormal_amplitude_and_Gaussian_log_slope"

    def __post_init__(self):
        f=np.asarray(self.frequencies_ghz)
        if (f.ndim!=1 or not len(f) or not np.isfinite(f).all() or np.any(f<=0)
                or any(not np.isfinite(v) or v<0 for v in
                       (self.fractional_common_amplitude_std,self.spectral_log_slope_std))):
            raise ValueError("Invalid shared calibration-prior parameters")
        object.__setattr__(self,"frequencies_ghz",immutable_array(f,dtype=float))

    def gain_moments(self):
        logf = np.log(self.frequencies_ghz)
        logf -= logf.mean()
        s2 = np.log1p(self.fractional_common_amplitude_std**2)
        b2 = self.spectral_log_slope_std**2
        mean = np.exp(.5*b2*logf**2)
        covariance = np.outer(mean, mean)*np.expm1(s2+b2*np.outer(logf, logf))
        return mean, covariance


@dataclass(frozen=True)
class ProxyObservingMetadata:
    npix: int
    pixel_arcsec: float
    frequencies_ghz: np.ndarray
    bandwidths_mhz: np.ndarray
    support: np.ndarray
    imaging_taper: np.ndarray
    primary_beam: np.ndarray
    thermal: ThermalCovariance
    calibration: CalibrationPrior
    operator_family: str = "artificial_Hermitian_Fourier_lattice_not_antenna_tracks"
    uv_convention: str = "unshifted_fftfreq_cycles_per_radian_x_west_y_north"
    taper_role: str = "imaging_weight_not_inverse_variance"


def canonical_fourier_data(packed_vis, weights):
    """Validate and unpack storage support, then retain independent real data."""
    weights, packed_vis = np.asarray(weights), np.asarray(packed_vis)
    if (weights.ndim != 3 or weights.shape[1] != weights.shape[2]
            or not np.isfinite(weights).all() or np.iscomplexobj(weights) or np.any(weights < 0)):
        raise ValueError("Invalid Fourier weight grid")
    n = weights.shape[1]
    support = weights > 0
    if not np.all(support.any(axis=(1, 2))):
        raise ValueError("Every Fourier band needs supported measurements")
    ids = (-np.arange(n)) % n
    if not np.array_equal(support, support[:, ids[:, None], ids[None, :]]):
        raise ValueError("Fourier support must be Hermitian")
    if (packed_vis.ndim != 1 or len(packed_vis) != int(support.sum())
            or not np.iscomplexobj(packed_vis) or not np.isfinite(packed_vis).all()):
        raise ValueError("Packed visibility shape/dtype/finite-value mismatch")
    full = np.zeros(weights.shape, dtype=packed_vis.dtype)
    full[support] = packed_vis
    partner = full[:, ids[:, None], ids[None, :]]
    eps = np.finfo(packed_vis.real.dtype).eps
    if not np.allclose(full, partner.conj(), rtol=8*eps,
                       atol=8*eps*max(float(np.max(np.abs(packed_vis))), np.finfo(packed_vis.real.dtype).tiny)):
        raise ValueError("Visibilities violate the declared real-sky Hermitian contract")
    unique, self_cells = independent_fourier_cells(n)
    coordinates = np.argwhere(support & unique[None])
    f, y, x = coordinates.T
    self_mode = self_cells[y, x]
    values = full[f, y, x]
    counts = np.where(self_mode, 1, 2)
    first = np.r_[0, np.cumsum(counts)[:-1]]
    vector = np.empty(int(counts.sum()), dtype=packed_vis.real.dtype)
    components = np.zeros(len(vector), dtype=np.uint8)
    vector[first] = values.real
    vector[first[~self_mode]+1] = values[~self_mode].imag
    components[first[~self_mode]+1] = 1
    return CanonicalFourierData(
        immutable_array(vector), immutable_array(np.repeat(f, counts)),
        immutable_array(np.repeat(y, counts)), immutable_array(np.repeat(x, counts)),
        immutable_array(components), immutable_array(np.repeat(self_mode, counts)))


def source_parameter_definitions(profile):
    """Protected label semantics; never append this result to NPE features."""
    sizes = {"gaussian_em": "full_major_axis_projected_EM_FWHM",
             "ellipsoid_em": "full_major_axis_projected_EM_boundary_diameter"}
    if profile not in sizes:
        raise ValueError("Unknown original generating profile")
    return {"profile": profile, "original_values": True,
            "size_parameter": "major_fwhm_arcsec", "size_definition": sizes[profile],
            "units": {"em0_pc_cm6": "pc cm-6", "te_k": "K", "major_fwhm_arcsec": "arcsec",
                      "axis_ratio": "1", "pa_deg": "deg"},
            "pa_definition": "north_through_east_modulo_180; uninformative_at_q_equal_1"}
