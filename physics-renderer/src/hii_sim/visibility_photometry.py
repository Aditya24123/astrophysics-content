"""Fixed phase-center correlated flux, not resolved total-source photometry.

Frozen photometry_protocol_v1 formulas: single channel, unpolarized RR/LL,
equal independent Gaussian receiver components and independent raw rows.
Calibration must come from separate measured observations, never source truth.
No flag handling, native unequal weights, gain/phase calibration or fitting.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta, norm

OPERATORS=("natural", "Gaussian120arcsec")


def _readonly(value):
    result=np.array(value,copy=True)
    result.setflags(write=False)
    return result


def _positive_scalar(value,name):
    a=np.asarray(value)
    if a.ndim or a.dtype.kind not in "fiu" or isinstance(value,(bool,np.bool_)):
        raise ValueError(f"{name} must be a positive finite real scalar")
    result=float(a)
    if not np.isfinite(result) or result<=0:
        raise ValueError(f"{name} must be a positive finite real scalar")
    return result


def _parallel_hands(rr_ll,*,correlations,unpolarized,flags,row_flags,native_weights):
    v=np.asarray(rr_ll)
    if (v.ndim!=2 or v.shape[0]!=2 or v.shape[1]==0 or
            not np.iscomplexobj(v) or not np.isfinite(v).all()):
        raise ValueError("Single-channel finite complex RR/LL array [2, row] required")
    if tuple(correlations)!=("RR","LL") or unpolarized is not True:
        raise ValueError("Only declared unpolarized RR/LL scope is supported")
    for field,shape in ((flags,v.shape),(row_flags,(v.shape[1],))):
        if field is not None:
            a=np.asarray(field)
            if a.dtype.kind!="b" or a.shape!=shape or a.any():
                raise ValueError("Flagged samples or malformed flag arrays are unsupported")
    if native_weights is not None:
        w=np.asarray(native_weights)
        if (w.shape!=v.shape or w.dtype.kind not in "fiu" or not np.isfinite(w).all()
                or np.any(w<=0) or not np.all(w==w.flat[0])):
            raise ValueError("Unequal hand/row native weighting is unsupported")
    return v


@dataclass(frozen=True)
class ThermalCalibration:
    sigma_component_jy: float
    sigma_99_9percent_interval_jy: tuple
    complex_RR_minus_LL_differences: int
    calibration_method: str = "independent measured RR-LL /sqrt2, lower-median complex power; unpolarized/equal independent hands assumed"
    physical_inverse_variance_from_native_WEIGHT: bool = False

    def __post_init__(self):
        sigma=_positive_scalar(self.sigma_component_jy,"component sigma")
        interval=tuple(self.sigma_99_9percent_interval_jy)
        if (len(interval)!=2 or any(not np.isfinite(v) or v<=0 for v in interval)
                or not interval[0]<=sigma<=interval[1]
                or isinstance(self.complex_RR_minus_LL_differences,bool)
                or not isinstance(self.complex_RR_minus_LL_differences,(int,np.integer))
                or self.complex_RR_minus_LL_differences<1
                or self.physical_inverse_variance_from_native_WEIGHT is not False):
            raise ValueError("Invalid independent thermal calibration record")
        object.__setattr__(self,"sigma_component_jy",sigma)
        object.__setattr__(self,"sigma_99_9percent_interval_jy",interval)


def measured_thermal_calibration(rr_ll,*,correlations=("RR","LL"),unpolarized=True,
                                 flags=None,row_flags=None,native_weights=None):
    """Lower-median RR-LL scale and 99.9% order-statistic interval.

    Caller supplies independent measured calibration observations. Independence,
    equal receiver noise and absence of polarization cannot be proven from one
    array. A polarized/calibration residual can contaminate RR-LL differences.
    """
    v=_parallel_hands(rr_ll,correlations=correlations,unpolarized=unpolarized,
                      flags=flags,row_flags=row_flags,native_weights=native_weights)
    z=(v[0]-v[1])/np.sqrt(2);power=np.abs(z)**2;n=power.size;k=(n+1)//2
    lower_median=float(np.partition(power,k-1)[k-1])
    scale=float(np.sqrt(lower_median/(2*np.log(2))))
    alpha=.001;ulo=beta.ppf(alpha/2,k,n+1-k);uhi=beta.ppf(1-alpha/2,k,n+1-k)
    lo=float(np.sqrt(lower_median/(2*(-np.log1p(-uhi)))))
    hi=float(np.sqrt(lower_median/(2*(-np.log1p(-ulo)))))
    return ThermalCalibration(scale,(lo,hi),int(n))


def observing_weights(uvw_m,frequency_hz):
    """Fixed natural and Gaussian120arcsec weights from observed UVW/frequency."""
    uv=np.asarray(uvw_m);frequency=_positive_scalar(frequency_hz,"frequency_hz")
    if (uv.ndim!=2 or uv.shape[0]!=3 or uv.shape[1]==0 or
            uv.dtype.kind not in "fiu" or not np.isfinite(uv).all()):
        raise ValueError("Finite observed UVW in meters, shape [3, row], required")
    radius=np.hypot(uv[0],uv[1])*frequency/299792458.
    sigma=np.deg2rad(120/3600)/np.sqrt(8*np.log(2))
    return np.stack([np.ones(len(radius)),np.exp(-2*np.pi**2*sigma*sigma*radius*radius)])


def prediction_critical():
    """Marginal 95% union bound: .001 scale failure plus .049 Gaussian tails."""
    return float(norm.ppf(1-(.05-.001)/2))


@dataclass(frozen=True)
class CorrelatedFlux:
    values_jy: np.ndarray
    covariance_jy2: np.ndarray
    marginal_conservative_95_intervals_jy: np.ndarray
    operators: tuple = OPERATORS
    estimand: str = "point_normalized_phase_center_real_Stokes_I_correlated_flux"
    uncertainty: str = "conditional_equal_independent_Gaussian_receiver_noise_only"
    resolved_total_flux: bool = False


def correlated_flux(rr_ll,uvw_m,frequency_hz,calibration,*,correlations=("RR","LL"),
                    unpolarized=True,flags=None,row_flags=None,native_weights=None):
    """Return two linear correlated-flux statistics and their full 2x2 covariance.

    No source size, true centroid/gain, noiseless template, aperture or image is
    accepted. Only a point at phase center has the total-flux interpretation.
    Intervals use the independent calibration upper scale and are marginal,
    not a simultaneous two-operator region; systematic gain error is excluded.
    """
    if not isinstance(calibration,ThermalCalibration):
        raise ValueError("An independently measured ThermalCalibration record is required")
    v=_parallel_hands(rr_ll,correlations=correlations,unpolarized=unpolarized,
                      flags=flags,row_flags=row_flags,native_weights=native_weights)
    a=observing_weights(uvw_m,frequency_hz)
    if a.shape!=(2,v.shape[1]) or np.any(a.sum(axis=1)<=0):
        raise ValueError("UVW rows must match data and support both fixed operators")
    signal=v.mean(axis=0).real;denom=a.sum(axis=1);value=a@signal/denom
    sigma=calibration.sigma_component_jy
    covariance=(sigma*sigma/2)*(a@a.T)/np.outer(denom,denom)
    upper=calibration.sigma_99_9percent_interval_jy[1]
    halfwidth=prediction_critical()*np.sqrt(np.diag(covariance))*upper/sigma
    intervals=np.column_stack((value-halfwidth,value+halfwidth))
    return CorrelatedFlux(_readonly(value),_readonly(covariance),_readonly(intervals))
