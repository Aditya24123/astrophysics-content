"""Explicit units in field names; public and protected contracts are distinct."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np


def positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Source:
    em0_pc_cm6: float
    te_k: float
    major_fwhm_arcsec: float
    axis_ratio: float
    pa_deg: float
    profile: str = "gaussian_em"

    def __post_init__(self):
        for key in ("em0_pc_cm6", "te_k", "major_fwhm_arcsec", "axis_ratio"):
            positive(getattr(self, key), key)
        if self.axis_ratio > 1 or not np.isfinite(self.pa_deg):
            raise ValueError("Require 0 < axis_ratio <= 1 and finite PA")
        if self.profile not in ("gaussian_em", "ellipsoid_em"):
            raise ValueError("Unknown projected EM profile")
        if not 5000 <= self.te_k <= 12000:
            raise ValueError("Temperature outside validated 5000–12000 K domain")


@dataclass(frozen=True)
class Grid:
    npix: int = 96
    pixel_arcsec: float = 2.5
    subpixels: int = 2
    spatial_rule: str = "midpoint"

    def __post_init__(self):
        if (any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer))
                for v in (self.npix,self.subpixels)) or
                self.npix < 16 or self.npix % 2 or self.subpixels < 1):
            raise ValueError("Even npix >=16 and subpixels >=1 required")
        if self.spatial_rule not in ("midpoint","gauss_legendre"):
            raise ValueError("Unknown spatial integration rule")
        positive(self.pixel_arcsec, "pixel_arcsec")

    @property
    def omega_sr(self):
        return np.deg2rad(self.pixel_arcsec / 3600)**2

    def coordinates(self):
        # FITS x increases west for negative CDELT1; north increases with y.
        v = (np.arange(self.npix) - self.npix//2) * self.pixel_arcsec
        north, west = np.meshgrid(v, v, indexing="ij")
        return -west, north


@dataclass(frozen=True)
class Sky:
    ra_deg: float = 180.0
    dec_deg: float = 30.0
    east_arcsec: float = 0.0
    north_arcsec: float = 0.0
    background_rj_k: float = 0.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Sky values must be finite")
        if not 0 <= self.ra_deg < 360 or not -90 < self.dec_deg < 90:
            raise ValueError("Invalid pointing")
        if self.background_rj_k < 0:
            raise ValueError("Negative background")


@dataclass(frozen=True)
class Instrument:
    frequencies_ghz: tuple = (1.06, 1.31, 1.44, 1.69, 1.82, 1.95)
    bandwidths_mhz: tuple = (128.,)*6
    frequency_quadrature: int = 3
    taper_fwhm_arcsec: float = 25.0
    min_baseline_m: float = 35.0
    max_short_spacing_ratio: float | None = None
    max_baseline_m: float = 3400.0
    sampling_fraction: float = 0.5
    sampling_seed: int = 9227
    rms_jy_dirty_beam: float = 0.0003
    noise_frequency_correlation: float = 0.25
    calibration_fraction: float = 0.05
    calibration_slope_std: float = 0.01
    pb_fwhm_arcmin_ghz: float = 42.0
    name: str = "development_uv_proxy_v1"

    def __post_init__(self):
        if len(self.frequencies_ghz) != len(self.bandwidths_mhz):
            raise ValueError("Frequency/bandwidth length mismatch")
        for f, b in zip(self.frequencies_ghz, self.bandwidths_mhz):
            positive(f, "frequency"); positive(b, "bandwidth")
            if f - b/2000 < 0.9 or f+b/2000 > 8.1:
                raise ValueError("Frequency outside validated 0.9–8.1 GHz domain")
        for key in ("taper_fwhm_arcsec", "max_baseline_m", "pb_fwhm_arcmin_ghz"):
            positive(getattr(self, key), key)
        if self.min_baseline_m < 0 or self.max_baseline_m <= self.min_baseline_m:
            raise ValueError("Invalid baseline interval")
        if self.max_short_spacing_ratio is not None and (
                not np.isfinite(self.max_short_spacing_ratio) or self.max_short_spacing_ratio < 1 or self.min_baseline_m == 0):
            raise ValueError("Short-spacing ratio guard requires ratio >=1 and positive minimum baseline")
        if not 0 < self.sampling_fraction <= 1 or not 0 <= self.noise_frequency_correlation < 1:
            raise ValueError("Invalid sampling fraction/correlation")
        for key in ("rms_jy_dirty_beam", "calibration_fraction", "calibration_slope_std"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"Invalid {key}")
        if self.frequency_quadrature < 1:
            raise ValueError("Invalid quadrature order")


def load_config(path):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    Grid(**c["grid"]); Instrument(**c["instrument"])
    for profile, count in c.get("profile_subpixels", {}).items():
        if profile not in ("gaussian_em", "ellipsoid_em") or isinstance(count,bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Invalid profile-specific subpixel setting")
    for profile, rule in c.get("profile_spatial_rules", {}).items():
        if (profile not in ("gaussian_em", "ellipsoid_em") or
                rule not in ("midpoint","gauss_legendre") or
                (profile=="ellipsoid_em" and rule!="midpoint")):
            raise ValueError("Invalid profile-specific spatial rule; hard boundaries require midpoint")
    profiles={c["profile"]}
    if c["challenge_sources"]>0:
        profiles.add("ellipsoid_em")
    for profile in profiles:
        render_grid(c,profile)  # Reject a Gaussian-only rule on hard-boundary sources before writing.
    ranges = c["sampling"]["ranges"]
    expected = ("em0_pc_cm6", "te_k", "major_fwhm_arcsec", "axis_ratio", "pa_deg")
    if set(ranges) != set(expected):
        raise ValueError("Exactly five generative parameter ranges required")
    for name, (lo, hi, scale) in ranges.items():
        if not np.isfinite(lo+hi) or hi <= lo or scale not in ("linear", "log"):
            raise ValueError(f"Invalid range {name}")
        if scale == "log" and lo <= 0:
            raise ValueError("Log range must be positive")
    if not set(c["reported_targets"]).issubset(expected):
        raise ValueError("Unknown reported target")
    if len(set(c["reported_targets"])) != len(c["reported_targets"]):
        raise ValueError("Duplicate target")
    fractions = c["sampling"]["split_fractions"]
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("Invalid train/validation/test fractions")
    return c


def render_grid(config, profile):
    """Protected profile-specific numerical controls, never observing features."""
    values=dict(config["grid"])
    values["subpixels"]=config.get("profile_subpixels",{}).get(profile,values.get("subpixels",2))
    values["spatial_rule"]=config.get("profile_spatial_rules",{}).get(profile,values.get("spatial_rule","midpoint"))
    grid=Grid(**values)
    if profile not in ("gaussian_em","ellipsoid_em"):
        raise ValueError("Unknown rendering profile")
    if profile=="ellipsoid_em" and grid.spatial_rule!="midpoint":
        raise ValueError("Hard-boundary ellipsoid requires explicit midpoint integration")
    return grid
