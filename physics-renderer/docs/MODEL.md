# Phase-one model specification (v1.0, 2026-10-19)

The physics equations below remain the source model. Instrument and numerical
settings describe legacy v1 unless stated otherwise. See
[INSTRUMENT_AUDIT.md](INSTRUMENT_AUDIT.md) for measured support, guarded v2
configurations, bandwidth restrictions and corrected sensitivity methods.
Legacy production uses two subpixels; revised Gaussian configurations use
four and ellipsoid challenges sixteen. No existing truth images were changed.
The new explicit Gaussian GL4 candidate uses the same 16 samples per pixel
with higher-order weights; see [GAUSSIAN_SPATIAL_INTEGRATION_V3.md](GAUSSIAN_SPATIAL_INTEGRATION_V3.md).
Historical midpoint configurations and their evidence are retained.

H II is ionized hydrogen. The code calculates numerical radiation and
measurements only. There is no inverse model, neural training, Bayesian
fitting, real-source inference, or M43 claim.

## Geometry and transfer

The initial family is an isothermal, pure-H, Maxwellian, unpolarized thermal
free–free emitter, in LTE for continuum transfer. Define tangent-plane east and
north offsets E,N from a separately supplied sky center. PA is measured east of
north, modulo 180 degrees. Let a=N cos(PA)+E sin(PA),
b=E cos(PA)-N sin(PA). The **projected emission-measure** profile is

    EM(E,N) = EM0 exp[-4 ln(2) ((a/theta)^2 + (b/(q theta))^2)].

theta is the full major-axis FWHM of EM, not a radius, disk diameter, or
necessarily the intensity FWHM. q is minor/major, 0<q<=1. EM0 is the peak
line-of-sight integral of n_e^2, in pc cm^-6. One possible realization is a
constant-depth slab whose density varies transversely as sqrt(EM/L). Neither L,
distance, nor density is determined or generated here. This projected family is
a deliberate simple hypothesis, not a homogeneous sphere or measured survey
profile. A uniform-density ellipsoid projection is reserved as a challenge:
EM=EM0 sqrt(1-r^2) inside r<1, where r is elliptical radius normalized by
semiaxes theta/2 and q theta/2. Thus theta is its full boundary diameter, a
different explicit size convention never mixed into the Gaussian training
family.

Adopt the Draine approximation as written in BeyondPlanck I (2023),
Eqs. 16–18 (natural logarithms):

    g = ln{exp[5.960 - sqrt(3)/pi ln(nu_GHz (Te/1e4 K)^(-3/2))] + e}
    tau = 0.05468 (Te/K)^(-3/2) nu_GHz^(-2) (EM/(pc cm^-6)) g
    Tb_RJ = Te (1-exp(-tau))
    I_nu = (2 k_B nu^2/c^2) Tb_RJ                 [W m^-2 Hz^-1 sr^-1]

Use expm1/logaddexp to avoid cancellation/overflow. This is the nonrelativistic
radio/Rayleigh–Jeans approximation, not an exact quantum Gaunt-factor solver.
For 5000–12000 K and 1–8 GHz, h nu/(k Te)<8e-5. Include a separate uniform
behind-source RJ background via Tb=Te(1-exp(-tau))+Tbg exp(-tau), and subtract
the unobscured Tbg when producing contrast images; default Tbg=0 K. Negative
contrast is possible if a background hotter than Te is explicitly requested;
the intrinsic emission remains nonnegative. No helium correction, dust,
synchrotron source component, scattering, line emission, gradients, or clumps
are included. Do not apply a helium factor intended for H-alpha conversion.

For each pixel average **intensity**, after transfer, over subpixels. Average
over top-hat spectral channels using Gauss–Legendre quadrature. Averaging EM
before nonlinear transfer is incorrect. Small-field pixel area is
(pixel_arcsec*pi/(180*3600))^2 sr. S_nu=sum(I_nu*Omega_pix), with 1 Jy=1e-26
SI. RJ brightness-temperature conversions are checked against Astropy. For a
Gaussian restoring beam Omega_b=pi Bmaj Bmin/(4 ln 2); Jy/beam=I[Jy/sr]*Omega_b
only after the appropriate beam operation. A dirty PSF has sidelobes and no
general Gaussian beam-area flux conversion. Its image units mean Jy per
unit-peak dirty beam; full-map summation is not integrated source flux.

## Sampling specification

All limits are configurable, explicit synthetic design choices, not population
priors. Base: EM0 log-uniform 1e4–5e5 pc cm^-6; Te uniform 5000–12000 K;
theta log-uniform 12–60 arcsec (EM FWHM); q uniform 0.5–1; PA uniform
[0,180) degrees. A scrambled five-dimensional Sobol design maps these marginals
with no imposed correlations. The temperature range brackets the THOR 8000 K
assumption; its endpoints are provisional stress margins, not catalogue
percentiles. The EM scale is motivated by THOR section 5.3, with the corrected
catalogue required for future population work. Gaussian peak EM and catalogue
beam-based EM are not identical. The angular range selects compact/resolved
cutouts within a 240 arcsec field and THOR's extended-flux limitations; it is
not a conversion of WISE MIR radius into radio EM FWHM. q limits are provisional
morphological coverage, with q=0.3 in sensitivity/challenge tests.

Separate turnover challenge: EM0 log-uniform 5e5–1e7, deliberately extrapolating
the base sample to probe opacity and missing identifiability. The configured
12 challenge sources are presently four profile, four turnover and four
changed-instrument cases; this is only a smoke set, not a statistical
distribution-shift evaluation. Separate profile and instrument challenges
retain distinct source IDs/splits. No rejection on
observed S/N; faint objects remain. Exclude the subpixel/hypercompact regime,
very extended sources, axis ratios below 0.5, and temperatures outside the
documented radio test domain. There are no source-specific values. Sky offsets are independent
uniform +/-10 arcsec nuisances; absolute pointing is an arbitrary synthetic
ICRS position, not M43. Inference targets are an explicit configurable list;
all five parameters are generated regardless. PA is undefined for q=1 and
must be treated periodically or marginalized. None is promised recoverable.

Range evidence and limitations: the THOR continuum paper's H II fits used
Te=8000 K for sources without a measured temperature; 151 fitted EM values
were 1e4–1e5 pc cm^-6 and its maximum was 4.2e5 pc cm^-6. That motivates a
log-uniform 1e4–5e5 design range with its upper edge as a stress margin; the
corrected catalogue must be used for any future population analysis. The
5000–12000 K Te limits bracket 8000 K as provisional stress coverage, not
observed quantiles. The 12–60 arcsec angular-FWHM range is a cutout coverage
choice; THOR used an effective-radius <60 arcsec selection in one fitting
analysis, and that radius is not interchangeable with this model's EM FWHM.
The 12-arcsec lower bound is below the fast path's ~20-arcsec dirty-beam FWHM
and should be expected to be poorly resolved there. Axis ratio 0.5–1 and a
uniform PA distribution are explicit morphology/nuisance design choices, not
measured population distributions. A future population prior needs corrected
catalogue and image-level selection analysis. M43 is a provisional later
validation target only; no M43 data are used here.

## Current instrument scope (2026-10-21)

The CLI defaults to `configs/compact_proxy_v2.json`: 250–3400 m baseline
cuts, 8 MHz bands and 1.2 mJy nominal dirty-beam RMS. Actual supported
baselines are reported per band and checked by the short-spacing guard;
a configured cut alone is not achieved coverage. The 240-arcsec Fourier
field remains a compact-source development proxy. Its taper is an imaging
weight, not an inverse-noise variance or a corruption applied to the data.
See [INSTRUMENT_AUDIT.md](INSTRUMENT_AUDIT.md) for supported effects and
finite-bandwidth checks.

CASA has since executed bounded point/extended-source development cases in
the separate WSL environment using CASA 6.7.4.8. The saved product-contract
acceptance and shared reconstruction limits are documented in
[CASA_TERMINAL_MILESTONE.md](CASA_TERMINAL_MILESTONE.md) and
[CASA_RECONSTRUCTION_PATH.md](CASA_RECONSTRUCTION_PATH.md). This is finite
single-setting evidence, not certification of multi-band training-data
transfer, native weight calibration or real-source fidelity.

The following v1 description is retained as historical provenance. Its
35 m cut, 128 MHz windows, 0.3 mJy noise and software-execution status do
not describe the current default or current validation state.

## Historical v1 instrument separation and assumptions

The source physics, measurement synthesis and image reconstruction are
separate modules. The v1 fast path in `instrument.py` was a development
approximation. It samples a seeded Hermitian mask on a regular Fourier grid,
multiplies by a Gaussian uv taper, applies a Gaussian primary-beam envelope,
adds Fourier-transformed image-plane Gaussian noise with 0.25 cross-frequency
correlation, then applies a common log-normal amplitude scale (fractional
standard deviation 5%) and a shared spectral log-slope error (standard
deviation 0.01). The toy settings use min/max baselines 35/3400 m, a 50%
pre-baseline mask request, 25-arcsec taper, 0.3 mJy/dirty-beam nominal image
noise, and 42 arcmin GHz PB FWHM coefficient. These are explicit configurable
development assumptions: the frequency centers and 128 MHz windows come from
the corrected THOR ReadMe, and 0.3 mJy/beam is the low end of THOR's reported
position/frequency-dependent 0.3 to >1 mJy/beam residual noise. The remaining
toy geometry, taper and calibration/correlation values do not claim to be
THOR's baseline tracks or calibration residuals. In particular, the random
uv mask does not represent Earth-rotation antenna tracks. The exact VLA path
is separate and described in `docs/CASA.md`.

Fast dirty images are made only from measured synthetic Fourier samples, not
from the source parameters. The image is normalized to a unit-peak dirty PSF
and has dirty-beam units. The approximate PB is stored and is not divided out.
Shared calibration draws and noise realizations are protected from the
predictor. At the v1 specification date, the real-array CASA path had a PB,
incomplete uv coverage and CASA thermal-noise model but had not yet run here.

## Evidence and limitations

- Wang et al. 2018, arXiv:1808.05990v2, sections 2, 3, 5.3:
  https://arxiv.org/html/1808.05990v2 (accessed 2026-10-19). Survey frequency,
  resolution, filtering and calibration motivate the development settings;
  its homogeneous fits and assumed temperatures are not exact labels.
- Wang et al. 2020, A&A 641 C1, doi:10.1051/0004-6361/201833642e:
  https://eprints.whiterose.ac.uk/id/eprint/166811/1/aa33642e-18.pdf . The
  official CDS ReadMe records its 2020-06-30 corrected catalogue and fitting
  table, the six usable 128-MHz windows and two RFI-discarded SPWs. The
  corrected tables supersede the on-line 2018 tables. No catalogue rows were
  imported as truth: https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/A%2BA/619/A124?format=html&tex=true .
- BeyondPlanck Collaboration 2023, A&A 675 A1, Eqs. 16–18:
  https://www.aanda.org/articles/aa/full_html/2023/07/aa44953-22/aa44953-22.html .
  This work uses the full nonlinear transfer, not that paper's later thin-only
  linearization. EM is pure-H and Te is in kelvin.
- Independent radio-opacity approximation and transfer: NRAO Essential Radio
  Astronomy, chapter 4, Eqs. 4.57–4.61:
  https://www.cv.nrao.edu/~sransom/web/Ch4.html . Validation compares the
  Draine opacity with the independent Mezger–Henderson approximation in
  Eq. 4.60. The alternate Gaunt expression in Eq. 4.59 uses a different
  convention and is not combined with Draine's g.
- GLOSTAR official project documentation, accessed 2026-10-19:
  https://glostar.mpifr-bonn.mpg.de/ . It describes 4–8 GHz VLA B/D observations
  and Effelsberg large-scale data. This motivates optional C-band challenges;
  base run uses THOR L-band windows only and is not a GLOSTAR reproduction.
- CASA 6.7.4 `simobserve` and `tclean` documentation:
  https://casadocs.readthedocs.io/en/v6.7.4/api/tt/casatasks.simulation.simobserve.html
  and https://casadocs.readthedocs.io/en/v6.7.4/api/tt/casatasks.imaging.tclean.html .
  These define the separate higher-fidelity integration path.
- Astropy 8.0.2 units/FITS/WCS documentation:
  https://docs.astropy.org/en/stable/units/equivalencies.html
  https://docs.astropy.org/en/stable/io/fits/
  https://docs.astropy.org/en/stable/wcs/

Independent validation combines the NRAO Mezger–Henderson opacity
approximation, Astropy brightness-temperature equivalency, independent radial
quadrature and transfer ODE integration. These check implementation and
approximation consistency, not agreement with real H II regions or precision
quantum calculations. The EM/Te continuum degeneracy remains severe in the
thin regime; see `docs/VALIDATION.md`.
