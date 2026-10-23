# Revised fast-instrument scope

The legacy `configs/base.json` is preserved as the specification of existing
products. It is not a 35 m array simulation. For N pixels of angular width
d radians, Fourier spacing is 1/(N d) wavelengths. On its 240 arcsec field
this is 859.436693 wavelengths. With the actual saved mask the shortest
sampled baselines, from 1.06 to 1.95 GHz, are 243.069, 196.681, 178.925,
152.457, 141.567 and 132.130 m. A lower-cut setting of 35 m does not create
any missing Fourier cells. DC is absent. The scale 1/u_min is reported as
a geometric scale only, not a survey maximum recoverable scale.

`coverage_diagnostics` reports actual support, independent complex and real
degrees of freedom, field and spacing, and missing DC. New shared instrument
JSON includes this report and is checksummed. New configurations set a
`max_short_spacing_ratio=1.5` guard: a shortest sampled baseline exceeding
1.5 times the requested lower cut raises an error. The margin is an explicit
development-grid acceptance choice, not an antenna-array accuracy claim.

- `configs/compact_proxy_v2.json`: 96 x 96 at 2.5 arcsec, lower cut 250 m,
  actual shortest baselines 253–344 m. This deliberately selects compact,
  spatially filtered observations. It cannot provide extended-source total
  flux. Gaussian EM rendering uses four subpixels per axis; ellipsoid
  challenges use sixteen.
- `configs/widefield_proxy_v2.json`: 1024 x 1024 at 2.5 arcsec, lower cut
  35 m, actual shortest baselines 36.9–45.6 m. This samples short spacings
  much better, at substantially greater memory/disk/time cost. No bulk run
  has been started. It still lacks DC and real antenna tracks.

## Bandwidth check and revised scope

The fast simulator averages the source spectrum across a top-hat channel but
freezes the primary beam and Fourier coordinates at its center. An independent
direct pixel-sum reference now integrates the source spectrum, chromatic
Gaussian PB and changing u,v for a fixed projected physical baseline. The
reference is a numerical check of this approximation, not a genuine array
simulator. It uses pixel-integrated flux samples, the same small-field sky
convention, and 3 versus 7 quadrature nodes. It does not use source predictions
to reconstruct images.

Eight representative comparisons of the legacy 128 MHz windows give small
fractional weighted visibility errors, but the worst tested single-cell
complex discrepancy exceeds the thermal per-component sigma by about 8.5.
Thus small relative error alone does not validate those windows at high S/N.
The legacy products have not been regenerated or relabeled as corrected.

Both revised configurations use **one 8 MHz channel at each of the six
documented THOR window centers**, not the full THOR continuum bandwidth.
Their assumed RMS is 1.2 mJy/dirty beam: 0.3 mJy times sqrt(128/8), an
explicit radiometer scaling at fixed integration, not measured survey noise.
This restricts the scope to a numerically tested narrow-channel approximation.
The full 128 MHz windows would need resolved spectral channels and measurement
averaging, plus channel covariance and imaging tests. Merely increasing
source spectral quadrature does not fix frozen-baseline smearing. The final
audit reports measured narrow-channel errors; do not extrapolate the tested
cells to arbitrary offsets, brightness or baselines.

## Included and absent effects

Included: Gaussian PB; incomplete Hermitian Fourier lattice; taper; correlated
thermal noise; shared lognormal amplitude and shared spectral-slope errors.
Cross-frequency noise rho is an explicit proxy assumption. Its noise is the
FFT of real Gaussian fields, so conjugate samples are duplicates. Real
self-conjugate cells have twice the usual per-component variance.

Not implemented: antenna positions/time tracks, noncoplanar w-term, phase
errors, flagging, time smearing, gain solving, calibration processing,
direction-dependent gains, polarization, realistic bandpasses and RFI.
This path assumes phase calibration has already succeeded; it is unsuitable
for learning reconstruction under phase errors or calibration failures.
No visibility product claims to reproduce THOR or GLOSTAR. The separate
[CASA path](CASA.md) is required for an actual array experiment. At this
audit's original date CASA was unavailable locally. Subsequent bounded WSL
execution and its remaining uncertainty/transfer limits are recorded in
[CASA_TERMINAL_MILESTONE.md](CASA_TERMINAL_MILESTONE.md); they do not establish
equivalence between the fast proxy and that array setting.

The bandwidth and missing-spacing interpretations follow the measurement
equation and wideband discussion in official [CASA synthesis-imaging
documentation](https://casadocs.readthedocs.io/en/stable/notebooks/synthesis_imaging.html)
(stable web edition, consulted 2026-10-19; runner API is pinned separately to
6.7.4). Survey settings derive from the
[THOR paper v2](https://arxiv.org/abs/1808.05990) and corrected
[CDS ReadMe](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/A%2BA/619/A124?format=html&tex=true), not
source-specific catalogue labels. Numerical settings introduced above are
explicit development decisions, not additional survey measurements.

## Sensitivity diagnostics

Use eta = (ln EM0, ln Te, ln major size, q, PA in radians). Central finite
differences use delta eta=0.001 (backward q difference at q=1). Step halving
is measured, including the nonsmooth ellipsoid edge. These are dimensionless
local responses; no unit-dependent condition-number recoverability claim is
made. The displayed 0.01 changes mean 1% in logarithmic quantities, 0.01 in
q, and 0.01 radians in PA.

Whitening retains one representative of each Hermitian pair. At each Fourier
cell, use C_fg = sigma_f sigma_g [rho+(1-rho) delta_fg] on the available
channels, independently for real/imaginary parts; self modes are real only
with 2C variance. Calibration nuisance columns are the predicted measurement
times fractional amplitude sigma and log(f/geometric_mean_f) times slope
sigma. With thermally whitened source Jacobian J and nuisance matrix A, the
reported local information is J'J-J'A(I+A'A)^(-1)A'J. This is a linearized
Gaussian nuisance approximation, not a posterior computation or MCMC fit.

Six regimes are tested separately: faint compact, thin resolved, bright
resolved, turnover compact, circular and ellipsoid. Circular sources have no
PA information. Thin-source EM/Te tradeoffs are local, not universal; use
turnover-spanning measurements or an explicitly external Te constraint when
appropriate. Resolved shapes require adequate beam sampling. These checks
do not certify final target count, global uniqueness or ML sample sufficiency.
