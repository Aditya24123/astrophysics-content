# Gaussian pixel integration repair, 2026-10-21

The independent saved probe identified a consequential midpoint-integration
bias in valid bright compact Gaussian sources. Image-relative L2 and a total
flux sum had hidden accumulated error over many measured Fourier samples.
Against independently converged GL12, midpoint4 gave joint thermal-noise
errors 0.363578 (base compact), 3.197747 (turnover compact) and 0.919743
(turnover extended). The declared 0.1 joint-noise budget is unchanged.
The original evidence remains in
`output/training_readiness_20261021/gaussian_quadrature_probe_01.json`.

## Explicit numerical remedy

`configs/compact_proxy_gl4_v3.json` is a new candidate configuration. It adds
`grid.spatial_rule="gauss_legendre"` and explicit profile-specific rules.
All original prior ranges, offsets, sample distribution, target definitions,
challenge scheme, frequencies/bandwidths, Fourier mask, PB, taper and thermal/
calibration settings are identical to `compact_proxy_v2.json`. The existing
CLI default and historical configurations are retained pending independent
verification; this file must be selected explicitly for the repaired candidate.
The midpoint v2 default has the diagnosed integration failure and must not
be used to claim a new qualified training release.

For a pixel centered at (E,N), let (x_i,w_i) be order-n Gauss–Legendre nodes
and weights on [-1,1]. The pixel **average intensity after transfer** is

    <I> = sum_ij (w_i w_j / 4) I(E + d x_i/2, N + d x_j/2).

Here d is the angular pixel width and I includes the nonlinear optical-depth
transfer, rotation, sky offset, temperature and any declared background.
The weights sum to one. The separate spectral quadrature remains unchanged.
`Grid.subpixels` means spatial samples per axis: n midpoint subdivisions for
the historical rule, or order-n nodes per axis for the Gaussian GL rule.
GL4 and midpoint4 both use 16 spatial samples per pixel. NumPy's pinned
`numpy.polynomial.legendre.leggauss` supplies the nodes/weights.

The smooth Gaussian family uses GL4; the hard-boundary ellipsoid challenge
retains midpoint16. A Gaussian-only rule applied to an ellipsoid is rejected,
rather than silently falling back. `render_grid(config, profile)` resolves
the protected profile-specific controls consistently for generation and
configured validation. Sensitivity validation receives the same rule map.
Historical absent-rule configs still use the exact original midpoint
summation; regression tests compare values bit for bit.

Integration controls describe numerical source rendering, not observable
instrument information. They are recorded in protected configuration,
source snapshots and each protected label's `render_spatial_rule` and
`render_subpixels`. They do not enter public observing JSON, FITS headers,
measurement arrays or NPE features. Source parameters and original size
definitions are unchanged. A changed rule/configuration requires a new
output directory under the existing strict resume guard.

## Verification and evidence boundaries

`tests/test_spatial_integration.py` reproduces the independent probe's four
cases and adds six compact pixel-phase/orientation cases at the same prior
and turnover extremes. It records actual joint thermal-noise, image and flux
errors in JUnit properties; the acceptance tolerance is separate. It retains
one Fourier conjugate per pair, available-channel covariance and self-mode
variance in its independent rank-one covariance calculation. GL8/GL12
refinement checks reference convergence. Fixed tests also cover:

- adaptive two-dimensional integration of nonlinear transfer through an
  individual rotated, offset pixel with nonzero background;
- independent radial flux integration in thin, base-bright and turnover cases;
- bitwise historical midpoint values for both profiles;
- exactly unchanged prior plans, weights/PB, thermal sigma and fixed-signal
  noisy outputs/corruption draws across v2/v3 grids;
- protected provenance, public-only measurement loading, original target
  identity and rejection of incompatible resumes.

Measured results for the original base-compact and turnover-compact cases
are joint thermal-noise errors 2.21342e-7 and 2.04766e-5 with GL4, reproducing
the independent probe. Across all ten fixed cases the maximum GL4 error is
5.68124e-4; the maximum GL8/GL12 reference difference is 2.02858e-10.
The adaptive pixel-transfer relative error is 1.22e-15; transferring the
mean optical depth instead would bias that pixel by 4.26e-4 relative.
GL4's largest independent radial-flux error in the three normalization tests
is 8.56709e-8 relative; GL8's is 4.44e-15. The initial test required GL4 flux
precision of 1e-8 and failed on the compact-turnover case. That report is
retained. The corrected test explicitly uses GL8 to anchor the independent
high-precision flux reference and retains the measured GL4 errors; GL4's
release accuracy criterion remains the original joint-noise budget. No
production precision, noise or acceptance tolerance was changed.

These are bounded source-integration regressions. The old 32-case convergence
suite continues to describe midpoint refinement; it must not be relabeled GL
validation. A fresh candidate/independent numerical audit and the separately
coordinated resource benchmark are still required. No new candidate, bulk
generation, CASA run or estimator is part of this repair.

The separately supplied precision-physics evidence is now available: primary
van Hoof et al. (2014) Maxwellian nonrelativistic pure-H Gaunt tables were
compared at 10,295 points over 5000–12000 K and 0.9–8.1 GHz. Saved Draine
differences are 0.0732–0.1379%, with interpolation-order sensitivity 1.89e-7
relative; the independent coefficient is 0.0546796904354 versus the renderer's
rounded 0.05468. The table, license, primary URLs and hashes are retained in
`output/training_readiness_20261021/reference_vanhoof2014/comparison_02.json`.
This closes the previously missing independent precision reference within
that sampled thermal domain. It does not establish composition/geometry or
real-array accuracy, and these source equations were not changed here.

The existing 2048-draw `observe()` audit passed all 38 predeclared thermal and
latent-calibration statistics; nine fresh CASA draws passed supplemental
product gates. Their reports retain their original source identities and
limited scopes. Instrument/noise implementation and numerical settings are
unchanged by this integration repair; fixed-input regressions check that
invariance. These results are reused as supplied evidence, not rerun or
rebranded as a current-source whole-pipeline certification.
