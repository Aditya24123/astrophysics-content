# Phase-one training-input readiness specification v1

Frozen before any new experiment or release-candidate generation, 2026-10-21.
This turn is inspection, specification and existing-evidence assessment only.
It authorizes no CASA execution, source edits, model training, parameter fitting
or bulk run. Current executable digest:
`c917b382df7097e86af7caeb05b831ac7783dfe1851b66454690d5e5435f639c`.
Current candidate configuration is `configs/compact_proxy_v2.json`, SHA-256
`b7aec83f127d438950d3675742065b27762304051219980d21022de805d5eafb`.

## 1. Intended use and independent decisions

The intended later NPE input is **measurements plus permitted observing
metadata**. Default targets are the ORIGINAL generated EM0, Te, Gaussian EM
major-axis FWHM and axis ratio q. PA is generated without a preferred
orientation and is configurable as an additional periodic target or a
nuisance. Four configured targets are not four guaranteed recoverable
properties. Original targets are never replaced by fitted/CLEAN/catalogue
estimates. No ML, NPE, MCMC or conventional physical-parameter fit is part of
this qualification stage.

Numerical image reconstruction is a distinct measurement-only product. It
must not use hidden source parameters, intrinsic images or true corruption
draws as controls. The forward renderer can accept physical parameters;
that is its generative purpose. Auditors may use protected truth to measure
simulation error, never to tune per-source reconstruction controls. Future
image-learning targets may use explicitly declared generating sky images;
measurement-derived CLEAN components are not physical truth labels.

There are TWO distinct suitability questions:

1. **P: declared-proxy training-input suitability.** Does the full configured
   source prior produce accurate, statistically specified, leak-resistant and
   reproducible measurements under `compact_proxy_v2`? All configured weak,
   partially resolved, bright and extended cases remain; no S/N or successful
   reconstruction selection is allowed. This is a useful synthetic inverse
   problem, not a renamed VLA simulation.
2. **T: CASA/real-data transfer suitability.** Are the training measurement
   operator, frequency/array support, noise/calibration model, selection and
   uncertainty metadata justified for the intended real measurements? P does
   not imply T. The present single-frequency CASA development evidence does
   not establish this multi-band transfer contract. A real-data claim stays
   blocked until a matched array/measurement release and independent audits
   support it; this specification does not shrink that goal to declare success.

A release-candidate inspection set is neither a complete production bank nor
a statistical guarantee of later ML performance. Learning curves and
posterior calibration require subsequent training and independent evaluation.

## 2. Physics, prior and exact operator being qualified

Pure ionized hydrogen, isothermal electrons, no helium correction, dust,
synchrotron, clumps or temperature gradients. EM is a projected generating
quantity in pc cm^-6; no density, mass or distance is inferred from it.

The base projected profile is
`EM(a,b)=EM0*exp[-4 ln(2)*((a/theta)^2+(b/(q*theta))^2)]`.
Theta is the full major-axis EM FWHM, not radius, boundary diameter or
necessarily the optically thick intensity FWHM. PA is north through east;
q=1 makes PA uninformative. Transfer uses the documented Draine-type Gaunt
approximation, `tau=0.05468*T^-1.5*nu_GHz^-2*EM*gff`, and
`I=2*k*nu_Hz^2/c^2*(Te-Tbg)*(1-exp(-tau))` in contrast form, converted to
Jy/sr. The baseline has Tbg=0. Intensity, not EM, is subpixel-averaged after
nonlinear transfer. Units and approximation validity remain explicit.

The full base design is retained, without post-selection:

| Generating quantity | Distribution |
|---|---|
| EM0 | log-uniform 1e4–5e5 pc cm^-6 |
| Te | uniform 5000–12000 K |
| Gaussian EM major FWHM | log-uniform 12–60 arcsec |
| q | uniform 0.5–1 |
| PA | uniform [0,180) degrees |
| East/north offsets | separate uniform [-10,10] arcsec nuisances |

A scrambled five-dimensional Sobol design maps the five source marginals;
no population correlations are imposed. These are documented synthetic
design ranges, not an empirical Galactic population prior. The corrected
THOR catalogue can motivate ranges, not provide exact generating labels.
Weak Te/EM information is retained and reported, not removed. If the later
science requires sharper Te constraints, turnover-spanning observations or
an explicitly external physical constraint may be needed; more noise copies
cannot create missing information. An external Te constraint is never silently
renamed instrument metadata or set equal to the true Te.

The current 12 challenges cycle through four ellipsoid-profile, four turnover
(EM0 5e5–1e7) and four changed-instrument parents. They are a smoke set,
outside base train/validation/test. The ellipsoid's stored size scalar means
**full boundary diameter**, despite the present `major_fwhm_arcsec` key. It
must have an explicit protected size-definition tag and profile-aware target
view before release; it must not be pooled or scored as Gaussian FWHM.
Preserve its original scalar, rather than silently deriving a new label.
Out-of-prior challenges cannot be required to have in-prior NPE coverage.
Additional fixed boundary/numerical fixtures are validation cases, not evidence
that twelve examples establish distribution-shift robustness.

The exact P operator is a six-band, artificial Hermitian Fourier lattice on
96x96 pixels at 2.5 arcsec (240-arcsec periodic field). Intrinsic intensity is
averaged over 8-MHz channels at 1.06,1.31,1.44,1.69,1.82,1.95 GHz using
three-node quadrature, multiplied by the central-frequency Gaussian PB,
converted to Jy/pixel and FFT-sampled. Values are zero outside the supplied
support. Imaging weights are a 25-arcsec Gaussian uv taper; they are NOT
per-visibility inverse variances. The mask is synthetic, not antenna/time
tracks. The configured 250–3400 m radial cuts yield actual minimum sampled
baselines of approximately 343.75,278.15,253.04,304.91,283.13,264.26 m in
the six channels; DC is absent. The 1.5 short-spacing ratio guard is enforced.
The inverse-minimum-uv angle is not a calibrated maximum recoverable scale.

Central-frequency PB/uv operation after intrinsic channel averaging is part
of this declared proxy. It does not reproduce full chromatic gridding,
bandwidth/time smearing, w effects, calibration processing or a real VLA
bandpass. T requires an independently checked instrument path, not merely
similar frequencies or an image that looks plausible.

## 3. Training-facing contract and label separation

The required explicit **measurement-only view** is:

- Observations: sampled complex visibilities in Jy, their frequency/cell map
  and availability mask; or a documented equivalent real/imaginary vector.
  Packed storage uses NumPy C-order over `(frequency,y,x)` at `weights>0`.
- Canonical independent view: use one representative of each Hermitian pair
  (`flat_index <= partner_index`); real/imaginary parts for non-self modes,
  real part only for self modes. Retain packing/selection metadata. Do not
  count conjugates or deterministic dirty images as extra independent data.
- Permitted metadata: frequency/bandwidth, WCS/grid, supplied sampling support,
  uv coordinates, imaging-taper role, PB/PSF, thermal uncertainty description
  and declared calibration-error assumptions. Distinguish known synthetic
  assumptions, independently measured quantities and estimated uncertain
  quantities. True gains/slope, exact centroid, source properties, source
  profile/size labels and seeds are excluded from feature inputs.
- Bookkeeping: opaque source/realization IDs, split and lineage are separate
  from model tensors. Numerical dirty/CLEAN images are separate derived
  products, not part of the default measurement-only NPE view.

`PredictorDataset` currently returns a general observations dictionary that
contains both visibilities and a dirty image. It correctly separates protected
labels, but a typed/allowlisted measurement-only view and its regression test
are still required before treating this as the agreed training interface.
The future conventional estimator must consume the same observation/metadata
contract without true labels. Its implementation is not authorized here.

`PairedDataset` explicitly adds original generating parameter values and
generating sky images for supervised losses/evaluation. It must preserve
target order `[em0_pc_cm6, te_k, major_fwhm_arcsec, axis_ratio]`, units and
size definition; PA inclusion is explicit. Derived logarithms/normalization
are future training transforms fitted only on training sources, not changed
labels in storage. No fitted parameters or catalogue estimates replace truth.
Release loading/auditing uses `verify=True`, not the current unchecked default.

Only `public/` is distributable to predictors. No true/noiseless signal is a
predictor feature. Noise-free signals and corruption draws are protected
validation/provenance material. Current sigma metadata is the known/assumed
**thermal per-real-component** scale for non-self cells; it is not total
calibration-plus-thermal uncertainty and is not automatically a measurement
of a real telescope's noise.

## 4. Statistical specification and uncertainty precision

Let F_f(k) be the noise-free proxy visibility and
`V_f(k)=G_f F_f(k)+N_f(k)`. Thermal real/imaginary components for one
independent non-self Fourier cell have
`C_N,fg=sigma_f*sigma_g*((1-rho)*delta_fg+rho)`, restricted to channels that
measure that cell, with rho=0.25. Distinct canonical modes are independent
under this generator. Self modes are real only with twice this variance.
The mask and taper enter the documented image covariance through the linear
imaging operator, not an assumed count of independent pixels.

Current gain assumptions are `X~N(-s_g²/2,s_g²)`,
`s_g²=ln(1+0.05²)`, `beta~N(0,0.01²)` independently, and
`G_f=exp(X+beta*l_f)` where `l_f=ln(f/geometric_mean(f))`.
Therefore `E[G_f]=exp(0.5*0.01²*l_f²)` and
`Cov(G_f,G_g)=E[G_f]*E[G_g]*(exp(s_g²+0.01²*l_f*l_g)-1)`.
The slope gives a small nonzero mean factor away from the reference band;
do not silently assume every channel's mean gain is exactly one. Shared gain
creates source-dependent cross-cell covariance (and complex pseudo-covariance),
not just a diagonal variance added independently to each pixel. NPE can learn
this generative distribution; the uncertainty metadata and audit must still
describe it correctly. The existing linearized Fisher nuisance treatment is a
local design diagnostic, not validation of this nonlinear marginal law.

For the univariate thermal checks below, standardize each non-self component
by its declared sigma (self modes by sqrt(2)*sigma); the common-cell
cross-band correlation is then rho. Full Cholesky whitening instead gives
identity covariance and is used for the joint error budget. Do not confuse
these two normalizations.

The following precision requirements are frozen before new draws:

| Audit estimand | Required precision/consistency |
|---|---|
| Six thermal standardized means | model zero inside simultaneous CI; CI half-width <=0.05 sigma |
| Six thermal standard deviations | ratio to declared value has CI wholly within [0.95,1.05] |
| Fifteen cross-band thermal correlations | declared value inside CI; half-width <=0.05 |
| Six nominal 95% observation-noise interval coverages | CI wholly inside [0.93,0.97] |
| Mean and SD of latent log common gain | declared values inside CI; mean half-width <=0.005 log units; SD relative half-width <=10% |
| Mean and SD of latent spectral slope | declared values inside CI; mean half-width <=0.001; SD relative half-width <=10% |
| Latent log-gain/slope correlation | zero inside CI; half-width <=0.10 |

These are 38 primary statistics for the proxy, with family-wise 95% confidence
using Bonferroni per-statistic alpha=0.05/38 at a single frozen qualification
look. Use exact Gaussian/chi-square intervals where their independent-Gaussian
assumptions hold, Fisher-z intervals for correlations and binomial intervals
for independent coverage indicators. Report effective independent units and
the conditioning used. Do not apply binomial/chi-square sample counts to raw
correlated image pixels, Hermitian duplicates or shared gain copies. Calibration
moments can be audited from protected draw traces; those traces remain
unavailable to predictors. The proposed RC's 1048 independent corruption draws
may supply many of these checks; actual CI widths, not that count alone, decide.
Any additional cheap nuisance-draw audit must exercise the real forward
routine, not merely repeat its formulas in a separate sampler.

For CASA, first qualify the independently declared/estimated noise model for
the actual selected channels/correlations and imaging operator. Current
WEIGHT=SIGMA=1 is explicitly uncalibrated. The saved paired-noise report gives
approximately 0.05519/0.05505 Jy real/imaginary scatter and a noise-only
border-MAD/predicted-dirty-RMS ratio 1.0813. This is measured limited evidence,
not a 5%-calibrated image uncertainty. Compare identical estimands and account
for gridding and image covariance before drawing a conclusion about that ratio.

**Fresh draws always start from the saved noise-free signal.** Each output is
`S_clean + new_noise` (and explicitly declared fresh calibration if applicable),
never `old_noisy_DATA + new_noise`. Hash the clean reference before/after and
verify the pre-corruption copy equals it. Record seed lineage privately. If
one noise field is deliberately shared across source cases for paired policy
tests, count it once as an independent draw, not once per case.

The user owns a three-seed WSL diagnostic pilot. It is not automatic noise
qualification. Use disjoint preassigned calibration and audit draws if any
empirical uncertainty scale/covariance is estimated; do not estimate a scale
and certify coverage on the same draws. A 3,6,12,24 cumulative-draw ladder is
a proposed maximum bounded pilot expansion per fixed setup, not an instruction
to launch it. With four planned looks, allocate confidence alpha across looks
as well as statistics. Include every earlier draw; do not discard failures or
pick seeds. Expand only if CI width/effective information is insufficient. If
a precise result fails consistency, investigate the noise model/estimator;
more draws or looser tolerance are not a repair. If 24 draws do not achieve
the necessary precision under the agreed resource budget, report unqualified
T/noise claims rather than silently extending the run. Marginal observation
coverage is not later NPE posterior coverage.

## 5. Non-circular physics/numerical gates

These are NEW qualification requirements, not measured results or alterations
of earlier CASA product tolerances. No pass is asserted yet.

1. Units/dimensions, finite values, nonnegative emission at zero background,
   thin/thick limits, PA conventions, spectral shapes and flux units must pass
   throughout the fixed domain. Numerical transfer/reference errors use
   relative 1e-8 for nonzero values or a documented absolute floor 1e-8 K near
   zero. RJ/unit conversions and FFT/point/PSF normalization use relative
   1e-10 at nonzero internal float64 test amplitudes. Released float32/complex64
   arrays must exactly reproduce the requested dtype cast on readback, with
   their quantization included in the joint noise budget; do not demand
   float64 precision from stored float32. Same pinned inputs/seeds are reproducible.
2. Independent numerical reference: integrate transfer with a separately
   implemented quadrature/ODE; compare intrinsic and PB-attenuated forward
   flux, not CLEAN component flux. For a Gaussian EM profile at zero background,
   the infinite-plane flux reference is
   `B_RJ(Te)*pi*q*theta_rad²/(4 ln2)*integral_0^1[(1-exp(-tau0*t))/t]dt`.
   Use stable independent quadrature, with thin/thick analytic limits as cross
   checks, plus independent two-dimensional apparent-flux integration for
   offsets/PB. Numerical intrinsic flux relative error, finite-field lost-flux
   fraction and pixel-field relative L2 must each be <=1e-3. Reference
   refinement must contribute <=10% of each error allowance.
3. More importantly, numerical measurement bias must satisfy
   `sqrt(E_G[delta_F_G_real^T C_N^-1 delta_F_G_real]) <= 0.1` per observation,
   using only canonical independent data and the declared gain distribution.
   This is a joint noise-unit budget, not a per-pixel average that hides
   accumulated error. For Gaussian conditional noise it bounds expected
   conditional KL error by 0.005 nat. It is an engineering accuracy budget
   tied to the observation law, independent of learned reconstruction success.
   Source parameters may control accurate forward integration, never image
   reconstruction settings. Correct quadrature/grid errors instead of
   removing difficult parents or shrinking the prior.
4. Predeclare boundary/interior tests spanning Te endpoints, EM/size/q corners,
   centered and near-subpixel/offset limits, multiple PA values and all bands.
   Include both profiles and turnover challenges. Refine spatial pixels,
   subpixels (e.g.4/8/16;16/32/64 at hard edges), field extent and spectral
   quadrature while comparing the SAME physical quantities and uv coordinates.
   All release records also need a finite-grid/truncation screen and a
   measured or justified conservative error certificate. A validated bound
   may cover interior cases; a few corner tests are not automatically a
   per-record bound. Full high-resolution rerendering of every parent is not
   required if that bound is defensible; otherwise its cost belongs in the
   candidate budget. The reference
   must itself converge; a failed challenge is not legitimate OOD evidence if
   its forward simulation is numerically inaccurate.
5. Physical-law cross-check: retain the existing independent opacity
   approximation comparison, but do not treat agreement between two
   approximations as exact physics. Require an independent established
   precision reference/table with uncertainty <=0.25% for the current radio
   Te/frequency domain before a <=1% opacity-accuracy claim. The existing
   Mezger–Henderson comparison (differences up to about2.56%) is a consistency
   diagnostic, not that precision reference. No1% quantum-physics claim is
   currently qualified. A documented failure requires a better calculation
   or an explicit model-discrepancy treatment before such a claim; it cannot
   be cured by fitted labels.

Maintain all original CASA image/beam/closure gates separately. Reconstruction
residuals, model flux and mixed-beam composites retain their different meanings;
do not use zero residual or component sum as evidence of true total flux.

## 6. Parent lineage, coverage, privacy and integrity gates

- Split underlying physical parents before any variants. All noise, gain,
  rotation, position or instrument descendants of one parent stay together;
  no new ID may disguise a related source. Current two variants are fresh
  noise/gain realizations, not two new physical sources. Challenge parents
  remain outside base splits. No future preprocessing is fitted on validation,
  test or challenge data.
- Preserve full configured prior marginals and independence assumptions.
  For512 Sobol base points, check transformed unit-space coverage, marginal
  quartile counts (128 each, apart from floating-point boundary accounting),
  pairwise occupancy/discrepancy, correlations, extrema and split diagnostics.
  These are design diagnostics, not proof of learned-model sufficiency.
  Require at least40 independent parents in each base validation/test split
  for this RC's interface diagnostics; if the pre-generated private plan fails,
  fix the split design or revise the bounded plan BEFORE observations. Do not
  search private seeds for attractive outcomes. Record all planned counts.
- Nested subsets preserve parent membership and only expose opaque-ID order.
  The actual source/Sobol/noise seeds, source ranks, HMAC key, true gains,
  labels and truth images remain protected. Public config examples must not
  determine a fresh private bank. Red-team replay the original ordering/seed
  attack, plus metadata/filenames/FITS/loader/path/creation-order routes.
  Recovering information from the measurements is intended; exact labels
  from non-observational bookkeeping are leakage. Existing disclosure of the
  legacy 16,384-source bank cannot be retroactively undone by reindexing or
  rotating filenames; it is not a blind evaluation release.
- Before publication, a separately audited public-only load must succeed with
  protected files inaccessible, return only the typed view, and validate
  shapes, units, finite values, packing, sigma meaning and metadata roles.
  Paired loading must link exactly to the immutable ORIGINAL parent values.
- SHA-256 every observation, truth/label record, shared mask/taper/PB/PSF and
  metadata, subsets, source/config/dependency snapshot and run history.
  Independent release audit compares committed inventory with physical files,
  parent groups, variant counts and hashes; no missing/extra uncommitted record
  is published. Current allowlists/checksums are useful but not this complete
  typed scientific release audit.
- Resume only exact source/config/runtime/plan versions, retaining failed and
  interrupted histories. Test source-atomic recovery and incompatible-resume
  rejection on tiny fixtures. Do not edit provenance to unlock old production.
  Freeze executable code before candidate generation and qualify that exact
  snapshot; current old banks cannot be resumed with the new digest.

## 7. Tentative bounded candidate and decision order

No generation is launched by this specification. The tentative candidate is
512 base parents plus12 challenge parents, two variants each:524 distinct
parents and1048 observations. It is an interface/physics/covariance release
candidate, not a full production bank or sample-sufficiency claim. Random
80/10/10 parent splits currently imply roughly410/51/51 base parents, not
guaranteed counts. Preserve nested subsets for later learning curves.

An older verified v2 run generated28 parents/56 observations in23.2262 s wall,
14.2656 s CPU and21,712,042 product bytes. Its mean0.829508 s and775,430 bytes
per parent projects this candidate to434.66 s wall (7.24 min),266.97 s CPU,
406,325,357 product bytes (0.3784 GiB), before initialization/audit overhead.
Another older pilot measured0.366603 s per parent. These are measurements of
older runs, not a fresh benchmark or a timing guarantee. The first run has a
larger fraction of expensive profile challenges, but future accuracy fixes
could still increase cost materially.

Require a small correctness pilot, then a frozen-source benchmark before
starting524 parents. Global NEW output allowance is <=1 GiB and <=600 s
elapsed, including setup, partial outputs, reports and any packaging copies.
No GPU is required or projected. The current generator checks limits between
parents and starts its timer after prepare(), so its flags alone do not impose
these exact whole-command bounds. Reserve startup and one-parent headroom;
an internal<=540 s / <=0.85 GiB guard is a proposed safety margin, subject to
measured worst-parent size/time and initialization cost. Do not start the next
parent if its bound would exceed the remaining budget. Do not kill CASA in a
write to manufacture a timing pass. If accurate generation will not fit,
stop and report an incomplete bounded checkpoint; do not relax science gates,
silently narrow the prior or treat a partial set as the complete512-parent RC.

Release sequence is: fix contract blockers -> freeze source/config/spec ->
independent numerical/noise pilot qualification -> resource benchmark ->
bounded candidate -> independent inventory/privacy/science audit -> explicit
P release decision. Failed or imprecise gates retain failed/inconclusive
status. T remains separate. Later production count and later ML sufficiency
depend on coverage, convergence, costs and learned-model learning curves;
no massive run is authorized here.

## 8. Current evidence assessment and concrete blockers

Relevant physics/instrument/reconstruction/schema/sensitivity/dataset/loader
file hashes exactly match `output/audit_v2/revised_verified/convergence.json`'s
recorded source manifest, although the aggregate project digest has changed
through CASA/tools work. Therefore unchanged-module evidence can be reused
with that provenance, not silently relabeled as a new current-code full audit.

Existing measurements include ODE maximum relative error1.38e-13, one
independent numerical flux comparison at floating precision, opacity
approximation differences -2.56% to-0.128%,32 integration stress cases and
six sensitivity regimes. Worst reported4-versus16 pixel-field errors are
0.00483 Gaussian and0.03457 ellipsoid over that stress suite (including
q=0.3 outside the base prior); configured ellipsoid rendering uses16, whose
worst16-versus32 refinement is0.001411. These do not establish the new
joint measurement-noise error budget or all release-record accuracy gates.
The faint compact diagnostic's1% Te response is only0.0157 in its scaled local
response units, with EM/Te column correlation about-0.99992; bright and
turnover responses differ. This is not a global recoverability theorem.

The saved integrated CASA report now has an existing independent acceptance
pass at `integrated_parity_01/independent_contract_check_02.json`. It closes
that saved case's product contract, not training suitability. The bounded
coarse/3x prediction evidence and exact five-array parity remain valid within
their documented development scope.

**Release blockers at this inspection:**

1. No implemented/tested explicit NPE measurement-only feature view and typed
   uncertainty/packing contract; the generic loader exposes redundant images.
2. Challenge-size target semantics need a protected definition/profile-aware
   paired view; Gaussian FWHM and ellipsoid boundary diameter cannot be mixed.
3. New per-record numerical/forward-flux/joint noise-budget qualification is
   absent; precision physical-law reference remains incomplete. Do not call
   the new gates passed because older, different diagnostics passed.
4. No complete nonlinear thermal-plus-shared-calibration distribution audit
   at the specified precision. Three fresh seeds are a pilot. CASA image-noise
   normalization/covariance needs independent same-estimand verification;
   native unit weights cannot stand in for uncertainty calibration.
5. A fresh current-snapshot512+12 candidate and independent public-only release
   audit do not exist. Legacy disclosure is permanent; a copied old bank is
   not this fresh candidate. Twelve challenges are not robustness coverage.
6. Whole-command resource enforcement/headroom needs qualification. Existing
   CLI limits are source-boundary checks and omit setup from elapsed timing.
7. Historical paragraphs in `docs/MODEL.md` still describe35 m/128 MHz/
   0.3 mJy toy settings and say CASA has not run. They must be clearly marked
   historical/superseded in a release card; the actual compact8-MHz/250-m/
   1.2-mJy config and current measured evidence govern this specification.

Current status: **specified, not yet qualified for a new training-input
release; real-data transfer not certified**. The next coordinated work is
to resolve these concrete blockers and gather only the missing evidence,
not to start inference or another massive bank.

## Sources and records

- [Corrected THOR CDS documentation](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/A%2BA/619/A124?format=html&tex=true), accessed2026-10-21: survey context and corrected-catalogue provenance, not source truth.
- [NRAO Essential Radio Astronomy, free–free chapter](https://www.cv.nrao.edu/~sransom/web/Ch4.html), equations4.57–4.61: EM, opacity approximation and transfer. Different Gaunt conventions must not be mixed.
- [SciPy1.18 chi-square documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html): distribution quantiles for the explicitly conditioned Gaussian variance checks, not a justification for treating correlated samples as independent.
- `docs/MODEL.md`, `docs/DATA_CONTRACT.md`, `docs/INSTRUMENT_AUDIT.md`, `docs/CASA_TERMINAL_MILESTONE.md` document earlier choices and limitations. The cited BeyondPlanck A&A article returned HTTP403 during this inspection; no new precision-accuracy assertion is inferred from inaccessible text or the Planck-band accuracy claim.
- Machine-readable gates, evidence hashes, measured-versus-assumed status and the unchanged-source inspection record are saved under `output/audit/training_readiness_spec_20261021/`.
