# CASA integration audit and executable path

This document retains the initial integration audit and its historical
Högbom/availability statements. For the current executed WSL path, shared
measurement-only multiscale controls, product acceptance and limitations, see
[CASA_RECONSTRUCTION_PATH.md](CASA_RECONSTRUCTION_PATH.md) and
[CASA_TERMINAL_MILESTONE.md](CASA_TERMINAL_MILESTONE.md). Saved reports retain
the source/configuration that actually produced them.

## Verified repairs (2026-10-19)

The previous runner constructed `project + ".vla.c.ms"` outside the project
folder and selected the noise-free MS despite requesting thermal noise. CASA
6.7.4 documents separate `.ms` and `.noisy.ms` products; NRAO's simulation guide
states that output products reside inside the project folder. The runner now
searches the actual project tree for MS directories containing `table.dat`,
records every discovered name, and requires exactly one `.noisy.ms` when noise
was requested. Missing or ambiguous noisy products fail; the runner never
silently falls back to the noise-free MS. The actual selected path is passed to
`tclean`. No specific array-derived filename is assumed.

Prepared schema-2 bundles contain relative FITS and pointing paths plus SHA-256
checksums, and can be copied intact between hosts. Absolute legacy Windows paths
are rejected with instructions to re-prepare. New inputs and output runs require
new directories; the scripts do not overwrite earlier simulations. The run
report records success/failure, task version, script and input-manifest digests,
actual selected MS, imaging settings and FITS checksums. `completed_unvalidated`
means tasks completed, not that science validation passed. A failed run retains
its partial output and an error report.

Input bundles and CASA run directories are protected research artifacts. Sky
models, simobserve intermediate images, seeds, preparation provenance and
validation case definitions must not be exposed to predictors. The bridge
copies celestial WCS rather than arbitrary truth-header keywords. It transforms
the pointing center to FK5/J2000 and supplies an explicit `tclean` phasecenter,
as recommended by CASA. Model FITS celestial coordinates retain their declared
frame; frame handling and astrometry must still be checked in a real CASA run.

The current runner makes two independent `tclean` calls against the selected
MS. The first is a frozen `niter=0` dirty-image/PSF/PB/residual measurement
product. The second writes a separate CLEAN image/model/residual/PSF/PB set and
cannot read the prepared truth image. In noisy runs the CLEAN threshold is
three times a robust `1.4826022185 × MAD` estimate from the outer 15% image
border of the observed dirty residual; the mask is the full image and the
maximum is 5000 Högbom iterations at gain 0.1. In noiseless controls the
threshold is exactly zero and the iteration cap is the stopping control. The
runner records the final residual peak and scalar fields returned by CASA so
the validator can distinguish a threshold stop, a reported iteration-limit
stop and an unresolved stop state. This border statistic can include source
sidelobes, so it is an observed conservative threshold proxy, not a calibrated
noise map. It should be compared with actual weights and residual statistics
after real runs.

For each selected MS the runner reads `DATA`, `FLAG`, `UVW`, `WEIGHT`, `TIME`,
antenna row IDs, `SPECTRAL_WINDOW/CHAN_FREQ`, `CHAN_WIDTH` and
`ANTENNA/POSITION`. The JSON records row/channel/correlation counts, flags,
visibility amplitude RMS, weight range, projected baseline range, time range,
antenna count, channel grid and antenna-position count. It records hashes for
all exported dirty and CLEAN FITS products. It does not silently derive
simulation visibility products from these summaries.

The post-run validator requires the separate CLEAN files and explicit units
(`Jy/pixel` model; `Jy/beam` restored image and residual). It measures model
component flux and residual aperture sums separately, with the finite PSF core
area convention and apparent PB-attenuated input flux recorded side by side.
That estimate is descriptive and does not correct absent spatial frequencies.
It also computes numerical restoration closure from the exported CLEAN model,
the CLEAN image's declared `BMAJ`/`BMIN`/`BPA` restoring beam and residual. The
local celestial WCS Jacobian supplies pixel geometry; the independently fitted
PSF core is retained as a descriptive comparison. Every dirty and CLEAN image,
PSF, PB, residual and input sky is registered against the same dirty-image WCS,
so a shifted CLEAN group cannot pass by agreeing only with itself. A separate
validation-only comparison convolves `(input sky Jy/pixel × measured PB)` and
the reconstructed CLEAN model with that same declared beam. The input sky is
never passed to `tclean` controls or masks. The small test-double fixture
validates these checks only; it is not CASA evidence.

## Scope and required setup

The bridge makes one **single-channel** Measurement Set per prepared plane. A
2D sky image with `inwidth='128MHz'` does not implement many narrow spectral
channels, frequency-dependent baseline migration, or verified bandwidth
smearing across that SPW. Such products are currently restricted to a narrow
field development comparison, not a validated wideband survey emulator. The
narrowband fixtures below use 8 MHz at 1.44 GHz to test the integration first;
128 MHz imaging needs an explicitly channelized sky cube and further convergence
checks before scientific use where bandwidth effects matter.

The named antenna configuration is CASA's `vla.c.cfg`; actual antenna positions
are retained in the generated MS. One explicit pointing is used, with 30 minutes
at transit, 10-second integrations, reference date 2026/10/19, TOPO spectral
frame, and optional CASA `tsys-atm` noise. These are synthetic design choices,
not the THOR schedule. Default source packages can use the corrected THOR six
usable SPW centers 1.06, 1.31, 1.44, 1.69, 1.82, 1.95 GHz, each 128 MHz wide.
CASA applies the VLA voltage-pattern primary beam. Natural-weighted `tclean`,
`gridder='standard'`, `specmode='mfs'`, `niter=0`, `pblimit=0.2`, `pbcor=False`
produces dirty image, PSF, PB and residual FITS from the selected measurement.
Image pixel count and cell come from the prepared sky. Reconstruction receives
measurements and observing settings, never source parameters.

CASA's built-in receiver/atmosphere noise is an approximation, not an empirical
THOR noise map. No shared gain drift, phase errors, calibration solve/apply,
RFI/flagging, pointing errors, polarization, wide-field w-projection, mosaicking,
or single-dish short spacings are supplied. Extended-flux losses are expected
without short spacings and must be measured separately from residual emission.

## Reproducible commands

From the Python project environment, prepare known 10 mJy point, 45-arcsec and
90-arcsec Gaussian sources, each with a noiseless and noisy package. This is six
bounded scientific input fixtures, not a dataset-generation run:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.venv\Scripts\python tools/prepare_casa_validation.py --out output/audit_casa_inputs_v2
```

The fixture command ran here. Inputs are in `output/audit_casa_inputs_v2`; their
`cases.json` is explicitly marked `inputs_prepared_CASA_not_run`. Each 256-square
image has 2.5-arcsec pixels and an 8 MHz channel at 1.44 GHz. The point is at the
reference pixel. Gaussian size means full intensity FWHM, and all three input
images integrate to 0.010 Jy before primary-beam attenuation. The images are
instrument-validation fixtures, not free-free parameter samples.

On a supported Linux host with CASA 6.7.4 and its CASA data repository including
`vla.c.cfg`, copy the prepared directory and standalone runner. Keep CASA's
bundled Python separate from this project's Python 3.13 environment. Execute:

```sh
for case in point_noiseless point_noisy extended45_noiseless extended45_noisy extended90_noiseless extended90_noisy; do
  casa --nogui --nologger --log2term -c src/hii_sim/casa_simulate.py -- \
    output/audit_casa_inputs_v2/$case output/casa_audit_runs_v2/$case \
    > output/casa_audit_runs_v2/$case.startup.log 2>&1 || exit 1
done
```

Create `output/casa_audit_runs_v2` first. The runner makes a fresh per-case
directory and refuses to overwrite it. Once all six real runs finish, quantify
the products (this command also refuses to overwrite an existing report):

```sh
python tools/validate_casa_run.py \
  --fixtures output/audit_casa_inputs_v2 \
  --runs output/casa_audit_runs_v2 \
  --fast-config configs/compact_proxy_v2.json \
  --out output/casa_audit_runs_v2/validation.json
```

The validator checks the actual exported FITS checksums and grids, point-source
peak flux against the measured primary-beam attenuation (10% relative
tolerance), point position (0.5 pixel), the fitted PSF core against CASA's
image-header restoring beam (10%), and the
`niter=0` image/residual closure (relative L2 at most 1e-3). It records noisy
residual RMS, a windowed positive-core morphology moment for each extended dirty
image, and separately reports the 90-arcsec case's dirty image; it never calls
a dirty-image sum recovered total flux. The moments are descriptive because
the dirty beam has sidelobes and missing modes. It compares genuine CASA images
and PSFs to the artificial Fourier-lattice proxy using relative L2 metrics,
with no equivalence threshold because the antenna tracks, uv coverage, primary
beam and noise model differ. The resulting report is a CASA fixture check, not
evidence of survey-wide fidelity. The thresholds are preregistered engineering
checks for these centered narrowband fixtures; passing them does not validate
calibration, short-spacing recovery or wideband imaging.

The scientific suite is exactly six required fixtures: `point_noiseless`,
`point_noisy`, `extended45_noiseless`, `extended45_noisy`,
`extended90_noiseless`, and `extended90_noisy`. Empty manifests and incomplete
subsets fail with a report listing missing cases. Narrow manifests are accepted
only when the suite and each case explicitly declare `test_double_only`; those
reports have status `test_double_only` and
`scientific_validation_pass: false`.

Every exported image, PSF, PB, residual and input sky must have matching
celestial WCS within 0.05 pixel at corners and interior probes. The PSF must
have a unit center peak within 1e-3 and admit a measured core-beam fit. The image
header must contain finite, positive, ordered `BMAJ`/`BMIN` agreeing with that
fit within 10%. Missing beam metadata, an invalid PSF, or shifted WCS fails the
case; the checker cannot omit these gates. Its synthetic unit fixtures are
explicit test doubles and do not constitute CASA runs or scientific validation.

Use fresh output directory names for another attempt. Preserve startup logs.
After execution, inspect the actual MS `DATA`, `FLAG`, `UVW`, `WEIGHT`, antenna
coordinates, channels and observing times; verify that noise differs from the
noise-free MS. Check image WCS/BUNIT, point peak and position, PSF resolution,
residual statistics, PB attenuation, and extended-source morphology/flux. For
extended flux, report CLEAN components plus a properly scaled residual term in
a declared aperture and compare with the input passed through the same PB and
uv response. A dirty-image sum or CLEAN-component sum alone is not recovered
source flux. These measurements remain required before this branch is validated.

For a protected source FITS cube instead of known fixtures:

```powershell
.venv\Scripts\hii-sim prepare-casa --truth PATH_TO_PROTECTED_TRUTH.fits --out output/casa_input_new --seed 42017 --duration 30min --integration 10s --hourangle transit
```

## Runtime status and remaining validation

The user reports a working isolated Ubuntu 24.04.5 environment with CASA
6.7.4.8, bundled Python 3.12.4, NumPy 2.3.5, SciPy 1.17.0 and Astropy 7.2.0;
`vla.c.cfg` and frozen CASA data are present there. The first genuine noiseless
point attempt failed in `simobserve` before MS creation because the runner
passed an absolute `project` path. CASA joined that absolute path into its
project-relative skymodel path, producing an invalid duplicated path. The
original failed attempt and log remain preserved by the user.

The runner now changes to the fresh absolute output directory for the
`simobserve` call and passes only the channel basename as `project`. It restores
the prior working directory in a `finally` block, keeps sky/pointing inputs
absolute, and discovers Measurement Sets under the absolute `output/tag`
directory. This addresses the reported path construction failure; no new real
CASA attempt was run from this Windows task. Point, extended, noisy and
noiseless genuine cases remain to be rerun by the user. The earlier Windows
WSL setup history is preserved in CASA_SETUP_LOG_2026-10-20.md (local-only artifact: `CASA_SETUP_LOG_2026-10-20.md`).

The next genuine point attempt exposed CASA parsing the bridge's colon-only
declination as nearly 90 degrees. Pointing strings now use explicit sexagesimal
unit markers, for example `J2000 12h00m00.00076072s +29d59m59.99090000s`;
RA remains converted to FK5/J2000 and declination carries an explicit sign.
The corrected six-case sky/input bundle is
`output/audit_casa_inputs_v3`; the earlier v2 bundle and failed CASA attempts
are preserved. A numerical comparison confirms exact sky-array equality and
celestial WCS equality at five image probes for all six cases, unchanged 10 mJy
point/45/90-arcsec source definitions and 1.44 GHz/8 MHz/30-min/10-s settings,
and a maximum parsed pointing-to-image-center separation of 6.04e-5 mas.
These are bundle checks, not CASA execution. Details and hashes are in
`output/audit/casa_pointing_comparison_v3.json`.

Six focused automated tests passed with:

```powershell
.venv\Scripts\python -m pytest tests/test_casa_preview.py --basetemp output/test_tmp_casa_v2 -q
```

They test relocatable bundles, Jy/sr-to-Jy/pixel conversion, header filtering,
actual-name discovery, correct noisy selection, missing/ambiguous-product
failures, checksum/path safety, preservation of earlier runs, and plot axes.
CASA task calls in these tests are explicit mocks producing placeholder files;
those files are not visibility products and provide no scientific CASA
validation. No mocks are used by the normal runner.

## Primary documentation consulted

### Saved-run unit/provenance repair (2026-10-21)

The six genuine CASA 6.7.4.8 runs are now preserved under
`output/casa_actual_20261020/attempt_03/runs`. No CASA task was rerun for this
repair. All 54 native FITS retain their original product-report checksums;
all six native product reports retain the independent batch report's hashes.
The original `validation_01.json` failure report remains unchanged.

The validator records native `BUNIT` and effective role units separately.
Blank residual/PSF/PB units are supported only for the audited CASA 6.7.4.8
tclean producer, bound to the case report and runner by independent runtime
hashes. The historical producer is recognized by its audited runner SHA;
new runners explicitly record `imaging_producer`. Incompatible explicit
units, unknown producers/releases and changed reports are rejected.
The niter=0 residual must exactly equal its explicit Jy/beam dirty image;
PSFs must retain unit central peaks, PB responses must obey normalized bounds,
and declared-beam restoration closure is tested numerically. Headers and
native pixel arrays are never rewritten. Future runner reports obtain
casatasks/casatools versions through `importlib.metadata`, with module
`__version__` as a fallback and `unknown` when unavailable.

Residual units are **Jy/dirty beam**. The native restored image combines a
model term in Jy/clean beam with that residual, so native numerical restoration
closure does not establish a common-beam flux sum. This follows the
[official CASA imaging guide](https://casaguides.nrao.edu/index.php?title=VLA_CASA_Imaging-CASA6.2.0)
and the dirty/clean-beam analysis in
[NRAO VLASS Memo 16](https://library.nrao.edu/public/memos/vla/vlass/VLASS_016.pdf).
The former guide is for CASA 6.2.0; applicability here is checked narrowly
against the actual 6.7.4.8 output roles, explicit image/model units, PSF
normalization and measured restoration closure. No blanket blank-header
compatibility with other versions is asserted.

`validation_04.json` is the final schema-1.1 report: all six declared product
gates pass, while `scientific_validation_pass=false` explicitly preserves the
distinction from overall fidelity. The preceding `validation_02.json` is
retained as an intermediate schema-1.0 report; its old broad boolean is
superseded by schema 1.1. `validation_03.json` is retained before the final
explicit-producer/package-conflict guard. Maximum closure relative L2 is 0.00154782; original
thresholds were unchanged. The noisy extended cases still have matched-beam
model relative L2 errors 0.541918 (45 arcsec) and 0.876318 (90 arcsec).
Their finite-PSF-core component-plus-residual aperture estimates are 7.070
and 0.219 mJy, respectively, versus approximately 10 mJy apparent input.
These aperture conventions and noise-dominated moments are descriptive,
not reliable total-flux or morphology recovery claims. No correction for
unmeasured spatial frequencies was applied. Noiseless runs hit the existing
5000-iteration limit; noisy runs satisfy their recorded residual thresholds.

`batch_runtime_and_noise.json` and `noise_weight_diagnostics.json` are linked
and checksummed as external evidence. Their WEIGHT=SIGMA=1 values remain
uncalibrated placeholders: measured real/imaginary component scatter is about
0.05519/0.05505 Jy and W*Var(real) is 0.003046. Unit consistency W*SIGMA^2=1
does not establish an absolute variance scale. The independent diagnostic's
noise-only border MAD is 1.0813 times its uniformly weighted prediction;
those noise estimates use paired simulator data for validation only.

Reproduce the saved-product checks into another **new** report path:

```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe tools/validate_casa_run.py --fixtures output/audit_casa_inputs_v3 --runs output/casa_actual_20261020/attempt_03/runs --fast-config configs/compact_proxy_v2.json --runtime-provenance output/casa_actual_20261020/attempt_03/batch_runtime_and_noise.json --noise-diagnostics output/casa_actual_20261020/attempt_03/noise_weight_diagnostics.json --out output/casa_actual_20261020/attempt_03/validation_NEW.json
```

The focused runner/checker regressions passed 38 tests in 18.64 seconds;
`output/audit/casa_blank_unit_fix_02.xml` records that run. After strengthening
the explicit-producer/package-conflict rejection, all 25 affected validator
tests passed in 9.44 seconds (`output/audit/casa_unit_provenance_final_01.xml`).
Exact source/configuration hashes are in `validation_04.json` and the saved repair
checkpoint. No production data, observing settings, ML or MCMC were changed.

- [CASA 6.7.4 simobserve task documentation/source](https://casadocs.readthedocs.io/en/v6.7.4/_modules/casatasks/simulation/simobserve.html): output suffixes, single-plane/channel semantics, pointing file format, primary-beam application, J2000 and phasecenter cautions, noise limitations. The rendered task URL failed to load during the audit; the versioned official source/docstring page was accessible.
- [NRAO CASA 6.6.5 ngVLA simulation guide](https://casaguides.nrao.edu/index.php/Simulating_ngVLA_Data-CASA6.6.5): project directory contains outputs. Used only for filesystem behavior, not ngVLA antenna settings.
- [CASA 6.7.4 tclean documentation](https://casadocs.readthedocs.io/en/v6.7.4/api/tt/casatasks.imaging.tclean.html): imaging settings.
- [CASA 6.7.4 release information and known issues](https://casadocs.readthedocs.io/en/v6.7.4/notebooks/introduction.html): separate environment and version limitations. This version notes simobserve issues on macOS 14, so Linux is the suggested validation host.

## MAIN row-flag audit correction (2026-10-21)

`audit_measurement_set` now requires the standard MAIN `FLAG_ROW` column with
exact shape `(nrow,)`, and defines effective sample flags as
`FLAG | FLAG_ROW[None, None, :]`. This follows the
[MeasurementSet v2 definition, MAIN FLAG_ROW](https://casacore.github.io/casacore-notes/229.html):
true marks the entire row as flagged. All flagged rows are excluded from usable
sample fractions, amplitude RMS and usable-weight summaries, even when every
element of their sample `FLAG` array is false. An entirely unusable table is
rejected. These changes affect metadata auditing, not CASA imaging controls.

`sample_flag_fraction` measures FLAG alone; legacy `flag_fraction` remains its
alias. `row_flag_fraction` measures FLAG_ROW over rows, while
`effective_flag_fraction` measures the sample-wise union without double
counting overlapping flags. `valid_unflagged_fraction` additionally requires
finite complex DATA and positive WEIGHT. Legacy `weight_*` summaries retain
their all-column scope, explicitly recorded as `weight_summary_scope`.
`usable_weight_summary` covers each correlation/row having at least one usable
channel, once per native WEIGHT cell. No weight value is interpreted as a
calibrated uncertainty. Existing geometry summaries still describe all MAIN
rows, including flagged ones.

The exact previously reported failure was reproduced before editing:
all eight FLAG_ROW values true with clear FLAG samples yielded usable fraction
1.0 and RMS 0.01 Jy. Regressions now require rejection of that case, verify
partial-row exclusion with independent sample flags/zero weights/non-finite
DATA and distinguish full-column from usable weights. Scalar, two-dimensional
and wrong-length FLAG_ROW arrays are rejected. The focused runner/validator
suite passed **45 tests in 11.71 seconds**, recorded in
`output/audit/casa_flag_row_fix_01.xml`.

The actual preserved six-case MS had clear FLAG_ROW according to the user's
independent casatools inspection; those MS were not reread or regenerated here.
Original FITS, MS, reports and prior checkpoints are unchanged. No new claim
about real MS execution is made by these mock-based regressions. The separate
user-controlled reimaging diagnostic is outside this patch; stopping settings
and all imaging parameters are unchanged. The new checkpoint is
`output/audit/casa_flag_row_checkpoint_20261021.json`.

## Observable multiscale follow-on (2026-10-21)

The frozen proposal and diagnosis are in
[CASA_MULTISCALE_POLICY_V1.md](CASA_MULTISCALE_POLICY_V1.md), with three noisy
development/control policies in
`output/audit/casa_multiscale_policy_20261021/policy.json`.
It is a proposed comparison, not a production-policy switch. Actual CASA
execution remains with the user. The numerical source/grid/beam/uv/noise
inputs reproduce scales [0,7,20,60] without reading physical labels, and each
threshold exactly matches its original noisy run. Existing data, mask,
weighting, observing settings and 5000-iteration cap are held fixed.

Only reporting behavior changes in the current runner: CLEAN requests
`fullsummary=True` and retains the entire JSON-safe CASA return tree, including
global stopping evidence and separate per-plane minor histories. NumPy values
are serialized; non-finite sentinels are explicitly tagged. Global stopcode is
not inferred from minor stopCode, and neither is called scientific convergence.
Exported-residual checks use the exact serialized Jy threshold with no added
tolerance. The post-validator independently recomputes this check and reports
disagreement with legacy flags. The older compact-summary limitation is kept
visible when old reports lack a task stop reason.

The separate noiseless90 diagnostic's recorded residual/threshold ratio is
1.007352; its retained summary does not identify a global task stop. The
strict exported check remains false. No convergence claim or numeric
threshold alteration is made. All 57 focused runner/validator/policy tests
passed in 12.91 seconds (`output/audit/casa_multiscale_policy_tests_01.xml`).
The saved investigation and checkpoint are under
`output/audit/casa_multiscale_policy_20261021/`. No multiscale imaging or new
dataset generation was performed here.

## Frozen second multiscale experiment (2026-10-21)

The actual first basis comparison is preserved under
`output/casa_actual_20261020/attempt_03/closeout_20261021/multiscale_basis_01`.
All 15 new FITS verify against its reports. Each global task stop is recorded
as code 2 / `threshold`, with exported threshold checks retained separately.
Its development improvements are evidence for the basis change, not complete
recovery or independent prediction.

The next single recipe is documented in
[CASA_MULTISCALE_DENSITY_POLICY_V2.md](CASA_MULTISCALE_DENSITY_POLICY_V2.md).
It pairs coarse versus midpoint-inserted scales (current-grid example
[0,7,12,20,35,60]) under identical fit-subset thresholds and controls, and
checks excluded visibility integrations. Both arms must be refit: earlier
full-data images cannot serve as held-out fits. The time selection and scale
rule are frozen without protected labels; the experiment is capped at 15
tclean calls and remains for user-owned execution. No numeric stopping-rule
or acceptance-tolerance reduction is proposed.

Machine-readable policy, signed residual diagnostics and checkpoint are in
`output/audit/casa_multiscale_density_policy_20261021/`. Executable code is
unchanged; the existing 57-test record remains applicable. No CASA task or
generation ran here.

## Integrated shared coarse reconstruction (2026-10-21)

The supported development coarse basis is now available through a shared
measurement-only implementation for both the simulation runner and saved-MS
reimaging. See [CASA_RECONSTRUCTION_PATH.md](CASA_RECONSTRUCTION_PATH.md) for
commands, observing-only configuration, scope and parity requirements.
`configs/casa_reimage_multiscale.json` keeps noise_multiplier=3 provisionally;
CLI/config overrides are recorded exactly. No threshold-challenger outcome is
presumed. Legacy two-argument simulation calls remain Högbom by default and
existing manifests/products are unchanged. The denser ladder is not integrated.

Focused runner/validator/policy integration tests: **66 passed in 17.04 s**,
`output/audit/casa_shared_reconstruction_tests_03.xml`. These include shared
numerical/control parity on explicit mocks, exact multiplier selection,
truth-control rejection, standalone imports, usable-baseline selection,
input-MS content-change detection and failure preservation. Actual CASA
parity of the refactored path remains pending user-owned execution.

Exact executable-source digest:
`46565c3be1d93cec66d3fca5d95023e02f6a6fbf0ffa8d030883f1baac50447c`.
New checkpoint: `output/audit/casa_shared_reconstruction_20261021/checkpoint.json`.

## Terminal bounded closeout (2026-10-21)

[CASA_TERMINAL_MILESTONE.md](CASA_TERMINAL_MILESTONE.md) records the actual
integrated parity, the completed 2x-versus-3x decision and remaining limits.
Coarse multiscale with **3x observed dirty-border MAD** is selected among these
bounded development comparisons. It is not a universal optimum. Denser scales
and the 2x challenger are not promoted; no further sweep is proposed.

User-owned integrated execution on saved extended90_noisy completed in
3.830074 s, preserved original MS content and gave exact equality of all five
CLEAN arrays to the frozen multiscale experiment. The post-success wrapper
metadata hashing error and its separate repair are preserved; no imaging was
rerun. The subsequent old validator path assumption was reproduced and fixed
only for the recognized saved-MS product origin: require exact normalized
absolute agreement of input, channel and externally bound selected MS paths.
Legacy relative suffix checks and all numerical gates remain unchanged.

**72 affected validator tests passed in 9.95 s**, JUnit
`output/audit/casa_absolute_ms_contract_tests_02.xml`. The final guard also
requires external runtime proof for saved-MS origins with explicit native units.
The saved actual report
passes the repaired identity contract; the user will rerun its numerical
acceptance. Source digest:
`c917b382df7097e86af7caeb05b831ac7783dfe1851b66454690d5e5435f639c`.
Checkpoint: `output/audit/casa_terminal_closeout_20261021/checkpoint_02.json`.
