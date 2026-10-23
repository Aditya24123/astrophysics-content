# Physics renderer audit — revision 0.2.0

The agreed architecture is unchanged: physical sky → instrument measurements
→ reconstruction from measurements. Protected source parameters remain future
training targets. No inverse model, ML training, MCMC fitting, real-source
inference or new bulk dataset run was performed. Existing production and its
provenance were not rewritten. Small temporary test datasets exercise recovery
and export; the scientific audit uses bounded numerical fixtures.

## Findings verified and repairs

| Finding | Evidence and repair |
|---|---|
| Recoverable public labels | The original attack recovered all five parameters for 16,384/16,384 base sources. Fresh generation uses a protected independently randomized design seed, HMAC IDs and opaque ordering, including file creation order. Nested membership and source splits are retained. V2 loaders reject legacy exports by default. |
| CASA product selection | The runner previously looked outside the project and selected the noise-free suffix. It now discovers actual MS directories inside the project, requires the intended noisy/noiseless product, and rejects ambiguity. Prepared bundles use relative paths and checksums. |
| Missing short baselines | The legacy field has 859.437-wavelength Fourier spacing; its actual minimum baselines are 132–243 m, not 35 m. Guarded compact and widefield configurations and per-channel actual-support metadata make that distinction explicit. |
| Dirty/CLEAN limitations | Measurement-only CLEAN now derives its restoring beam from the PSF, reports stopping conditions, and keeps component, restored-model and residual units separate. Point, Gaussian, disk, DC-control and missing-spacing fixtures measure flux, morphology, resolution and residual closure. |
| Sensitivity errors | One Fourier conjugate per pair; real self-mode variance handled; channel covariance and linearized shared calibration nuisances included; dimensionless coordinates and six regimes replace raw-unit condition claims. |
| Integrity/provenance | Exact resume checks remain strict. Private source snapshots, dependency/tool hashes and retained run histories are added. Weights, PB, PSF, metadata and subsets have checksum coverage. Export verifies original commitments and protected anchors where available. |
| Validation/display | Reports store measured ODE errors and exact code/configuration identities. Thirty-two integration cases include compact elongated and ellipsoid sources; nine spectral-grid cases are checked. Preview Fourier axes are in actual kλ and east is left. |

## Dataset disposition

`output/production` remains a **legacy disclosed target bank**. Rotating IDs
and unordered exports cannot revoke labels/fingerprints already disclosed.
It must not be used as a secret blind evaluation set. Existing dirty images
also retain their original instrumental approximations and integration errors.
The corrected default loader refuses them; `allow_legacy=True` is explicitly
for audit. Keep protected files inaccessible to future predictor processes.

The v2 exporter is implemented and tested on small fixtures. A full ~12 GiB
production copy was not launched. It requires a new destination, retains the
original provenance, rotates IDs, and preserves exact nested memberships.
It does not resimulate or improve legacy physics. The old source snapshot was
not preserved at generation time; it cannot be recreated by stamping today's
code hash onto old products. The original provenance remains authoritative.
The old catalogue of public subsets has no historical checksum, and shared
instrument hashes can establish their current state only. These limitations
are recorded in export provenance.

Read-only before/after hashes match for the ten inspected production
metadata/shared artifacts in `output/audit/production_metadata_before.json`
and `production_metadata_after.json`. This is not a new full rehash of every
production observation. Regression fixtures do verify every committed product,
tampered shared arrays, interrupted-source recovery and incompatible resumes.

## Scientific results

Final audit directories are `output/audit_v2/legacy_verified` and
`output/audit_v2/revised_verified`. Each contains provenance, per-file
checksums, JSON numerical results and PNG previews. The latter tests the
explicit compact v2 scope, not a retroactive correction of production.

The final integrated regression suite passed **40 tests in 41.46 seconds**;
the retained result is `output/audit/tests_final_02.xml`. This includes the
original leakage attack fixture, v2 export/membership integrity, interruption
and strict-resume failures, instrument covariance/DFT checks, physics limits,
measurement-only reconstruction and explicitly mocked CASA runner tests.

- Measured maximum transfer-ODE relative error is **1.38e-13**, separate from
  its acceptance tolerance. The band-averaged flux reference now uses matching
  spectral integration rather than a monochromatic center value.
- Legacy 128 MHz approximation: maximum sampled weighted visibility relative
  error **0.00273**, but maximum individual complex error **8.52 thermal
  component sigma**. This fails a high-S/N approximation criterion. Revised
  8 MHz compact scope: corresponding maxima **1.22e-5** and **0.00386 sigma**
  across the tested cells/cases. This is finite evidence, not a global bound.
- The 90 arcsec historical case reproduces **2.422581 mJy** of CLEAN
  components from **9.985948 mJy** apparent input. In the declared aperture,
  the residual estimate is **0.00976–0.01672 mJy**, depending on PSF-area
  convention. Components inside/outside the aperture are reported separately.
  Residual emission does not account for the missing total. A constant 5 mJy
  addition changes no sampled visibility when DC is absent.
- With a 1024-pixel field and actual 45.575 m shortest sample, the same
  noise-free 90 arcsec fixture gives **9.891151 mJy** of components and
  **0.004742** restored-image relative L2 error. This uses a 1 µJy threshold,
  far below either configuration's noisy-image RMS. It is not a production
  sensitivity result or proof that faint extended sources can be recovered.
- Across the 32 integration stress cases, worst image L2 differences versus
  sixteen subpixels are **2.08% Gaussian / 11.1% ellipsoid** at two subpixels,
  **0.483% / 3.46%** at four, and **0.0959% / 0.759%** at eight. These include
  q=0.3 outside the base q≥0.5 range. Worst-case sixteen-to-thirty-two checks
  give **0.0239% / 0.141%**. Revised configs use four for the Gaussian base and
  sixteen for ellipsoid challenges. Original production used two everywhere.
- EM/Te local column correlations vary by regime, from roughly
  **−0.805 to −1 across the two audited configurations**, rather than one
  universal value. Circular-source PA response is numerical zero. Ellipsoid
  finite-difference information changes by about **1.95%** under step halving
  in the revised case; its sharp edge remains a numerical caution. These are
  local design diagnostics, not posterior uncertainty or target recoverability.

The scientific audit source/dependency/tool digest was
`84c9e564393a37a84fb3998d62a82ef79b217d6116f8d784388f1e03540bb6ad`.
Legacy and revised audits took 42.04 and 92.40 seconds wall time respectively.
Their source-unchanged flags are true and all stored audit-product hashes
were rechecked. `output/audit_v2/final_verification.json` also records the
actual refusal to resume legacy production under the revised code.

The subsequent CLI-default repair changes only config selection and argument
dispatch. Current help and focused CLI regressions verify the compact v2
default, its 1.5 short-spacing guard, and explicit legacy override. Those
13 focused tests passed in 5.19 seconds; the current code digest is
`b774d1d8954a9a8983d2a64042b12e43dd2ac2b22b7ffdf621b1c5778a17737c` at that
verification point. The bounded holdout was generated under that exact digest.
The scientific audit results above are reused for the configurations recorded
in those reports; no unchanged widefield or imaging audit was repeated.
Evidence is in `output/audit/cli_defaults_verified.json` and
`output/audit/tests_cli_defaults_03.xml`.

## Bounded security holdout and additional sensitivity cases

After the CLI repair, a fresh protected holdout used 16 base sources plus 12
challenge sources (4 each profile, turnover, changed-instrument), with two
noise realizations per source. It completed in **23.23 s wall / 14.27 s CPU**,
writing **21.71 MB** (0.830 s and 0.775 MB per physical source on this host).
All 56 public records passed verified loading, and the paired loader reproduced
the protected target values. Every source's two variants share one split.
Replaying the prior checked-in-seed plus public-order attack against this new
export produced **0 exact source-label matches**. The public scan covered four
JSON files and 56 FITS headers; it found no private generating seed, source
index, target fields, or protected FITS keywords. This is a bounded side-channel
check, not a claim that physical source parameters cannot be inferred from
genuine radiation measurements. The recorded result is
`output/audit_v2/secure_holdout_v2/protected/verification.json`; generation
provenance and labels remain in that protected directory.

An additional local Fisher design comparison profiles angular size, axis ratio
and PA, then examines the joint ln(EM), ln(Te) block for a thin resolved and a
turnover compact source. Under the explicitly hypothetical assumption of five
independent 8-MHz channels at 4.5–7.5 GHz with the same 1.2 mJy/channel noise
scale as the L-band proxy, the thin case's local fractional response scales
remain very weak: approximately **65% EM / 192% Te** for L plus hypothetical
C band versus **69% / 202%** for L alone. For the turnover compact case the
same comparison is **4.70% / 3.47%** versus **6.71% / 4.95%**. These are local
linearized design diagnostics, not estimator errors or guarantees. The
hypothetical channels do not reproduce GLOSTAR setup or sensitivity. Adding an
illustrative external 500-K Gaussian Te constraint changes the thin-case local
response scales to **4.30% EM / 6.25% Te**; that is information supplied by an
external constraint, not by this radio-only simulator. Full numbers and
assumptions are in `output/audit_v2/sensitivity_extra_design_v2.json`.

Adding the standalone CASA post-run validator and its focused tests subsequently
changed the current complete code digest to
`ce4a739b76391ab643d70de867d3ca2fadf8c31125f2c68f3b77451fee5ec98a`. Seventeen
focused CASA-preview/validator tests passed in 17.88 seconds; the holdout's
protected provenance correctly retains the digest of the code that generated
it and was not relabeled after the addition.

For extended emission use larger fields/actual short baselines and, where
total-flux recovery is required, suitable short-spacing/single-dish data or
an explicitly declared prior. For 128 MHz science use resolved channels and
validated averaging, not the frozen-baseline proxy. Faint compact-source
shape responses are weak at the revised assumed RMS: more integration,
adequate angular resolution, or restricted reporting scope is needed. A
declared external Te constraint or coverage that measures optical-depth
curvature may help the EM/Te tradeoff, but no configuration guarantees all
five candidate targets. No robustness claim follows from the original twelve
challenge sources or the expanded bounded audit alone.

## Reproduce without touching production

Use fresh output directories; the audit refuses to overwrite a prior run.
With the installed virtual environment, no editable package install is needed
if `PYTHONPATH` is set:

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m pytest -q --basetemp output/audit/tests_rerun_01 --junitxml output/audit/tests_rerun_01.xml
.venv\Scripts\python.exe -m hii_sim.audit --config configs/base.json --out output/audit_v2/legacy_rerun_01 --widefield
.venv\Scripts\python.exe -m hii_sim.audit --config configs/compact_proxy_v2.json --out output/audit_v2/revised_rerun_01 --widefield
.venv\Scripts\python.exe tools/reconstruct_sample.py --public output/production/public --out output/audit_v2/public_reconstruction_rerun_01 --allow-legacy-audit
.venv\Scripts\python.exe tools/preview_sample.py --dataset output/production --out output/audit_v2/preview_rerun_01.png --allow-legacy
```

The public reconstruction command accesses no labels or truth. Its FITS
contains WCS, frequency/bandwidth, fitted beam, model, components, residual,
dirty image and PSF. A restored component model is not automatically a
complete recovered sky; see [RECONSTRUCTION.md](RECONSTRUCTION.md).

## CASA blocker and remaining work

This Windows host has no `casa` executable, `casatasks` or `casatools`; WSL is
not installed. An actual invocation failed with `ModuleNotFoundError:
No module named 'casatasks'`, retained in `output/audit_casa_blocker_v2.txt`.
Six runner/preview tests use explicit CASA test doubles. Six portable point,
45 arcsec and 90 arcsec input bundles (noisy/noiseless variants) are prepared
at `output/audit_casa_inputs_v2`; they contain no fabricated MS products.

Install CASA 6.7.4 with its data/configuration files in a supported Linux or
macOS environment (or set up a suitable Linux host/WSL first), then follow
[CASA.md](CASA.md). Run both noise-free and noisy point/extended fixtures and
measure flux, PSF, missing scales and residuals. Actual CASA execution and
fast-versus-CASA scientific comparison remain unperformed. The package
integration tests and direct-DFT checks do not substitute for that validation.
The post-run checker `tools/validate_casa_run.py` now records measured point
flux, frame-aware sky position, PSF/header-beam agreement, WCS registration of
the input sky and every image product, image/residual closure and fast-proxy
comparison, using predeclared tolerances. Missing or invalid PSF/beam metadata
and shifted WCS now fail closed. The required suite is six named point and
extended-source noisy/noiseless cases; empty or incomplete manifests produce a
failure report and nonzero exit. Narrow suites only run as explicitly marked
test doubles and cannot have a scientific-pass status. Seventeen focused
checker/bridge tests passed. The checker cannot be validated against real CASA
products until CASA is available. The exact current blocker is
`ModuleNotFoundError: No module named 'casatasks'`; no `casa`/`casatools` runtime
or Linux subsystem is available on this Windows host. The smallest required
external action is an existing supported CASA 6.7.4 Linux/macOS runtime with
its `vla.c.cfg` data, or approval to provision one. No visibility products
were fabricated.

The already-generated protected holdout was re-verified in place, not rerun:
all 28 physical sources and 56 observation records passed checksummed predictor
and paired loading, two realizations per source remained within each source
split, and the recorded public seed/order attack remains at zero exact matches.
Legacy production stays disclosed development data, not a blind benchmark.

## Interrupted CASA runner repair continuation

The source-side CASA path now writes independent `niter=0` dirty products and
measurement-only CLEAN products from the selected MS. No source image or
physical parameter is passed to CLEAN. Noisy runs derive the three-sigma
threshold from a documented fixed-border robust statistic on the observed
dirty image; noiseless controls use exactly zero threshold and an iteration
cap. The selected MS is audited for `DATA`, `FLAG`, `UVW`, `WEIGHT`, times,
channels and antenna positions. Product checksums, model/restored/residual
units, actual residual peaks, scalar task stop summaries and model-plus-
residual restoration closure are retained. Extended-source component flux,
residual aperture contribution and the apparent PB-attenuated input are
reported separately. The finite PSF-area residual estimate is descriptive and
does not recover missing spatial frequencies.

The final full suite passed **63 tests in 60.65 seconds** on 2026-10-20, including
the original public-order/seed leakage attack, protected split and resume
checks, shared weight/PB/PSF tamper checks, Fourier coverage and covariance,
physics/convergence, reconstruction, and CASA control-flow/report tests. The
JUnit result is [final_repair_tests_03.xml](../output/audit/final_repair_tests_03.xml).
The exact synthetic CASA checker report is
[report.json](../output/audit/final_repair_tests_03/test_validation_report_records0/report.json);
it is explicitly `test_double_only`, with measured restoration closure
**2.25546e-8**. That is a test-double metric, not a CASA result. The checker
report records source digest `3931a6427366bfd3dcae7d77408bc2e6895da86c6d0cc1a719a5f4e066c6781a`,
validator SHA-256 `26739a3c23e0f9969713d8431d9a37773ecc54865269d6efe74228ae39f4bb41`,
the exact fast test configuration hash, and Python 3.13.14, NumPy 2.2.6,
SciPy 1.18.0 and Astropy 8.0.2. The test config is deliberately a point-fixture
acceptance configuration; it is not an observation recommendation.

Real CASA execution remains **unvalidated**. Current WSL commands return
`Wsl/E_ACCESSDENIED`; no Linux distribution or CASA runtime is available.
Consequently the required six point/extended noisy/noiseless array cases, real
MS table audit, antenna-track comparison, and real CASA flux/morphology/residual
checks did not run. See [CASA.md](CASA.md) and the setup log for the exact
runtime blocker and required post-restart verification. Production and the
existing protected holdout were not rewritten or regenerated.

## CASA checker WCS and restoring-beam follow-up

A bounded adversarial review found that an entire shifted CLEAN product group
could pass because its products were compared only with the shifted CLEAN image
WCS. The checker now compares every dirty/CLEAN/input product to the common
dirty-image celestial WCS. The regression shifts `CRVAL1` by one degree in all
CLEAN FITS products and refreshes their checksums; the case now fails alignment.

Restoration closure now constructs a unit-peak elliptical Gaussian from the
actual CLEAN restored-image `BMAJ`, `BMIN`, `BPA` and local celestial WCS pixel
matrix. Missing or invalid beam metadata fails. A deliberately changed beam
header with image pixels unchanged fails model-plus-residual closure. The PSF
core fit remains separately reported as a descriptive comparison, not as the
beam used for restoration. A validation-only matched-beam diagnostic compares
the PB-attenuated input sky (`Jy/pixel × PB`) with the reconstructed CLEAN model
after applying the same declared restoring beam; no truth enters imaging.

The focused CASA preview/validator suite passed **22 tests in 18.89 seconds**.
Its JUnit record is [casa_checker_final_01.xml](../output/audit/casa_checker_final_01.xml).
The test-double-only report is
[report.json](../output/audit/casa_checker_final_01/test_validation_report_records0/report.json):
measured restoration closure **2.44830e-8**, matched-beam sky/model relative
L2 **0.0**, and test-double status. Current source digest is
`7fa22195b4dffd94081e63c9c2525b2d5d79796b63eeee98567d16f6c717d7bd`.
These measurements validate checker logic only; the six real CASA fixtures
remain unrun pending the separate runtime setup.

### CASA `simobserve` project-name contract

The first genuine CASA attempt then reported that `simobserve` had received an
absolute project path and constructed an invalid duplicated model path. The
runner now invokes it with a slash-free channel basename while its current
directory is the fresh absolute output directory; absolute input FITS and
pointing paths are retained, discovery remains anchored at `output/tag`, and
the original working directory is restored on success and exceptions. A
nested-output regression checks absolute product placement, rejects path-like
project names, and verifies both cwd restoration paths.

The focused CASA runner/checker suite passed **24 tests in 14.91 seconds**;
see [casa_project_name_fix_02.xml](../output/audit/casa_project_name_fix_02.xml).
No real CASA retry was run. Current source digest is
`e4f71ba7d5c5e6956df77c437f61b4f441c7527e5dc30e97a64599156d1545ac`; runner
SHA-256 is `00533bd4ddeef2119f9de3357d4f10658715546e838ec6783c0f5f48f97657e06`.
The actual six-case CASA validation remains pending the user's fresh attempt.

### CASA pointing serialization and v3 validation fixtures

CASA's actual measures parser treated the previous colon-only declination as
nearly 90 degrees, so `simobserve` found no overlap between model and pointing.
The bridge now serializes FK5/J2000 RA with explicit `h/m/s` and signed DEC
with explicit `d/m/s`. Positive and negative declination regressions parse the
tokens back and verify physical values. The focused preview/checker suite then
passed **26 tests in 11.21 seconds**; see
[casa_pointing_fix_02.xml](../output/audit/casa_pointing_fix_02.xml).

A new six-case bundle is saved at
[audit_casa_inputs_v3](../output/audit_casa_inputs_v3), preserving the prior v2
bundle. Its comparison report is
[casa_pointing_comparison_v3.json](../output/audit/casa_pointing_comparison_v3.json).
Across all six cases the sky arrays are exactly equal, celestial WCS matches at
all five tested image probes, settings are unchanged, and parsed pointing
centers differ from the model centers by at most **6.04e-5 mas**. No CASA task
or Measurement Set was generated by this comparison. Current source digest is
`759e810eb0093d10b3a0ab7c6f32603003039b1f30a5e5ccc9696310159b6727`; bridge
SHA-256 is `1dc43e4f40646262101e6c0592cd1524084191676ccf5f89a47e2bb8bf61346d`;
v3 case-manifest SHA-256 is
`aca06cbac4c17131a29b446f030a7386650034f151e1ef403fcfa58da7c8c143`.

### Bounded saved-CASA-product repair completed (2026-10-21)

The user-supplied actual six-case batch supersedes the earlier pending-runtime
status. Native blank residual/PSF/PB units are interpreted through narrowly
verified CASA 6.7.4.8 producer/role provenance, with separate raw/effective
units, exact dirty-image/residual equality, PSF/PB normalization checks and
declared-beam restoration closure. Explicit conflicting units are rejected.
Future runners record installed CASA package versions. See the saved-run
repair section in [CASA.md](CASA.md) for scope, equations/units and commands.

Focused suite: **38 passed in 18.64 s**; JUnit
`output/audit/casa_blank_unit_fix_02.xml`. A final producer/package-conflict
guard then passed all **25 affected validator tests in 9.44 s**; JUnit
`output/audit/casa_unit_provenance_final_01.xml`. All six existing case product gates
pass in `output/casa_actual_20261020/attempt_03/validation_04.json`.
Overall scientific validation remains false: noisy extended recovery and
native weight calibration have material limitations. No tolerance or
observing setting was changed, no CASA rerun was made, and no dataset was
modified. All 54 FITS and six native reports verify against their original
checksums. Original failure and intermediate validation reports are retained.

Exact final executable-source digest:
`f476d5240515e33275031c493c40b0236b7d9ba1d6f8f84bfcf71f7e4ebf4204`.
Validator SHA-256:
`1d8fe01a33929e316fd5472a17f27f63ad32a57659a99194aeb017d985607325`.
Future runner SHA-256:
`b5ce0b93928e814ed58eaa094d0e2bfe6cb5e64f6d573c7da66629cbd5dcdf98`.
The genuine preserved batch used historical runner
`00533bd4ddeef2119fde3357d4f10658715546e838ec6783c0f5f48f97657e06`;
its provenance is linked, never rewritten to match new code.

### Narrow MAIN FLAG_ROW closeout (2026-10-21)

Reproduced and corrected the metadata-audit bug that counted whole-row-flagged
samples as usable. The audit now reads FLAG_ROW, validates one flag per row,
unions it with sample FLAG, rejects fully unusable data and separately reports
sample, row and effective flag fractions. Full-column positive-weight summaries
retain their prior meaning; a distinct usable-weight summary excludes invalid
samples and does not imply calibrated inverse variance. Details and the
MeasurementSet v2 primary reference are in [CASA.md](CASA.md).

Focused runner/validator regressions: **45 passed in 11.71 s**, including
all-row-flagged, partial-row-flagged and five malformed-shape cases. JUnit:
`output/audit/casa_flag_row_fix_01.xml`. Only
`src/hii_sim/casa_simulate.py` changed among executable source files since the
previous checkpoint. New executable-source digest:
`67935528808feeb42cb0307b7e1d3c5db9b8b732958b34b5b31cde07acc1ae93`;
runner SHA-256:
`98dce53fd49491b9e5500ab3925d7fb5b0883f72d40af9ec47a2a42884eeff65`.
The original six runs/reports, `validation_04.json` and previous checkpoint
remain historical evidence at their original source versions. No CASA run or
competing reimaging, dataset work or stopping-parameter change occurred.
New checkpoint: `output/audit/casa_flag_row_checkpoint_20261021.json`.

### Frozen multiscale policy and stop-reporting follow-on (2026-10-21)

See [CASA_MULTISCALE_POLICY_V1.md](CASA_MULTISCALE_POLICY_V1.md) for the
measurement-only proposal, actual noisy-case limitations, exact controls,
primary references and required covariance/flux/morphology diagnostics.
The pure policy helper selects beam/grid/short-baseline-bounded kernel scales;
it accepts no physical parameters and never runs CASA. Current measured inputs
give [0,7,20,60]. The three saved thresholds exactly match the original noisy
runs; the 5000 cap, full-field mask, natural weights and data are fixed.

The proven reporting defect was loss of task stopcodes/descriptions and
structured histories. Future runner calls request fullsummary and retain its
tree. Exported residual checks remain strict, use the serialized threshold,
and are distinct from task exit evidence. The validator now recomputes those
checks instead of trusting legacy booleans; a regression exercises a stale
flag. The external noiseless90 diagnostic remains unresolved with exported
peak/threshold=1.007352; neither threshold relaxation nor convergence is inferred.

Focused runner/validator/policy suite: **57 passed in 12.91 s**; JUnit
`output/audit/casa_multiscale_policy_tests_01.xml`. Frozen policy, investigation
and new checkpoint are in `output/audit/casa_multiscale_policy_20261021/`.
Executable-source digest:
`342c080ab2b619e7b6dc91b3eb14e802e7f6948d9621578b41421ddbcf3b3202`.
No actual CASA task, multiscale experiment or generation was performed here;
existing six-case data and reports remain historical immutable evidence.

### Next frozen density/prediction test (2026-10-21)

Inspected the user-owned actual three-case multiscale experiment and verified
its 15 FITS/report hashes. The saved task summaries stop at their frozen
thresholds, with model-basis improvements reported only as development
evidence. Read-only signed multi-scale residual diagnostics show modest
large-scale contrast; they are not independent significance or a basis for
blindly lowering thresholds.

[CASA_MULTISCALE_DENSITY_POLICY_V2.md](CASA_MULTISCALE_DENSITY_POLICY_V2.md)
and `output/audit/casa_multiscale_density_policy_20261021/policy.json` freeze
one paired coarse/midpoint-scale experiment with fit-only observing inputs
and excluded-visibility prediction. Maximum 15 actual CASA calls, executed
only by the user. No project code change or test rerun occurred; source digest
and the prior 57-test evidence remain unchanged. New observational diagnostic
and preservation hashes are in that directory's checkpoint. No protected
holdout, physical labels, true sizes or noiseless data guide the fit controls.

### Shared coarse reconstruction integration (2026-10-21)

Independent excluded-integration evidence supports coarse multiscale over
Högbom on both extended development cases, with unchanged point predictions;
extra midpoint scales do not consistently improve prediction. The predeclared
matched-Högbom amendment used 21 calls in 28.64 s and preserved original MS.
This is development evidence with shared noise, not a new untouched-test claim.

`casa_simulate.reconstruct_measurements` is now the one dirty/CLEAN/export
implementation used by both run() and reimage_saved_ms(). It accepts observing
settings and current usable MS metadata, rejects source/startmodel/mask
controls, derives coarse scales from measured beam/grid/uv geometry and
records the exact noise multiplier and serialized threshold. Default legacy
run behavior remains Högbom; coarse saved-MS configuration uses provisionally
3x MAD. No new data or real CASA imaging was produced during integration.

Shared full summaries, mixed-beam reporting and strict exported checks are
preserved. Saved-MS calls require new outputs, keep model writes disabled,
hash input content before/after (excluding locks) and fail on mutation.
The original frozen v1 policy function remains output-compatible.

**66 focused tests passed in 17.04 s**, JUnit
`output/audit/casa_shared_reconstruction_tests_03.xml`. Actual independent
parity is pending. Executable digest:
`46565c3be1d93cec66d3fca5d95023e02f6a6fbf0ffa8d030883f1baac50447c`.
Commands and configuration are in
[CASA_RECONSTRUCTION_PATH.md](CASA_RECONSTRUCTION_PATH.md); checkpoint:
`output/audit/casa_shared_reconstruction_20261021/checkpoint.json`.

### Terminal saved-MS origin contract and bounded policy decision (2026-10-21)

The integrated actual saved-MS run has exact five-array parity against the
frozen coarse experiment and unchanged original MS content. Its report,
source snapshots and runtime proof remain intact, including the original
wrapper's post-success metadata error history. The remaining acceptance
integration defect was an absolute-versus-relative MS identity assumption.
For the recognized saved-MS origin only, the checker now compares all three
absolute locators by strict lexical-normalized identity. Unknown origins,
malformed/mismatched paths, wrong package/producer/source/report proof remain
rejected; legacy case-directory suffix behavior and numerical gates persist.

**72 focused validator tests passed in 9.95 s**. Saved-MS origins require
external checksum-bound runtime proof even with explicit native units. Actual saved identity proof
now passes without any CASA execution or image regeneration. User-owned
numerical acceptance rerun remains the sole current closeout action.

The independently verified 2x challenger worsens excluded prediction means
in all three cases and both halves. Therefore coarse/3x is selected among the
bounded development alternatives, with no universal-optimum or calibrated
likelihood claim. Dense scales and 2x are not promoted. Remaining noise,
uv/model/flux and bank limits are assessed in
[CASA_TERMINAL_MILESTONE.md](CASA_TERMINAL_MILESTONE.md).

Current source digest:
`c917b382df7097e86af7caeb05b831ac7783dfe1851b66454690d5e5435f639c`.
Only `tools/validate_casa_run.py` changed among executable files after the
integrated helper version. Checkpoint:
`output/audit/casa_terminal_closeout_20261021/checkpoint_02.json`.

An earlier integrated run under `*_scope_final` is retained as development
history; the revised one was invalidated when a loader repair changed source
files during execution. Only the `*_verified` runs are the final evidence.
Test failures and subsequent corrected runs are retained as XML history.
