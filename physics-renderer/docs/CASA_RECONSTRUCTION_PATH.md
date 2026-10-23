# Shared measurement-only CASA reconstruction

The supported coarse observable multiscale policy now has a callable shared
path for both newly simulated measurements and saved Measurement Sets. It is
supported by the bounded development experiments, not certified as an exact
instrument or complete flux-recovery model. Dense intermediate scales remain
experimental and are not exposed as an integrated option.

`casa_simulate.reconstruct_measurements` is the single implementation of the
dirty image, observed controls, CLEAN call, FITS exports and stopping report.
The simulation runner invokes it after simobserve; `reimage_saved_ms` invokes
it on existing measurements. The latter never opens a prepared source bundle
or true sky image. User-owned actual execution subsequently demonstrated
exact array parity for one saved extended-source case; the independent
acceptance-report rerun after the metadata path repair remains pending.
See [the terminal milestone assessment](CASA_TERMINAL_MILESTONE.md).

## Saved-MS command

In the existing CASA 6.7.4.8 Python environment, from the repository root:

```bash
PYTHONPATH=src python -m hii_sim.casa_simulate reimage \
  --ms /absolute/path/to/existing.ms \
  --settings configs/casa_reimage_multiscale.json \
  --out /absolute/path/to/NEW_reconstruction
```

The MS and output paths are user-supplied. The output must be new and outside
the input MS; existing products are never overwritten. `savemodel=none` keeps
model prediction out of the input MS. The wrapper records before/after hashes
of MS content files, excluding transient `table.lock`, and fails if content
changes. Failed tasks retain partial products and a failed report. A process
interruption can leave incomplete products; do not reuse that prefix or
declare it complete. Start another named output for a retry.

`configs/casa_reimage_multiscale.json` is an observing-only example for the
existing 256-pixel, 2.5-arcsec development field and its recorded J2000
pointing. For other data, set the actual observing grid/pointing. Frequency
and shortest usable projected baseline are read from the selected MS rather
than source labels or a catalogue. The current path is deliberately limited
to a single-channel MFS observation, square orthogonal pixels, standard
gridder, natural weights and uncorrected PB images. It requires a positive
observed noise proxy for multiscale. It does not claim wideband, wide-field,
calibration or real-survey fidelity.

If using CASA's script entry point instead of a package install, keep
`src/hii_sim/casa_simulate.py` and `casa_reconstruction_policy.py` adjacent:

```bash
casa --nogui --nologger -c src/hii_sim/casa_simulate.py -- reimage \
  --ms /absolute/path/to/existing.ms \
  --settings configs/casa_reimage_multiscale.json \
  --out /absolute/path/to/NEW_reconstruction
```

The standalone runner loads the adjacent policy by file path; it does not
depend on the invocation directory for policy imports.

## Explicit threshold control

The configuration currently selects:

```json
"reconstruction": {"policy": "multiscale", "noise_multiplier": 3.0, "noisy": true}
```

The multiplier is **3, selected among the bounded development comparisons**.
The completed 2x challenger worsened excluded-prediction means in all three
cases and both observation halves, so it was not promoted. This does not
establish a universal optimum or calibrated likelihood. No broad sweep or
automatic threshold selection is implemented.
The chosen value can be set explicitly by configuration or
`--noise-multiplier VALUE`; a CLI value takes precedence. This option is not
evidence that an arbitrary multiplier is scientifically validated.

Both paths compute the same `1.482602218505602*MAD` from the observed outer
15-percent dirty-residual border and serialize multiplier times that value as
`%.12gJy`. `threshold_policy` records the exact multiplier, noise proxy,
serialized task argument and parsed task threshold. Effective observing
settings, including CLI overrides, are saved separately from the original
configuration's hash. This statistic can include sidelobes; neither native
WEIGHT=1 nor this border proxy is labeled calibrated thermal sigma.

The coarse ladder remains 0,1,3,9 times measured major-beam FWHM/cell, rounded
half upward and capped by the existing field/short-spacing rule. Controls
come from the generated niter=0 image's WCS/beam, actual SPW frequency and
shortest positive projected baseline among usable rows. FLAG, FLAG_ROW,
finite complex DATA and positive native weights determine usability. The
legacy full-table baseline fields remain unchanged; new usable-baseline
fields prevent flagged short rows from choosing the ladder.

## Compatibility and contracts

The original two-positional-argument simulation command still works and
defaults to legacy Högbom, including its noiseless zero-threshold control.
Prepared schema-2 manifests are not rewritten. For a future explicitly
authorized simulated run, select the shared coarse option with
`--policy multiscale --noise-multiplier 3`; no regeneration was launched here.
Noiseless multiscale needs a separately justified threshold policy and is
currently rejected; legacy Högbom remains available for those controls.

An observing config accepts only the documented instrument/imaging fields.
Starting models, true EM/temperature/size and source masks are rejected. The
shared function's arguments contain no physical labels. The original frozen
`multiscale_policy` v1 function remains unchanged and reproduces old policy
artifacts; `configured_multiscale_policy` adds explicit configuration while
reusing the same coarse geometry.

Outputs retain native dirty and CLEAN image/model/residual/PSF/PB files and
checksummed FITS with WCS. No artistic rendering or parameter-predicted image
is substituted. Headers are not relabeled. The native restored image is a
clean-beam model term plus dirty-beam residual: its composite is not silently
treated as one uniform-beam flux product. No missing-spacing correction is
applied. Native weights retain their uncalibrated interpretation.

Reports retain the full NumPy-aware CASA return tree, distinguish global task
exits from per-plane minor-cycle stops, and compare the exported residual
strictly to the exact serialized threshold. No tolerance was relaxed.
`completed_unvalidated` means tasks completed; `scientific_fidelity_certified`
remains false. Existing scientific gates remain required; current-code parity
has been demonstrated for one actual saved case, while its final acceptance
report is awaiting the user-owned rerun.

## Evidence and focused verification

The preserved excluded-integration development experiment used a documented
pre-execution Högbom amendment: 21 calls in 28.64 seconds, unchanged original
MS content. Independently recomputed coarse-minus-Högbom excluded MSE deltas
are -1.5621e-6 and -1.5465e-6 Jy squared for the two extended cases, with
improvement in each observation half; point predictions match. Dense scales
show no consistent extra benefit. These sources were already inspected during
development and share noise, so this is not a new untouched test-set claim.

Focused tests cover legacy behavior, shared runner/reimage control and
numerical-array parity on explicit mocks, no source-bundle dependence,
FLAG_ROW/usable-baseline behavior, exact multiplier overrides, rejected truth
controls, output preservation, input-MS content-change detection, failure
reporting and standalone imports/CLI forwarding. They do not simulate actual
CASA visibility data. The final integration checkpoint records JUnit results,
exact source/config/test hashes and preserved historical artifacts.

See [CASA 6.7.4 tclean documentation/source](https://casadocs.readthedocs.io/en/v6.7.4/_modules/casatasks/imaging/tclean.html)
for the scale, stopping, savemodel and fullsummary semantics. Project-specific
policy bounds and their scientific limitations remain documented in
[the frozen coarse proposal](CASA_MULTISCALE_POLICY_V1.md).
