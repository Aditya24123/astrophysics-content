# Dataset contract v2 and the immutable legacy bank

New datasets expose `public/` through `PredictorDataset`; the directory can be
copied without `protected/`. Predictor values are observations and permitted
observing metadata. `PairedDataset` explicitly adds selected target parameters
and intrinsic truth. Both loaders reject schema 1 by default; `allow_legacy=True`
is an explicit audit-only opt-in, accompanied by a warning. No future estimator
should accept true parameters through the predictor interface.

The explicit measurement-only `NPEInputDataset` and `PairedNPEDataset` now
provide canonical independent Fourier data, typed thermal/calibration laws
and profile-aware original target definitions. They verify integrity by
default and never open derived images. See [NPE_INPUT_CONTRACT.md](NPE_INPUT_CONTRACT.md)
for the feature layout, protected challenge semantics and cooperative budget
guards. The original generic predictor interface remains image-oriented.

## Verified legacy leakage and its limits

The original schema 1 subset lists were in physical generation order within
split. Regenerating the scrambled Sobol bank from the checked-in seed and
zipping these lists recovered all five parameters for all 16,384 base sources.
HMAC filenames did not prevent this ordered-list attack. The read-only evidence
is `output/audit/legacy_leakage_audit.json`. The original production data and
provenance remain unchanged.

Shuffling lists alone does not make an already disclosed bank secret. An old
recipient can retain the ID/label table or recognize identical observation
content by hashes. Nested membership also reveals coarse membership in the old
Sobol prefixes. A rotated-ID v2 legacy export blocks the direct ordered-list
attack and preserves source groups and nested membership, but is **not** a
secret blind evaluation bank. This limitation cannot be repaired retroactively
without generating new independent sources. No such bulk run is authorized now.

## New-dataset seed and ordering policy

`generate` creates a private random 63-bit design seed at initialization. The
checked-in config seed is an example seed for planning, not the seed used for a
new training bank. The actual configuration, exact source plan, ID HMAC key and
all measurement seeds are stored only under `protected/`. This also prevents
an attacker from reconstructing the complete candidate parameter bank even if
some source ordering is known. Reproduce a source using the saved actual
configuration, source/realization plan and code snapshot; resume uses this same
saved plan. A second new directory deliberately receives an independent design.

The private plan defines splits and nested membership once, before variants.
Public manifest/subset lists use opaque-ID order. Generation and export copying
also use independently keyed ID order, so file creation times and directory
insertion order do not reproduce Sobol rank. Subset membership remains nested;
list position has no physical meaning. Partially generated subset levels are
omitted until every member is committed. Challenge cases remain outside the
three base splits. Future learned preprocessing must use training sources only.

## Paths and permitted metadata

| Path | Contents | Access |
|---|---|---|
| `public/manifest.json` | opaque IDs, split, instrument ID, observation paths, observation and shared SHA-256 values | predictor bookkeeping |
| `public/subsets.json` | fixed nested source membership in opaque-ID order | evaluation bookkeeping |
| `public/observations/*.npz` | sampled complex `vis_jy`, per-channel `sigma_real_jy` | predictor |
| `public/observations/*.fits` | measurement-derived dirty images, units, WCS, frequencies/bandwidths | predictor |
| `public/instruments/*` | observing design, weights, primary beam and PSF | predictor |
| `protected/plan.json` | actual sources, sky offsets, source indices, IDs, seeds, split assignment | generation |
| `protected/subset_membership.json` | full fixed nested membership | audit |
| `protected/labels/*`, `protected/truth/*` | target values, nuisance parameters, corruption draws and intrinsic images | paired training/evaluation |
| `protected/provenance.json` | requested config, actual private config, resume specification, initial checksums | audit |
| `protected/source_snapshot/*`, `source_manifest.json` | exact executable package/tools/dependency declarations and hashes | reproducibility |
| `protected/shared_checksums.json` | weights/beam/PSF archive and observing-metadata hashes | integrity |
| `protected/commits/*` | atomic physical-source commits with every required product hash | resume |
| `protected/runs/*` | retained per-invocation status, code/runtime versions, limits and timing | audit |
| `protected/last_run.json` | convenience pointer-like copy of latest run status | audit |

Real observations can provide visibilities, WCS, frequencies, bandwidths,
weights, beam/PSF models and estimates of noise/calibration uncertainty. The
fast path's exact mask and covariance/gain distributions are synthetic observing
assumptions, not independently measured uncertainty. True gains, centroid,
physical properties, seeds and intrinsic images remain protected. The loaders
allowlist metadata and observation-array keys, reject path traversal, and keep
IDs/splits in a separate bookkeeping object rather than feature tensors.

`verify=True` verifies observations and all shared instrument arrays/metadata.
The FITS dirty beam is unit peak and has sidelobes: a whole-image sum is not an
integrated source-flux estimator. See the reconstruction documentation for
restored images, residuals and beam-area corrections.

## Resume and versioned migration

Resume requires exact requested config, source count, runtime versions and
source digest (including executable tools and dependency declarations). It
also verifies the private plan/snapshot/shared metadata. Changed code does not
become compatible by changing the stored provenance. Existing production was
created by a different code digest and remains immutable; the current generator
refuses to resume it. If its historical source snapshot was not saved, that
missing provenance cannot be manufactured now.

A source is committed only after all of its variants, labels and truth are
written and hashed. An interrupted uncommitted source is regenerated on resume;
existing valid commits are verified and retained. A caught interruption records
a run-history entry and removes the writer lock. A hard process kill can leave
a lock: inspect the PID/run record and confirm no writer remains before manually
removing it. No tool now auto-removes a stale lock. Initialization itself commits
provenance last; a partial initialization is rejected instead of silently reused.

`tools/rebuild_dataset_index.py` now requires explicit `--source` and `--out`:

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe tools/rebuild_dataset_index.py --source output/production --out output/production_export_v2
```

This is a separate **copy export**, not a resume or an in-place repair. It checks
all source commits, rotates IDs and filenames, preserves split and nested set
membership, establishes shared hashes, and retains the original provenance plus
export provenance. Legacy shared arrays were never originally checksummed;
new export hashes establish current integrity and cannot prove their historical
state. The export uses independent copies, not mutable hardlinks. Budget roughly
one additional dataset's disk space. Only small fixtures have been exported in
this audit; this production command is documented but has not been run.

Regression checks (use a fresh workspace-local temporary directory on Windows):

```powershell
.venv/Scripts/python.exe -m pytest tests/test_dataset_security.py -q --basetemp output/audit/pytest-data-security-new
```
