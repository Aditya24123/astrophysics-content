# Measurement-only input contract

Implemented 2026-10-21; no neural estimator, training or physical-parameter
fitting is implemented by these interfaces. This contract is for the declared
artificial Fourier proxy. It does not turn its samples into CASA antenna data
or certify transfer to real observations.

```python
from hii_sim.loaders import NPEInputDataset, PairedNPEDataset

public = NPEInputDataset("path/to/public", split="test")  # verify=True
example = public[0]
features = example["features"]
measurements = features["measurements"]
observing = features["observing"]
# example["bookkeeping"] contains IDs/split; never concatenate it to features.

paired = PairedNPEDataset("path/to/dataset", split="train")
inputs, labels = paired[0]  # Explicit protected access for training/evaluation.
```

`NPEInputDataset` reads only a separately distributable schema-2 public
directory. It never opens dirty/CLEAN FITS or protected files. Observation,
shared-operator and subset hashes are checked by default; the image's stored
hash remains in the manifest, but this measurement view does not read/hash
that derived product. `PredictorDataset` remains available for image-oriented
use. Image reconstruction is a separate measurement-only operation.

## Canonical data and uncertainty

`CanonicalFourierData.values_jy` is a real vector in Jy, with parallel
`frequency_index`, `y_index`, `x_index`, `component` and `self_conjugate`
arrays. Coordinates use the unshifted FFT lattice: x increases west and y
north. Band-major cells satisfy `y*n+x <= ((-y)%n)*n+((-x)%n)`. A nonself
cell contributes real then imaginary values; a self-conjugate cell contributes
one real value. Conjugates are checked for consistency, then omitted. Four
self cells exist on an even square lattice, when measured. No independent
information is assigned to their zero imaginary parts. Support may differ
by frequency, so use the explicit indices rather than assuming aligned rows.
Returned arrays are copies marked read-only.

`ProxyObservingMetadata` contains grid sampling, frequencies/bandwidths,
supported cells, imaging taper, primary beam and two typed uncertainty laws.
The taper is **not inverse variance**. The PSF is checked against the stored
taper but is unnecessary for this measurement vector. Unknown or incomplete
metadata, unsupported coordinate/operator conventions, inconsistent array
dimensions, non-Hermitian data, or inconsistent per-band sigma are rejected.
Arbitrary strings from the stored metadata are not copied into features.

`ThermalCovariance.covariance(channels, self_conjugate=False)` returns
`sigma_f*sigma_g*((1-rho)*delta_fg+rho)` in Jy² for the same canonical cell
and same component. Real and imaginary nonself components and different
canonical cells are independent under this proxy. Self-mode covariance is
twice this value; no imaginary self-mode is represented. Select the channels
that actually measure a cell. This is a conditional Gaussian thermal law,
not total measurement uncertainty.

`CalibrationPrior` declares the shared positive lognormal amplitude (mean
one, specified fractional standard deviation) and independent zero-mean
Gaussian spectral log-slope. Its `gain_moments()` returns dimensionless gain
mean/covariance, including nonlinear slope effects. Multiplying the unknown
sky visibility by these gains produces covariance between different cells
and channels and a complex pseudo-covariance. It cannot be added as an
independent diagonal noise floor. The interface exposes the prior law,
never the true gain/slope draw. Both uncertainty objects explicitly identify
their provenance as known **synthetic assumptions**. Real-data use requires
independently estimated/declared uncertainty, and is not supported by silently
filling missing metadata with these defaults.

## Protected targets and challenges

`PairedNPEDataset` returns the exact original selected parameter values;
it checks them against the checksum-anchored generating plan. It neither
fits nor replaces them with CLEAN or catalogue estimates. Label metadata
contains target order, units and original profile/size definitions. New
generation stores these definitions in protected labels. For existing
schema-2 records without annotations, a paired view derives the definition
from the original protected profile and reports `definition_provenance`;
it does not rewrite any stored value or product.

Source integration controls, including the Gaussian GL4 candidate rule/order,
remain protected numerical provenance. They are never appended to observing
features; see [GAUSSIAN_SPATIAL_INTEGRATION_V3.md](GAUSSIAN_SPATIAL_INTEGRATION_V3.md).

Gaussian `major_fwhm_arcsec` means full major-axis projected **EM FWHM**.
The historical ellipsoid field of that name stores full projected **boundary
diameter**. Both original numbers remain unchanged. They must not be treated
as a common training target. The paired output includes
`parameter_definitions.size_definition`; default NPE views exclude all
challenge records. Explicit `challenge_evaluation=True` enables the paired
challenge view (`include_challenges=True` for unpaired inputs), whose labels
are marked `base_training_eligible=False`. Hidden profiles never enter the
feature object. The general `PairedDataset` also requires explicit
`allow_profile_challenges=True` when requesting ellipsoid size. This is an
intentional stricter guard against mixing definitions.

All variants retain their original source-group split and challenge status.
Duplicate realization IDs are rejected across the whole manifest, even when
loading a single split. Opaque IDs and subset membership are bookkeeping,
not input tensors. The schema-1 exposed bank remains audit-only; these
interfaces do not make its previously disclosed labels secret.

## Generation budget accounting

`generate` now starts its elapsed timer before initialization. Its optional
`source_headroom_seconds` and `source_headroom_mib` reserve caller-supplied
measured one-parent/finalization bounds before admitting another parent.
The CLI spells these `--source-headroom-seconds` and `--source-headroom-mib`.
Invalid budgets are rejected before creating outputs. Histories retain
initialization time, reservations, measured elapsed time and bytes before
final history writes. An observed overrun is `budget_exceeded` and causes a
nonzero CLI exit, while preserving valid atomic commits for compatible resume.

These are cooperative admission guards, not hard process/disk quotas.
Initialization itself and a slow parent can exceed a limit; final summary
writes require reserved space/time. The caller must benchmark worst-parent
cost and measure the whole command, including packaging, before claiming the
readiness specification's external resource gate. Strict source/config/runtime
resume checks remain; old production cannot be resumed with this changed code.

## Focused verification

`tests/test_npe_contract.py` tests public-only access without images/protected
files, lossless canonical packing and self modes, uncertainty limiting cases,
original target identity, profile definitions, challenge opt-in, grouping,
and malformed metadata/arrays. `tests/test_generation_budgets.py` uses tiny
fixtures and an injected clock for startup, parent overrun, reservations,
CLI failure and invalid-budget rejection. Security/resume regressions remain
in `tests/test_dataset_security.py`.

These checks close software-contract gaps. They do not qualify the frozen
[readiness specification](TRAINING_DATA_READINESS_SPEC_V1.md)'s new independent
numerical/noise-distribution gates, a fresh release candidate, or real-data
transfer. The separate saved checkpoint records exact source and test hashes.
