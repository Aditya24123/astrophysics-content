# Initial training-data entry point, October 22, 2026

This prepares and loads the existing fresh candidate. **No model training is
performed by these commands.** Start with a four-parameter synthetic inverse
experiment under the declared compact Fourier proxy. This is not a release
for inference on real nebulae.

## Dataset and splits

Dataset: `output/training_readiness_20261021/candidate_gl4_512_01`, schema 2.0,
generated with `configs/compact_proxy_gl4_v3.json`. The immutable generation
code digest is `0a31e1c6c436ca14d4def031a53fb76ee159db7c2b209308277ffb7fb45f2e6a`.
Keep the candidate's provenance/snapshot unchanged when adding downstream tools.

| Split | Independent physical parents | Observations | Initial use |
|---|---:|---:|---|
| train | 397 | 794 | Fit normalization and later model weights |
| validation | 67 | 134 | Later architecture/stopping choices; never fit normalization |
| test | 48 | 96 | Public loading smoke now; protected evaluation after freezing choices |
| challenge | 12 | 24 | Separate stress tests; excluded from this starter |

Each parent has two stochastic observations, each with six bands. The 1,048
observations represent 524 skies, not 1,048 independent sources. All variants
of a sky remain in the same split. Preserve source groups for future resampling
and report both parent and observation counts. Never randomly split rows again.
The older `output/production` 32,792-observation bank is a different, exposed
legacy bank; this starter rejects that schema by default.

## Run the saved starter

From the repository root in PowerShell, use the existing pinned environment:

```powershell
$env:PYTHONPATH = 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
.venv/Scripts/python.exe -B examples/load_training_batch.py --root output/training_readiness_20261021/candidate_gl4_512_01 --prepared output/training_ready_20261022/initial_02
```

The preparation artifact contains `normalization.json`, `normalization.npz`
and `readiness_report.json`. Normalization is bound to the public manifest,
training membership and original target provenance; array hashes are checked
on load. To repeat the complete preparation, select a **new** output directory:

```powershell
.venv/Scripts/python.exe -B tools/prepare_training_data.py --root output/training_readiness_20261021/candidate_gl4_512_01 --out output/training_ready_20261022/recheck_03
```

Preparation checks all committed file hashes, lineage, current core code and
runtime versions; reads all public measurements; fits only training statistics;
then traverses all base batches. It reuses the dataset-bound all-parent numerical
audit and independent noise-law evidence rather than rerunning generation or
CASA. A failed gate produces `FAIL` with its reason. An existing output directory
is never overwritten. No original candidate file is written.

## API and units

```python
from pathlib import Path
from hii_sim.training_data import load_transform, iter_batches

root = Path('output/training_readiness_20261021/candidate_gl4_512_01')
transform = load_transform('output/training_ready_20261022/initial_02', root/'public')
for batch in iter_batches(root, transform, split='train', batch_size=32,
                          shuffle=True, order_seed=22022, include_targets=True):
    x = batch['features']['values']          # float32 [B, 3906], dimensionless
    y = batch['targets']['standardized']     # float32 [B, 4], dimensionless
    original = batch['targets']['original']  # float64 [B, 4], original labels
    # Feed x and y to a subsequently authorized model; no model is supplied here.
```

Inputs are canonical independent **real** Fourier components, with real and
imaginary parts in the stored canonical order and self-conjugate cells real
only. They come from measured complex visibilities in Jy. The base operator
and ordering are identical across train/validation/test. The loader rejects
another layout/operator rather than padding or silently concatenating it.
Dirty/CLEAN images, true sky, source IDs, split, source parameters, seeds and
realized gains are not in `batch['features']`. Original parameters are returned
only under `targets`; bookkeeping stays separate. `order_seed` only shuffles
batch order and has no connection to simulator noise seeds.

Target order is EM0 [pc cm^-6], Te [K], full major-axis **EM FWHM** [arcsec],
axis ratio q [1]. EM0 and size are log10-transformed, Te and q left linear,
then standardized using one vote per unique training parent. Original labels
are unchanged; `transform.original_targets(y)` reverses the transformation.
The size is not necessarily the observed intensity FWHM. PA and positional
offsets are sampled nuisance parameters, not extra input truth.

Feature means/scales are fitted per coordinate from all 794 training
observations only (population standard deviation). This is not per-source
truth/peak normalization. `transform.original_features(x)` returns Jy, subject
to float32 batch rounding. `feature_scale_jy` in each batch makes the conversion
explicit. Target normalization statistics are not predictor inputs.

`batch['observing']` contains permitted, shared instrument metadata in original
units: frequencies GHz, bandwidths MHz, pixel scale arcsec, primary beam,
support/taper, conditional thermal sigma Jy and calibration **prior** parameters.
It contains no realized gain. Thermal covariance is Jy^2, includes the declared
cross-frequency correlation and special self-mode variance. For normalized
coordinates, divide covariance(i,j) by `feature_scale_jy[i]*feature_scale_jy[j]`.
Empirical standardization is **not** whitening or a claim of independent noise.
Shared multiplicative calibration adds signal-dependent covariance, including
across cells; thermal sigma alone is not total uncertainty. Imaging taper is
not inverse variance. Any future likelihood must account for these distinctions.

Use `split='validation', include_targets=True` for validation. Test batches
default to measurements only and explicitly reject `include_targets=True`.
After freezing the experiment, the existing `PairedNPEDataset(..., split='test')`
is the deliberate evaluation path. Preparation parses/checksums protected
lineage but does not use held-out targets for normalization or fit choices.
The test is not a never-seen external set: yesterday's protected scientific
audit checked all parents. Its targets remain excluded from model development.
For strictly public deployment copy only `public/`, and use `NPEInputDataset`;
the full local bank deliberately contains protected labels for supervised work.

## What the PASS does and does not establish

The readiness report separates fresh construction/loading checks from inherited
numerical and statistical tests. The maximum combined numerical/storage metric
0.0109014 is a joint noise-whitened forward error, **not 1.09% image error,
parameter error, or posterior coverage**. The 2,048-draw noise audit checks the
same proxy stochastic law through 38 predeclared statistical gates. Separate
single-frequency CASA photometry tests do not certify this six-band proxy as
a real telescope. CLEAN recovery errors are not errors in original labels.

The base prior is a small smooth, isothermal pure-hydrogen Gaussian family.
The Fourier lattice, global real calibration gains and thermal law are explicit
approximations. Real transfer, learning-curve sufficiency, posterior calibration
and parameter identifiability remain unmeasured by this preparation. In
particular, EM/Te and calibration degeneracies can limit recovery. All weak
and extended prior draws remain; no truth-based quality/SNR selection is used.
Challenge morphology/size semantics and changed sampling require separate,
explicit evaluation rather than mixing into fixed-vector base training.

Next authorized stage would be a small baseline experiment followed by
parent-grouped validation/learning curves and simulation-based posterior checks.
Starting that training requires a further instruction; none has run here.
