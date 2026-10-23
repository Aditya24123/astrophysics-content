# One next controlled test: scale midpoints with excluded-visibility prediction

Frozen 2026-10-21. This is an experimental recipe, not a production change.
Actual CASA execution belongs to the user. Executable project code is unchanged
from digest `342c080ab2b619e7b6dc91b3eb14e802e7f6948d9621578b41421ddbcf3b3202`.
The existing 57-test record remains applicable; no expensive checks were rerun.

## What the first actual multiscale run established

The preserved `multiscale_basis_01` experiment contains three genuine CASA
6.7.4.8 runs with unchanged original MS content, global stopcode 2 and task
description `threshold`, at 26/25/35 iterations. Its 15 exported FITS verify
against the experiment reports. The full task summaries and strict exported
checks are retained separately, including CASA's finite-precision internal
threshold value; no numeric threshold was changed to reconcile them.

The provided post-freeze development comparison improves the model-only L2
errors from 0.542 to 0.275 and 0.876 to 0.398 for the extended cases. Component
sums rise to 8.576 and 7.643 mJy; the point control stays unchanged. This
supports the basis-policy diagnosis. It does not demonstrate total flux
recovery, independent prediction or scientifically complete reconstruction.
Residual emission and unmeasured spatial modes still matter.

Read-only signed residual diagnostics are saved in
`output/audit/casa_multiscale_density_policy_20261021/observed_residual_diagnostics.json`.
They use the already specified top-hat filters at diameters 0,7,12,20,35,60
pixels, an unchanged outer-15-percent valid border, and pointing-centered
radius-60-pixel search region. For the two extended cases, signed positive
peak/border-MAD ratios are approximately 2.96/2.27 at diameter 35 and
2.68/3.00 at diameter 60. These modest ratios are not Gaussian significance:
pixels are correlated, positions/scales were searched, residuals were fitted,
and the border may contain sidelobes. They do not justify simply cleaning
deeper or declaring a hard noise limit.

## The single question and controlled variable

Test whether finer spacing of the same bounded multiscale basis improves
prediction on measurements excluded from each fit. Preserve the coarse scale
ladder and insert the rounded geometric midpoint between consecutive positive
diameters: floor(sqrt(a*b)+0.5). This gives the current-grid example
**[0,7,12,20,35,60]**, versus coarse **[0,7,20,60]**. The maximum scale and
spatial-frequency/field cap do not increase. No size labels, source masks,
truth amplitudes or physical parameters enter this rule.

The machine-readable frozen recipe is
`output/audit/casa_multiscale_density_policy_20261021/policy.json`.
Run the two arms on the noisy point control and the two noisy extended
development cases. No parameter sweep, threshold sweep or production
promotion is included.

## Exclude measurements before reconstruction

1. On each original MS, sort the distinct exact MAIN TIME values. Exclude all
   rows at zero-based time ranks divisible by five; fit the others. Keep all
   baselines, channels, correlations and related/conjugate entries from one
   integration together. Use the same groups for both arms; record the exact
   membership and its hash. This is a visibility selection within development
   sources, not a change to dataset train/validation/test source splits.
2. Use new diagnostic MS copies with original DATA, UVW, weights, channel
   metadata and original flags unchanged. Add selection flags only to the
   copies: exclude test integrations from the fit copy and fit integrations
   from the prediction copy. Apply original FLAG_ROW, sample flags, finite
   DATA and positive relative weights. No original MS or noise values change.
   Verify DATA and original MS content hashes before/after. Fail if selection
   is empty or there are too few time groups to assess variation.
3. Make one niter=0 image from fit integrations only. Measure its beam,
   shortest usable fit uv radius and border MAD. Freeze the coarse ladder by
   the existing v1 observable rule and the dense ladder by midpoint insertion.
   Set the common threshold to 3 times that fit-only dirty-border MAD,
   serialize it once as `%.12gJy`, and share exactly across both arms. Record
   these quantities before either CLEAN. Never estimate this threshold from
   excluded data or a noiseless reference. It can differ from the earlier
   full-MS threshold because the data subset changed; it is not tuned or
   lowered to improve either arm's image agreement.
4. Fit both arms from an empty model and fresh unique image prefixes. Keep
   deconvolver=multiscale, smallscalebias=0, gain=0.1, niter=5000, nsigma=0,
   natural weighting, original grid/phasecenter/PB treatment and full-field
   component-center mask. Request fullsummary. Use savemodel=none. Only the
   scale ladder differs between the arms. Do not compare a subset-trained
   dense model to the previously full-data-trained coarse model as though
   that were a controlled comparison.
5. After both fits, predict each reconstructed `.model` on excluded integrations
   using CASA's same-gridder predict-only path on diagnostic MS copies:
   niter=0, calcpsf=False, calcres=False, restoration=False,
   savemodel=modelcolumn, a fresh prediction imagename and startmodel set to
   that reconstructed image. This is prediction from a measurement-derived
   image, not the simulator truth. Preserve each prediction before the next
   arm, verify finite MODEL_DATA on evaluated samples, and never write model
   columns to original MS. If this established prediction path fails, retain
   the failure; do not replace it with an unvalidated Fourier approximation.

Bound: three fit-only dirty calls, six CLEAN fits and six predict-only calls
(15 tclean calls total, six fits capped at 5000 minor iterations). No retries
or additional variants are preauthorized by this recipe. No actual call was
made here. The user's runner should record actual time/storage and stop with
a clear failure if required metadata or products are missing.

## Evaluation and interpretation, fixed before new truth comparisons

For each usable excluded complex sample, form

`d_i = |DATA_i - MODEL_dense_i|^2 - |DATA_i - MODEL_coarse_i|^2`.

Report native-relative-weighted means per excluded TIME group, the pooled
weighted mean, equal-time mean, empirical scatter/SE and signs in the first
and second halves of the observation. Negative values favor dense scales.
Weights are relative: WEIGHT=1 is not calibrated inverse variance. Count
recorded measurements only; do not add artificial conjugates. Retain real
and imaginary residual means/RMS and baseline-binned summaries to expose
coherent error rather than only one large aggregate number.

Time grouping keeps within-integration correlations together. Existing noise
diagnostics suggest small adjacent-time/correlation dependence for this fixed
uncorrupted simulation, but do not prove independence. Any time-group interval
is descriptive unless that covariance assumption is supported. The three
development cases share the same simulated noise realization; they cannot be
pooled as independent noise replications. These observations have also been
inspected in earlier development: the excluded rows are withheld from these
new fits/noise estimates, not an untouched final project test set. Protected
holdout sources and labels remain unopened.

Retain the original point/beam/closure gates and strict stop reporting. Compare
signed model/residual/composite separately; use the previously declared
mixed-beam flux and finite-PSF-area sensitivity conventions. Increased model
flux or lower post-fit truth-image error alone cannot select the denser ladder.
Shortlist it only if excluded predictions improve consistently for both
extended development cases without point-control degradation, new negative
bowls or product-gate failures. Small/noisy/mixed changes are inconclusive;
do not force a winner with tolerances or label-guided settings.

This test isolates extra scale freedom. It deliberately does not change the
per-beam stopping rule or claim that rule is optimal. If prediction does not
improve, that constrains evidence for denser scales at this noise/threshold,
not all possible recovery. A later noise/stopping experiment would need its
own frozen controls and independent check; it is not bundled here.

## Official method reference

[CASA 6.7.4 tclean source/docstring](https://casadocs.readthedocs.io/en/v6.7.4/_modules/casatasks/imaging/tclean.html)
defines scale diameters and uv/field limits, global peak stopping, and separate
predict-only model-column saving. Midpoint insertion and the time partition
are declared project experimental choices, not CASA defaults. Preserving the
same gridder during prediction avoids substituting a different measurement
operator. The exact prediction workflow still requires actual user-owned
execution and checking.
