# Frozen measurement-only multiscale comparison, 2026-10-21

This is one proposed development experiment, not an accepted production
reconstruction policy. The user owns actual CASA execution. No new imaging,
visibility generation, holdout evaluation, training or parameter fitting was
performed here. Original six-case products and prior reports remain immutable.

## Findings and interpretation

The current runner uses full-field Högbom, gain 0.1, a 5000-iteration cap, and
3 times the observed dirty-border MAD in noisy cases. A point-component basis
can model diffuse emission inefficiently; a per-beam peak threshold can also
leave integrated emission in residuals. Neither observation proves that the
visibilities lack all information about the extended source.

The saved 45/90-arcsec noisy development cases have only 3.789/0.828 mJy in
their component models. Their previously reported 0.542/0.876 relative L2
errors compare **convolved model only** against a validation sky. Residual
emission is omitted from that comparison, so these numbers are not total-image
recovery errors. They are existing development diagnostics, never policy
inputs. Positive-only large-aperture moments are biased by noise rectification:
even the noisy point fixture's reported widths are about 52.5 by 42.9 arcsec
against a roughly 16.65 by 15.47 arcsec beam. Those moments alone cannot prove
poor intrinsic morphology recovery.

Native WEIGHT=SIGMA=1 are uncalibrated. The independently paired-data noise
check measures real/imaginary component scatter about 0.05519/0.05505 Jy;
that reference is for validation, not threshold construction. The existing
observed-border MAD includes sidelobes and is a conservative proxy. Keep that
proxy fixed in this first comparison rather than silently replacing it with
the lower reference-data noise.

The recorded minimum projected baseline is about 77.9 m at 1.44 GHz; zero
spacing and many uv samples are absent. That limits unconstrained diffuse
modes, but does not establish that these particular sizes are irrecoverable.
The separate noiseless90 diagnostic went beyond the old iteration cap. Its
9394 iterations, 6.728051e-8 Jy exported residual peak, and 6.678947e-8 Jy
threshold leave a strict peak/threshold ratio 1.007352. The old summary lost
global stopcode/description. Its task reason remains unresolved here; neither
convergence nor a relaxed threshold is inferred.

## One frozen policy

Use `hii_sim.casa_reconstruction_policy.multiscale_policy`. It accepts only
measured beam major FWHM, cell size, square field size, shortest **usable** uv
radius in wavelengths, and observed dirty-border MAD. It neither opens truth
images nor accepts physical labels, true sizes or source masks.

Let b = measured major-beam FWHM / cell size, p = cell size in radians, and
u_min = shortest usable uv radius. Round 1b, 3b and 9b to nearest integer
(half upward), include point scale 0, remove duplicates and reject scales above
min(npix/4, 0.5/(u_min*p)). The quarter-field cap is a provisional edge-control
choice; half the reciprocal-baseline angle is a conservative geometric bound,
not a measured largest angular scale. The geometric ladder is frozen before
any multiscale result. It must not be adjusted to match hidden fixture sizes.

For these measured beams/fields this yields **[0, 7, 20, 60] pixels**, capped
at 64 pixels. CASA scale sizes are kernel diameters, not labels for Gaussian
source FWHM. The policy sets deconvolver=multiscale, smallscalebias=0.0,
gain=0.1, niter=5000, nsigma=0 and the **same** 3*MAD threshold as the original
noisy run. Mask, natural weighting, DATA selection, field, pixel grid,
frequency and PB handling are held fixed. No taper, new mask, threshold
reduction or iteration-cap increase is bundled into this comparison.

`output/audit/casa_multiscale_policy_20261021/policy.json` stores exact
observational inputs, native report/FITS hashes and overrides for the noisy
point control and both noisy extended development cases. Thresholds are
0.000825399364445, 0.000601850030823 and 0.000536408029914 Jy, respectively.
The legacy reports' baseline ranges include all rows. For this batch only,
their clear sample flags and positive/finite-data audit are supplemented by
the user's independent casatools finding that all FLAG_ROW values are false.
For any new flagged data, compute u_min from genuinely usable samples.

Reproduce each saved policy without CASA:

```python
import json
from pathlib import Path
from hii_sim.casa_reconstruction_policy import multiscale_policy
report = json.loads(Path("output/audit/casa_multiscale_policy_20261021/policy.json").read_text())
for entry in report["cases"]:
    assert multiscale_policy(**entry["policy"]["inputs"]) == entry["policy"]
```

For user-owned execution, merge the recorded `tclean_overrides` into that
case's native `dirty_imaging` settings, supply the same Linux MS and a fresh
unique imagename, and retain all other settings. `restart=False`, no startmodel
and `savemodel='none'` avoid warm starts and model writes. Record CASA versions,
effective arguments, full returned summary/log, all products and checksums;
compare original MS content checksums before/after (exclude transient locks).
Do not overwrite the six original run directories. No execution is included
in the Python reproduction snippet above.

## Required diagnostics and decision rule

1. Preserve the complete task-return dictionary with field/channel/stokes
   structure, raw global stopcode/stopDescription and minor-cycle histories.
   A minor stopCode is not a global exit code. Store whole-export and, when
   relevant, clean-mask residual peaks separately from task statistics. Compare
   peaks to the exact serialized threshold, strictly; do not forgive the
   noiseless90 overshoot. A task exit reason is not proof of scientific
   convergence. The runner now requests fullsummary and preserves its tree;
   the validator recomputes the exported check rather than trusting old flags.
2. Compare model, residual and native composite separately against the frozen
   Högbom products on identical grids. Report signed model sum, positive and
   negative model flux, residual peaks and border MAD, measured beam and
   numerical restoration closure. The native composite mixes clean-beam model
   brightness and dirty-beam residual brightness; it is not a uniform-beam
   flux map. Do not divide that composite by clean-beam area as total flux.
3. Plot signed residuals smoothed at the same fixed diameter ladder. For this
   diagnostic use a discrete circular top-hat (pixel-center radius <= scale/2),
   normalized to sum 1; scale 0 is identity. This diagnostic kernel is not
   CASA's multiscale basis. Use linear convolution without wrap; exclude any
   output whose kernel footprint crosses the image boundary. Estimate MAD
   in the remaining pixels of the original outer-15-percent border, with the
   same kernel applied to the entire image, and record the exact valid region.
   Smoothed values retain averaged native brightness units, not Jy per new
   beam. If no adequate background support remains, report noise as undefined.
   This carries image correlation through smoothing. Pixel count does not
   supply independent-beam count or a calibrated detection probability. The
   border may still contain sidelobes. If significant coherent large-scale
   residual persists after the per-beam stop, that is evidence to review the
   stopping policy in a later controlled experiment, not permission to lower
   this experiment's threshold.
4. Use fixed pointing-centered apertures with radii 1b, 3b and 9b, clipped to
   the field, and signed background-aware moments/curves of growth. Flag
   noise-dominated or negative normalization instead of rectifying all
   residual noise into a source. Keep model component flux separate from
   residual aperture sums. If residual sums are expressed as flux estimates,
   declare the finite dirty-PSF integration aperture and show area-convention
   sensitivity (e.g. radii 1, 1.5 and 2 measured major-beam FWHM); undefined or
   non-positive beam integrals remain undefined. No zero-spacing correction
   or preferred convention chosen against truth is allowed.
5. Inspect sampled-visibility consistency on a separate diagnostic copy if
   prediction requires scratch columns. Use actual flags/relative weights,
   independently measured noise and binned real/imaginary residuals; do not
   interpret WEIGHT=1 as 1 Jy variance, duplicate conjugates as measurements,
   or unmeasured zero-spacing as constrained. No visibility fitting algorithm
   or MS mutation is implemented here.

Keep all original product/closure tolerances. An improvement requires reduced
coherent signed residuals without new point-control, negative-bowl, beam or
closure problems. Increased component flux alone is insufficient. Residual
statistics from fitted data are not an unbiased goodness-of-fit distribution.
If the basis comparison fails, record it as a failed policy test. Evaluate
frozen reconstruction against development truth only after settings are fixed;
protected holdout labels never guide settings. No total-recovery promise or
automatic production promotion follows from this experiment.

## Primary documentation and version applicability

- [CASA 6.7.4 tclean source/docstring](https://casadocs.readthedocs.io/en/v6.7.4/_modules/casatasks/imaging/tclean.html): scale-diameter semantics and beam/uv limits; threshold, nsigma and MAD controls; fullsummary; smallscalebias=0.0; restoration and savemodel behavior. The rendered 6.7.4 task/notebook URLs were inaccessible during this investigation; the versioned docstring was accessible. Numerical policy caps above are declared project choices, not CASA-prescribed constants.
- [CASA 6.6.4 Synthesis Imaging, returned dictionary](https://casadocs.readthedocs.io/en/v6.6.4/notebooks/synthesis_imaging.html#Returned-Dictionary): global versus per-plane minor stops and structured convergence histories. It is an adjacent-version reference; new code preserves raw 6.7.4 fields and does not map codes to a claimed convergence status.
- [NRAO VLASS Memo 16](https://library.nrao.edu/public/memos/vla/vlass/VLASS_016.pdf): dirty/clean-beam residual bias and approximate finite-area corrections. Its multi-term VLASS results are not adopted as quantitative corrections for this single-frequency experiment.
