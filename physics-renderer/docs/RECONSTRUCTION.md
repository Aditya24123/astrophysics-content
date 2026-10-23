# Measurement-only image reconstruction baseline

The image path is `measured visibilities + observing weights -> dirty image
and PSF -> signed Högbom CLEAN components -> PSF-derived Gaussian restoration`.
`reconstruct_measurements` accepts numerical measurements, weights, pixel size,
noise RMS and explicit imaging controls. It accepts no physical source, true
image, generating seed or label. The validation fixtures know the test sky;
the reconstruction function does not. Existing production dirty images remain
unchanged, and no bulk regeneration is required to use this baseline.

## Numerical policy and products

The dirty image is the weighted inverse FFT divided by the mean weight, so a
point source has its flux density as the peak in Jy/dirty beam. Högbom iterates
over the largest absolute residual inside a supplied mask, or over the full
field by default. A fraction (`gain=0.1`) becomes a signed delta component,
then its shifted PSF is subtracted. The default limit is 3000 iterations and
the default threshold is three times the declared dirty-image thermal RMS.
The output records the actual iteration count and stopping reason; reaching
the iteration limit does not count as convergence. The residual is recomputed
from the completed component model and measurement PSF to check numerical
closure.

A two-dimensional elliptical Gaussian is fitted to the connected PSF core
above half maximum. The fitting residual, major/minor FWHM and position angle
are retained. A poorly sampled core is rejected. Position angle is north
through east, with image array x increasing west. Restoration is a circular
convolution on the same periodic FFT field as the development instrument. No
unadvertised 4.25-pixel restoring width remains. A caller can explicitly supply
a legacy circular sigma, which is recorded as a choice rather than a fitted
beam.

Products retain separate units:

- CLEAN component image: Jy/pixel. Its sum is the component flux only.
- Restored component image: Jy/clean beam, with a unit-peak Gaussian and its
  discrete beam area. Its full-field sum divided by that area equals the
  component sum.
- Residual: Jy/dirty beam, with the actual unit-peak PSF retained.
- Dirty image, dirty PSF and fitted clean beam are saved independently.

The API retains a clearly named mixed-beam image for diagnostic display; the
FITS command deliberately saves the separate products. Apparent-sky images are
not primary-beam corrected. The fitted clean beam is not evidence that absent
Fourier modes have been recovered.

This follows the conventional restoration structure described in the
[CASA 6.7.4 tclean documentation](https://casadocs.readthedocs.io/en/v6.7.4/api/tt/casatasks.imaging.tclean.html).
The distinction between clean- and dirty-beam units is described by NRAO's
[CLEAN beam documentation](https://www.cv.nrao.edu/~abridle/deconvol/node16.html)
and examined quantitatively in
[Czekala et al. (2021), MAPS II](https://arxiv.org/abs/2109.06188). Our finite
PSF-aperture diagnostics are explicit conventions, not a universal residual
scaling correction or an implementation of all CASA imaging features.

## What the historical 2.423 mJy result means

The audit exactly reproduces the old first-band, 96x96, 2.5-arcsec-pixel case:
a 10 mJy intrinsic circular Gaussian with 90-arcsec intensity FWHM. Gaussian
primary-beam attenuation leaves 9.985948 mJy apparent flux. The historical
35–3400 m configuration did not sample its nominal short-baseline range on
this small FFT field. At gain 0.2, maximum 3000 iterations and a 1 microJy
threshold, CLEAN stops after 844 iterations with:

- 2.422581 mJy summed over all signed components;
- 3.882883 mJy in components inside the predeclared 100-arcsec aperture, and
  -1.460301 mJy of signed components outside it;
- a peak residual of 0.990932 microJy/dirty beam;
- a residual-aperture estimate of 0.009761–0.016723 mJy when the central PSF
  integration radius varies from 0.5 to 3 fitted beam FWHM.

Thus residual emission must be counted separately, but this residual estimate
does not explain the large flux difference. The sum of the full-field PSF is
zero to floating-point accuracy because zero spacing is absent. There is no
valid full-field residual-to-flux conversion using that area. Adding a constant
5 mJy across the apparent sky changes every measured visibility by exactly
zero: the total flux is mathematically unconstrained without additional
measurements or an explicit support/prior assumption. Positive input emission
does not by itself remove this ambiguity. A tiny image residual is not proof
of correct total flux or morphology.

The historical restored image differs from the true apparent sky convolved
with its fitted clean beam by relative image L2 error 0.56595. Positive-part
second-moment widths are about 91.91x57.91 arcsec, versus 92.92x92.00 arcsec
for the known sky convolved with the same beam. These moment widths are
morphology diagnostics, not Gaussian source fits.

## Controls and larger-field comparison

The validation uses an independent, specified image family: a point source,
an offset 30x20-arcsec Gaussian, a 90-arcsec Gaussian, and a 50-arcsec-diameter
uniform disk. All have 10 mJy input flux. It compares the frozen historical
operator against a mathematical complete-coverage Gaussian-taper control that
includes zero spacing. The latter is a diagnostic control, not a realizable
claim about a VLA observation.

For complete coverage, components **plus full-field residual** recover the
specified total flux with measured relative errors from 0 to 4.44e-16 in the
run. Even a deliberately early 50-iteration extended-source reconstruction
has the correct total when the remaining residual is included. Component-only
flux does not pass that test. Restored-model image L2 errors at the longer
6000-iteration diagnostic limit were 9.13e-6 (point), 4.57e-4 (30x20 Gaussian),
7.02e-4 (90 Gaussian), and 3.15e-4 (disk). Some extended cases reach the
iteration limit; their nonzero residuals and stop reasons remain in the JSON.

A separate bounded 1024x1024 field at the same 2.5-arcsec pixel scale samples
shorter Fourier spacings. With the same artificial sampling rule and a 35 m
lower cutoff, its actual shortest sampled baseline is **45.575 m** at 1.06 GHz.
For the 90-arcsec Gaussian it gives 9.891151 mJy of components from 9.985689 mJy
apparent input, a fitted beam of 25.10x24.97 arcsec, restored-model relative L2
error 0.004742, and weighted visibility residual relative L2 0.004505. It stops
after 1410 iterations at a 1 microJy threshold. The fuller coverage improves
this example substantially, but it still lacks DC and cannot establish a
general total-flux recovery guarantee. The report also retains unweighted
visibility residuals, which can be larger because very weakly tapered cells
contribute little to the image objective.

These comparisons are **noise-free** algorithm and spatial-information checks.
Their 0.1–1 microJy thresholds are much lower than the configured 0.3 mJy/beam
production thermal RMS. A 10 mJy, 90-arcsec source has low surface brightness;
the production three-sigma policy can leave it entirely in the residual.
Lower noise, additional short spacings/total power, or a constrained and
validated extended-emission imaging policy is needed before expecting such
objects to be reconstructed reliably. This audit does not validate antenna
phase errors, flagging, calibration solutions, wide-field imaging, or CASA.

## Reproduce and inspect

Run the focused automated regressions:

```powershell
.venv\Scripts\python -m pytest tests/test_reconstruction.py -q
```

Generate the numerical reconstruction report and plot, optionally including the
bounded 1024-pixel comparison:

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python -m hii_sim.reconstruction_validation --config configs/base.json --out output/audit_v2/reconstruction --widefield
```

The report contains actual errors, software versions, source-file SHA-256
digests, the exact tested configuration and its hash, plus a flag if files
changed while it ran. Run once after editing has stopped for the release
record. `output/audit_v2/reconstruction/reconstruction_validation.json` and
the accompanying PNG are numerical diagnostics; the figures are not predictor
inputs.

Reconstruct one observation from a versioned public export only:

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python tools/reconstruct_sample.py --public PATH_TO_V2_PUBLIC --out output/reconstruction_one --index 0
```

This command verifies the public input checksums, saves a numerical FITS file
with WCS, frequencies and explicit beam units, and records input/output hashes
and code versions. It refuses to overwrite an existing reconstruction.
`--allow-legacy-audit` is available only for examining old leaked public data;
it does not make those datasets suitable for unbiased future evaluation.
