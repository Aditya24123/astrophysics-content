# Bounded simulator/reconstruction repair milestone, 2026-10-21

## Coordinator numerical closeout completed

The independent final checks now pass for all six original fixtures and the integrated saved-MS reconstruction. The integrated five CLEAN arrays exactly match the frozen supported-policy experiment, and the original MS is unchanged. The final record is final_result.json (local-only artifact: `../output/casa_actual_20261020/attempt_03/closeout_20261021/final_artifacts/final_result.json`). This supersedes the pending coordinator actions in the earlier agent assessment below; scientific-fidelity limits remain.

The selected threshold is precisely 3 times the robust scatter proxy, where scatter = 1.482602218505602 times the raw outer-border MAD. It is not calibrated thermal sigma.

## Earlier saved agent assessment

The coarse observable multiscale path is integrated and has one successful
actual saved-MS parity execution. The selected stopping control remains
**3 times the observed dirty-border MAD**, chosen among the bounded
development comparisons below. Neither denser scales nor the 2x multiplier
are promoted. No further sweep is proposed.

This is a usable development reconstruction milestone, not a claim of complete
instrument fidelity, universal recovery, a universally optimal stopping rule
or final training-data certification. The one outstanding closeout action is
the user-owned numerical acceptance rerun on the saved parity products after
the narrow absolute-path contract repair. No new CASA imaging is needed.

## Actual evidence and the selected policy

The same coarse scale rule is used by the simulation runner and saved-MS
entry point: measured beam/grid/usable uv geometry determine the bounded
0,1,3,9-beam ladder; no physical source labels or true source sizes are inputs.
The current full-grid example is [0,7,20,60] pixels. Noise controls use the
observed border statistic, never simulator truth, paired noiseless data or
hidden parameter targets.

The first frozen-basis experiment improves the full-data, model-only
matched-beam relative L2 from 0.541918 to 0.274659 and from 0.876318 to
0.397779 in the extended development cases. Those are nonzero model errors,
not total recovered-image errors: residual emission is separate. Components
alone are not total flux. The noisy point result is unchanged.

The predeclared excluded-integration experiment, with the user's documented
matched-Högbom amendment, used 21 calls in 28.64 s. Both policies were fitted
on the same selected observations; excluded measurements were used only for
prediction scoring. An additional single six-call 2x-versus-3x challenger used
the identical fit/excluded selection without truth-image evaluation.

| Development case | Coarse minus Högbom excluded MSE, Jy² (empirical SE) | 2x minus 3x coarse excluded MSE, Jy² (empirical SE) |
|---|---:|---:|
| Point | 0 (0) | +1.6821e-7 (2.7669e-7) |
| Extended 45 | -1.5621e-6 (7.3707e-7) | +1.3942e-6 (7.2859e-7) |
| Extended 90 | -1.5465e-6 (7.3616e-7) | +2.2267e-6 (8.6606e-7) |

Negative differences favor the first named policy. Coarse improves over
Högbom in both observation halves for both extended cases; dense-minus-coarse
has no consistent benefit. The 2x challenger has positive differences in both
halves for all three cases. The frozen decision therefore retains coarse/3x.
These empirical SEs are descriptive. The development data were already
inspected and the source cases share simulated noise; do not pool them as
independent replications, call this an untouched project test set or interpret
native weights as a calibrated likelihood.

The user then executed the integrated reconstruction on saved full-MS
extended90_noisy in 3.830074 s. All five CLEAN image/model/residual/PSF/PB
arrays are exactly equal (relative L2=0) to the frozen first multiscale run.
The report records unchanged original MS content. Its source snapshots and
runtime evidence bind the exact task versions, source and native report.
A wrapper subsequently failed while hashing the `source/__pycache__`
directory; metadata was repaired separately, without rerunning reconstruction.
The original wrapper and log remain preserved.

Evidence paths:

- `output/casa_actual_20261020/attempt_03/closeout_20261021/multiscale_basis_01/`
- `output/casa_actual_20261020/attempt_03/closeout_20261021/heldout_basis_01/independent_score_verification.json`
- `output/casa_actual_20261020/attempt_03/closeout_20261021/stopping_depth_01/independent_decision.json`
- `output/casa_actual_20261020/attempt_03/closeout_20261021/integrated_parity_01/`

## Narrow acceptance-contract repair

The original acceptance checker assumed every channel MS locator was a
relative path beneath its run case. Reimaging correctly records an existing
external MS with an absolute locator, so the genuine parity report failed that
old metadata assumption. Its original failed check is preserved.

For the recognized `measurement_only_reconstruction_of_existing_ms` origin,
the checker now requires exact lexical-normalized absolute agreement among
report.input_measurement_set, channel.measurement_set and independently
bound selected_ms. POSIX and Windows paths are handled without resolving an
external Linux path on the Windows validation host. Separator/redundant-dot
normalization is supported; case, drive/share/host identity remain exact.
Parent traversal, relative/drive-relative paths, device paths, control
characters, empty/root-only locators and unsupported origins are rejected.
No symlink aliases are guessed. An explicit tclean producer is required for
this saved-MS origin, and independently checksum-bound runtime evidence is
mandatory even if native units are explicit. Legacy relative products retain their case-directory
suffix guard and cannot enter the new branch merely by giving an absolute
channel locator.

Runner/report checksums and CASA package/producer guards remain in force.
No physics, flux, astrometry, PSF, beam, WCS or restoration gate was changed,
and no numerical tolerance was loosened. The affected validator suite passed
**72 tests in 9.95 s**, covering positive and negative origin/path/provenance
cases plus existing scientific checker regressions. The saved actual report
now passes the identity-contract check. Full numerical acceptance was left
for the user's saved-product rerun, not silently declared passed here.

## Remaining scientific and scope limits

- Finite noise and image-correlated noise remain. Border MAD can include
  sidelobes or diffuse emission; 3x this proxy is not a universal detection
  probability. Native WEIGHT=SIGMA=1 is uncalibrated. Current independent
  paired-data noise measurements remain validation evidence, not predictor
  inputs or stopping controls.
- The actual 1.44-GHz, 8-MHz single-channel, 30-min VLA-C development setting
  has incomplete uv coverage (recorded projected baselines approximately
  77.9–3387.5 m), finite field and no zero spacing. Total diffuse flux and
  unmeasured spatial modes are not guaranteed recoverable. The fast Fourier
  proxy remains a separate approximation with substantial differences from
  CASA, not an equivalent realistic array model.
- Native restoration mixes clean-beam model brightness and dirty-beam
  residual brightness. Flux bookkeeping retains both terms and explicit
  finite-PSF-area conventions; it does not manufacture a missing-spacing
  correction. Noise-rectified moments, component sums and model-only L2 each
  have limits and cannot individually establish morphology or total flux.
- The bounded evidence covers simple point/extended development fixtures in
  one observing setup. It does not certify other arrays, frequencies, wideband
  behavior, spatial profiles, background complexity or calibration regimes.
  Phase/gain corruption and calibration processing are still outside the
  tested scope. No real-source/M43 reproduction is claimed.
- The thermal free–free model is an intentionally simplified initial source
  family. Geometry/composition/temperature assumptions and parameter
  tradeoffs remain; five candidate properties are not guaranteed identifiable.
  No density is inferred from EM without geometry/path length and distance.
- Original generating parameters, source-group splits, bank and historical
  outputs remain intact. This closeout neither regenerates nor retroactively
  makes the older exposed-label production bank secret. Its documented
  loader/export restrictions remain. Future ML sufficiency and preprocessing
  decisions belong to later training/evaluation, not this repair.

The [usable shared-path guide](CASA_RECONSTRUCTION_PATH.md) and
`configs/casa_reimage_multiscale.json` retain 3x explicitly. No physical
parameter fitting, ML/NPE, MCMC, bulk generation or further speculative
experiment is part of this milestone.
