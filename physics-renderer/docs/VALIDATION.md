# Validation and bounded production plan

**Historical v1 record, superseded by [AUDIT_V2.md](AUDIT_V2.md).** Original
outputs are preserved. The old ODE report stored an acceptance tolerance as
if measured; old flux comparisons mixed channel averages with monochromatic
integrals; the old Jacobian double-counted conjugates, omitted covariance and
used unit-dependent scales. Those results are not current validation claims.
Production used **two**, not four, subpixels per axis. Its public subset
ordering exposed all five generating parameters. Do not resume this bank
with revised code or treat it as a blind benchmark. Current commands/results
and the separate export procedure are in the audit and data-contract notes.

## What ran

On 2026-10-19, `hii-sim validate` ran with Python 3.13.14, NumPy 2.2.6,
SciPy 1.18.0, Astropy 8.0.2, and Matplotlib 3.11.0. Its JSON report and plot
are `output/validation/validation.json` and
`output/validation/pilot_validation.png`. The calculations use the configured
six 128 MHz THOR continuum centers, a 128x128 grid at 1.25 arcsec/pixel, and a
32-arcsec EM FWHM, q=0.72 source. They are model/code checks, not evidence of
agreement with a real object.

- The full transfer solution agrees with an independent numerical solution of
  dTb/dtau = Te - Tb to the reported solver tolerance; the thick limit gives
  Tb/Te = 1.0000000000000002. Emission is finite and nonnegative.
- The Draine opacity is within -2.6% to -0.13% of the independent
  Mezger–Henderson approximation (NRAO Essential Radio Astronomy Eq. 4.60)
  over Te=5000–12000 K and 1–8 GHz. This is a check of approximate opacity,
  not a precise quantum Gaunt-factor calculation.
- Integrated rendered flux agrees with separate radial quadrature at relative
  error 1.0e-5–1.9e-5. Astropy's Jy/sr-to-RJ-K equivalency matches transfer
  exactly at displayed precision. Three-point channel quadrature differs from
  five points by 1.1e-11 in relative image L2 norm in the pilot case.
- At fixed 160-arcsec field, 2.5, 1.25 and 0.625 arcsec spatial grids give
  monochromatic flux errors of 1.15e-10, 9.95e-11 and 9.56e-11 versus radial
  quadrature. The fast PSF has a 20-arcsec central-lobe FWHM at the pilot
  frequency. One-component Högbom CLEAN recovers 9.999990 mJy from a 10 mJy
  point source, with maximum residual 9.81 nJy/dirty-beam. These are
  algorithm-specific checks, not general clean-flux accuracy guarantees.
- Subpixel factors 1, 2 and 4 differ from an 8x reference by 2.93e-3,
  6.95e-4 and 1.39e-4. Four was used in this fixture; production used two. This case alone does not support it for the
  configured pixel scale, but compact/high-opacity edge cases remain a
  challenge requirement.
- The fast operator is exactly reproducible for a fixed seed. The PSF peak is
  one. A 10 mJy point source has a 10 mJy/dirty-beam peak; one-component
  Högbom cleaning recovers 9.999990 mJy, leaving a 9.81 nJy/dirty-beam maximum
  residual. For a 10 mJy, 90-arcsec Gaussian extended source on a 240-arcsec
  field, low-baseline Högbom components sum to only 2.423 mJy and the maximum
  residual is 0.991 microJy/dirty-beam. Raising the toy minimum baseline from
  35 to 1400 m reduces the absolute dirty-center response to 2.18e-4 of the
  low-baseline value (the high-baseline signed center is a tiny negative
  sidelobe). This exposes both missing-short-spacing loss and the simple
  point-component CLEAN's extended-flux failure. These checks are not a
  general extended-source flux-recovery guarantee.
- At the fiducial source, visibility differences for perturbations of 1% EM,
  80 K Te, 1% major size, 0.01 in q and 1 degree PA are respectively 12.6,
  4.35, 17.8, 14.8 and 7.75 diagonal-noise units. The local EM/Te Jacobian
  correlation is -0.99995 and the five-parameter condition number is
  2.25e7. The diagonal noise metric omits cross-frequency covariance, shared
  calibration uncertainty, parameter-boundary effects and nonlinear posterior
  structure. Treat all five as generated labels; these figures do not certify
  that any one is recoverable.

## Pilot and scaling

The bounded fast-path pilot used 8 base Sobol physical sources plus 12
separate challenge sources, 2 noise realizations per source, 96x96 pixels,
2.5 arcsec pixels and 2 subpixels. The exact command was:

```powershell
.venv\Scripts\hii-sim generate --config configs/base.json --out output/pilot_release_v2 --sources 8 --limit 20 --max-seconds 300 --max-gib 2
```

It generated 20 independent sources and 40 observations in 7.33 seconds wall
time (2.86 seconds process CPU) and wrote 15,508,143 bytes of per-source
products. Measured means are 0.367 seconds and 775,407 bytes per physical
source, including its truth, labels and two variants. Shared metadata/manifest
overhead is small and excluded from the per-source estimate. No GPU is
required or used.

The production run completed with 16,384 independent base sources and 12
separate challenge sources (4 profile, 4 turnover, 4 changed-instrument),
each with two noise realizations: 16,396 sources and 32,792 observations.
The source split is 13,038 train, 1,663 validation and 1,683 test; all 12
challenge sources are held out. The scrambled Sobol base design reaches each
configured marginal boundary within the finite design, with observed minima
and maxima recorded in `output/validation/production_distribution_diagnostics.json`.
Its five marginal rank correlations are exactly zero in this design. The
reproducible histogram and split plot is
`output/validation/production_distribution_diagnostics.png`. These coverage
checks describe the generated design; they do not establish posterior
identifiability or training sufficiency. Nested source-level powers-of-two
subsets are in `output/production/public/subsets.json`.

The production tree contains 114,782 files and 12,793,706,656 bytes
(11.91 GiB), including protected labels/truth, public observations, manifests
and shared instrument metadata. Its source files average about 0.78 MB per
physical source. Across the bounded/resume invocations, observed elapsed wall
time was about **2 h 58 min**, including SHA-256 integrity scans of completed
records on resume. This is substantially slower than the tiny pilot's linear
wall-time extrapolation of 1 h 40 min because resume validation and filesystem
I/O dominate repeated invocations. The first three fully recorded batches
used 23 min 55 sec process CPU in total; the final continuation did not persist
its CPU counter before index recovery. The pilot-based production CPU estimate
was **39 min**, but the complete production CPU total is unavailable and that
estimate should be treated as provisional. GPU use was zero. Use the measured
disk footprint plus headroom for future runs; OneDrive sync and filesystem
load can change wall time.

The historical command below is retained for provenance only. **Do not run it
against existing production with revised code**; the strict resume guard
rejects it. No new production run is authorized in this audit.

```powershell
.venv\Scripts\hii-sim generate --config configs/base.json --out output/production --sources 16384 --limit 4096 --max-seconds 1800 --max-gib 20
```

The production dataset in this workspace was completed through bounded
resumable invocations; its final continuation used
`--limit 9000 --max-seconds 7200 --max-gib 20` after 8,155 sources had already
been committed. Each invocation's time limit applies after checking completed
source checksums, so integrity scans can extend total wall time beyond
`--max-seconds`. The output cap is cumulative. Every source is atomically
committed. The old `tools/rebuild_dataset_index.py` verified newly unindexed
commits only, so its earlier full-verification claim was not justified. The
revised tool requires a separate destination and verifies committed products
for a versioned export; it does not rewrite this dataset. Repeated noise realizations
improve measurement-noise coverage only; they do not increase the number of
independent physical sources.

The 16k source count is a practical first NPE dataset, not a proof of
training adequacy. A learning curve and held-out calibration study are still
required. The 12 challenge sources are only a smoke set; increase and
stratify challenge counts before claiming robust distribution-shift coverage.

## Known validation gaps and measurement guidance

The fast path has no antenna tracks, realistic flagging, w-term, gridding
convolution kernel, direction-dependent gains, or survey noise maps. Its PB,
uv sampling, taper, calibration amplitudes and noise correlations are explicit
development assumptions, not a complete telescope model. The CASA bridge has
not run. Point-source tests establish normalization; the extended test exposes
missing-scale loss. Cross-checking the fast path against representative CASA
outputs remains necessary before using it as a stand-in.

Within 1–2 GHz, low-EM free–free emission is nearly optically thin. Its
continuum amplitude primarily constrains a combination of EM and Te, so the
observed near -1 local EM/Te derivative correlation makes separate inference
unstable even at high S/N. For optically thick/high-EM sources, frequency
coverage through the turnover can add temperature information. For the low-EM
thin regime, a declared external radio-recombination-line Te constraint or
reporting a combined amplitude-like quantity is the concrete route to reducing
the degeneracy. GLOSTAR's 4–8 GHz band is useful for spectral-shape and
compactness coverage but remains optically thin for most base cases; it does
not alone resolve the thin-regime EM/Te amplitude degeneracy. Keep Te and EM
candidate targets configurable, and do not claim five independent recovered
properties. The 12-arcsec end of the size range is unresolved by the fast
path's measured 20-arcsec dirty beam; use a longer-baseline array configuration
or restrict angular-size reporting to sources demonstrably resolved by the
chosen beam. Axis ratio/orientation also need multiple synthesized-beam
elements across the minor axis, and PA is undefined at q=1.
