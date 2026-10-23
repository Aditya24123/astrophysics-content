# Measurement-only correlated flux and residual-accounting scope

`hii_sim.visibility_photometry` implements the fixed formulas independently
audited in `output/training_readiness_20261021/photometry_protocol_v1.json`,
`photometry_measurements_v1.py` and `photometry_validation_01.json`. It is an
array-based numerical helper: it neither runs CASA nor changes an MS, image,
dataset configuration or physical label. No fitting or training is involved.

```python
from hii_sim.visibility_photometry import measured_thermal_calibration, correlated_flux

# Separate measured calibration observations; arrays are complex Jy [RR/LL, row].
thermal = measured_thermal_calibration(calibration_rr_ll,
    flags=calibration_flags, row_flags=calibration_row_flags,
    native_weights=calibration_native_weights)
result = correlated_flux(observed_rr_ll, observed_uvw_m, channel_frequency_hz,
    thermal, flags=observed_flags, row_flags=observed_row_flags,
    native_weights=observed_native_weights)
# result.values_jy: natural and Gaussian120arcsec correlated flux
# result.covariance_jy2: full estimated 2x2 conditional thermal covariance
```

## Exact estimands and assumptions

Only a single channel with unpolarized parallel hands in the explicit order
RR, LL is supported. Real/imaginary receiver components and raw rows must
have equal independent Gaussian thermal noise. Rows must not be artificial
conjugate copies or correlated averages counted as independent samples.
The caller must establish these observing assumptions and supply calibration
observations independent of the target measurements. Independence and absence
of polarization cannot be established from the arrays alone.

Finite complex `[2, nrow]` hand data, matching real UVW `[3, nrow]` in meters,
and a scalar positive frequency in Hz are required. Any flagged sample,
malformed flags, unequal hand/row native weights, polarization declaration
outside this scope, additional channel axes or mismatched rows are rejected.
Passing `flags=None`, `row_flags=None` or `native_weights=None` declares that
the corresponding unflagged/uniform precondition already holds. Uniform
native weights are checked but never used to determine the physical noise
scale. No use is made of native SIGMA.

For separate calibration measurements, `z=(RR-LL)/sqrt(2)` removes the common
unpolarized signal. With n complex differences and k=(n+1)//2, the lower
median power P gives `sigma=sqrt(P/(2 ln 2))`, the per-hand, per-real-component
receiver scale in Jy. Beta(k,n+1-k) order-statistic quantiles give the frozen
99.9% scale interval. A degenerate zero scale is rejected. True source gains,
noiseless signals, source geometry and fitted scatter from the target audit
are not inputs to this calibration routine.

Define `I_j=Re[(RR_j+LL_j)/2]` and two fixed observing-only weights:

    a_natural,j = 1
    a_Gaussian,j = exp[-2 pi² s² (u_j²+v_j²)]
    s = radians(120 arcsec)/sqrt(8 ln 2)
    (u,v) = (UVW_u,UVW_v) * frequency_Hz / 299792458
    F_a = sum_j a_j I_j / sum_j a_j
    Cov(F_a,F_b) = (sigma²/2) sum_j a_j b_j / [(sum_j a_j)(sum_j b_j)]

These are point-normalized phase-center **correlated fluxes** in Jy. An
unresolved point at phase center has its total-flux interpretation under the
declared calibration assumptions. A resolved source generally does not:
different operators intentionally return different spatially filtered values.
No source size, true centroid/gain, source mask, sky image or noiseless
reference is accepted by the operational API. Both statistics use the same
measurements and have nonzero covariance; their errors are not independent.

The reported covariance plugs in the measured receiver scale and is
conditional thermal uncertainty, excluding systematic gain/phase errors,
polarization leakage and scale-estimation uncertainty as a nuisance law.
The separate conservative marginal 95% intervals use the independent 99.9%
scale upper bound and `norm.ppf(1-(.05-.001)/2)` (about 1.96859). The frozen
union bound assigns .001 to scale failure and .049 to Gaussian tails. These
are marginal intervals, not a simultaneous two-operator confidence region,
and are conditional on the receiver model and independent calibration.

The supplied audit used **64** independent actual CASA noise fields, paired
with three saved signals (**192 combinations**, not 192 independent draws).
All eight predeclared gates passed. Full-observation SD ratios were about
1.0055 and 0.9001, with moderate 64-field precision; integration-block results
do not confer that larger sample precision on full observations. This is
bounded evidence for these operators/setting, not zero bias, general CLEAN
photometry, resolved total-flux recovery or real-data transfer certification.
The helper is tested for exact frozen-formula parity, point normalization,
linearity, full covariance via receiver-basis propagation, scale invariance,
protected-argument rejection and domain validation. No new noise campaign
was run to integrate it.

## Historical residual-aperture diagnostic

`tools/validate_casa_run.py` retains its historical numeric fields and units
in `extended_clean_flux_bookkeeping`. New explicit annotations identify it
as **validation-only truth-aperture and finite-PSF-convention accounting**.
Its aperture radius uses the true case FWHM and its center uses the peak of
the input sky. It has no calibrated photometric uncertainty. It must not be
used as an operational total-flux estimator, calibrated missing-flux
correction or predictor feature. A regression with nonzero residuals compares
all historical fields exactly with the immutable 0a31 validator snapshot.

Native CLEAN restoration combines a model restored in Jy/clean beam with a
residual in Jy/dirty beam. A finite dirty-PSF core integral depends on the
chosen radius, sidelobes and truncation, and a wider signed area can approach
zero without zero spacing. Blindly applying a beam-area ratio to residuals
does not restore unmeasured spatial modes or calibrate correlated noise. It
also changes the interpretation of native restored-image values. Any such
convention requires separately justified accounting and uncertainty; this
repair leaves original products and historical numeric estimates intact.
