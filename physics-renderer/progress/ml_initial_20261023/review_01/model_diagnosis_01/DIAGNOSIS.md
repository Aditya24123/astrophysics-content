> Sanitized publication copy. Private identities, labels, machine paths and task prompts are omitted. Original hashes below identify local originals, not these edited copies. Some linked raw artifacts are intentionally local-only.

# Frozen neural model: validation diagnosis

**Prominent unresolved result:** temperature and axis-ratio validation MAEs exceed their prior-median diagnostics. The source-paired uncertainty below determines how strongly this sample supports that difference. These validation sources were used for early stopping; this is development evidence, not fresh blind inference.

| Target | Unit | Neural MAE | Prior median MAE | Difference | Paired source-resampling 95% interval for difference | Marginal density NLL difference from prior |
|---|---|---:|---:|---:|---|---:|
| em0_pc_cm6 | pc cm^-6 | 33718 | 87035.2 | -53317.2 | [-71435.1, -37314.5] | -0.544409 |
| te_k | K | 1832.04 | 1788.08 | 43.9656 | [-59.6738, 146.744] | 0.023476 |
| major_fwhm_arcsec | arcsec (full major EM FWHM) | 7.53589 | 11.4158 | -3.87992 | [-5.44763, -2.38833] | -0.295433 |
| axis_ratio | 1 | 0.130725 | 0.128671 | 0.00205414 | [-0.00533643, 0.00942681] | 0.0134858 |

Joint density NLL difference from the exact uniform prior in common prior-unit coordinates: -0.694412 nats (negative favors the neural density). All label Jacobians are included/cancelled correctly; original physical units are not mixed in one point-error score. Proper density loss does not directly optimize the realized sample MAE, but an exact posterior median is the Bayes action for expected absolute error. Thus loss/point-summary mismatch is not a justification for calling poorer point estimates accurate.

## Prior dominance and PCA information

The following KL values are descriptive 20-bin histograms, not hypothesis tests; conventional samples are correlated. Small values and prior-like interval widths suggest weak information in those observations, not global non-identifiability. Local Fisher retention uses the FULL correlated thermal covariance transformed through PCA; it is not merely variance retention and is not a global sufficiency proof.

| Case / target | Bayes KL vs prior [nats] | Neural KL vs prior [nats] | Bayes 95% width / prior-unit range | Neural 95% width / prior-unit range | Median PCA retained nuisance-projected local information |
|---|---:|---:|---:|---:|---:|
| 1 / em0_pc_cm6 | 2.04858 | 0.45899 | 0.10381 | 0.64190 | 0.52611 |
| 1 / te_k | 0.00684 | 0.00934 | 0.95465 | 0.93585 | 0.52154 |
| 1 / major_fwhm_arcsec | 2.25580 | 0.09123 | 0.07971 | 0.84939 | 0.16292 |
| 1 / axis_ratio | 1.30890 | 0.00699 | 0.26875 | 0.92875 | 0.19466 |
| 2 / em0_pc_cm6 | 1.93932 | 0.46414 | 0.12226 | 0.67016 | 0.73091 |
| 2 / te_k | 0.00549 | 0.00930 | 0.93883 | 0.94228 | 0.72910 |
| 2 / major_fwhm_arcsec | 1.87129 | 0.01300 | 0.13505 | 0.91045 | 0.44577 |
| 2 / axis_ratio | 0.88541 | 0.00460 | 0.42683 | 0.94332 | 0.57788 |

## Interpretation limits and possible bounded remedy

No leakage, target-unit conversion, precision-factor sampling or point-median implementation defect was found in the frozen route: raw predictions and preprocessing reproduce exactly and analytic/known-distribution regressions pass. This does not prove the estimator is statistically adequate. Only 397 training sources and 67 reused validation sources, one training seed and a single restricted conditional family limit conclusions. The half-data/full-data validation NLL comparison is only one diagnostic and cannot establish data sufficiency.

At zero information, a uniform prior-unit u transforms to a logistic distribution under logit(u), while the model assumes a Gaussian in that coordinate. An otherwise identical Gaussian in probit(u) can represent the exact prior when its mean is zero and covariance identity. One frozen-budget probit-transform sensitivity probe would therefore isolate a recoverable prior-family choice without changing observations or physical priors. It cannot restore information discarded by PCA or ensure a complex posterior is Gaussian. Any such probe must freeze its protocol before training and remain a validation-development comparison with no test scoring or automatic model replacement.
