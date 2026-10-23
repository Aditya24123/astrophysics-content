# Initial synthetic machine-learning stage

This directory is separate from the simulator and the immutable dataset. The
pre-result protocol is `plan_v1.json`. No existing dataset or preparation report
is overwritten. This is an initial experiment in H II ionized gas, not H2.

The neural estimator learns a joint conditional probability distribution for
EM0, electron temperature, full major-axis EM FWHM, and axis ratio. It receives
only the existing loader's radio measurement vector. Labels enter the loss.
PCA and all fitted input scaling use training rows only; source splits are
inherited unchanged. Validation selects the checkpoint, then the protocol and
checkpoint are frozen before test labels are opened once for final evaluation.

A full covariance logit-normal is a deliberately small posterior approximation,
not a claim of exact Bayesian inference or Bayesian neural-network weights.
It respects prior bounds but cannot represent every degeneracy. The conventional
Bayesian comparator uses explicit physics and nuisance priors. It is evaluated
on a predeclared small validation subset, with sampler convergence reported.
Neither algorithm is ground truth. The primary image reconstruction remains
independent of these parameter estimates.

Run commands and resulting evidence will be recorded under
`output/ml_initial_20261023/`. No real-data work or uploads are part of this run.
