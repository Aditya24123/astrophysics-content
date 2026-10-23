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

The initial experiment ran and is now paused. Its selected checkpoint is epoch
22, with validation stopping at epoch 52. All 96 test observations have already
been evaluated once; keep them out of further model selection. Both conventional
Bayesian pilots failed convergence checks. Forward checks were interrupted at
the last console milestone 44/48; 46 cases were subsequently found saved.

Run commands, evidence and limitations are documented in the editable project
[README](../README.md). Local artifacts are under
`output/ml_initial_20261023/run_03/`. `run_initial.py` is not a resume-only command.
No real-data work or upload has occurred. The editable
[next-agent prompt](NEXT_AGENT_PROMPT.md) does not itself resume execution.
