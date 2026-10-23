# Editable prompt for the next ML agent

Continue the H II physics-renderer ML stage in this existing project. First read
README.md, ml/plan_v1.json and output/ml_initial_20261023/run_03/. Preserve the
simulator, datasets, reports and frozen checkpoint; never substitute the exposed
legacy bank. Report the current state before starting new work.

Develop using the existing source-grouped train/validation splits and loader.
Keep truth, realized gains, seeds and true sky out of predictor features. Train
a joint neural conditional posterior with a justified bounded CPU budget,
train-only preprocessing and validation stopping rules written before results.
Check numerical correctness, parameter support, uncertainty and degeneracy;
do not assume the neural method wins.

Compare fairly with conventional Bayesian posterior inference using matched
radio measurements, priors, forward physics and nuisance assumptions. Both
existing Bayesian pilots failed convergence; diagnose that before interpreting
them as a reference. Prefer a small declared subset with convergence checks.

The epoch-22 checkpoint was already evaluated on all 96 test observations.
Do not treat those labels as untouched or use them to select new models.
Use a new independent evaluation set before claiming blind generalization for
changed models. Do not generate it or download real data without my direction.

If finishing the interrupted forward checks, load the frozen checkpoint and
resume only missing evaluation cases; run_initial.py has no resume-only CLI.
Keep profile residuals distinct from full posterior predictive validation.
Save reproducible commands, hashes, logs and honest limitations in a new output
directory and update the editable README. No paid/cloud compute, credential
changes, real-data training, upload or GitHub push without explicit authorization.
