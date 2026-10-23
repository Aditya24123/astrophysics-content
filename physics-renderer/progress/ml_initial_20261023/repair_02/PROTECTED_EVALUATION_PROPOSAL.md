> Sanitized publication copy. Private identities, labels, machine paths and task prompts are omitted. Original hashes below identify local originals, not these edited copies. Some linked raw artifacts are intentionally local-only.

# Proposed fresh protected evaluation — approval required before generation

**Proposal only. Nothing in this document authorizes or launches generation.**
The original 96 test observations have already been scored. Continued model
development uses the existing 134 validation observations and cannot recover a
blind test by rescoring the original archive.

If a future model is selected, propose 256 fresh physical source groups in a
separately seeded protected bank, each with two independently corrupted
observations (512 observations).
Use the unchanged Gaussian-EM simulator, current six-band/layout contract,
target and nuisance priors, thermal covariance and calibration law. Use a fresh
private randomized Sobol scramble, independently generated from the old bank.
Code inspection found that the preserved planner hardcodes scrambled Sobol;
changing `sampling.method` to an IID label would not implement IID draws. This
proposal preserves the executable generator and explicitly does **not** claim
IID draws within the new bank. An IID planner would need a separate validated
implementation change and is not silently substituted here.
Preserve the original bank; create a new separately named artifact only after
explicit approval of its count, disk/compute estimate and generation command.

A human or independent custodian should run generation and escrow the private
seed and protected target/nuisance labels outside the development workspace.
Existing `prepare` draws a fresh secret63-bit seed and ID salt; the config's
example seed is deliberately replaced. No seed is chosen here. Freeze
the final checkpoint, preprocessing, configuration, prediction seed, interval
levels, metrics, source-resampling method and numerical/convergence gates before
releasing inputs. Record hashes and source-group membership; verify no lineage,
source-ID or observation overlap with the original bank. Keep all variants of
each source together. Commit predictions and an evaluation receipt before
unsealing labels. Permit one final evaluation; any subsequent development makes
that evaluation a development archive too.

Prespecify per-target MAE/RMSE in physical units; common prior-coordinate log
scores; 50/80/95% coverage and widths; source-level uncertainty; and an explicitly
costed, source-selected Bayesian subset with the same observed compression,
priors, likelihood and nuisances. Failed convergence stays failed and is counted.
Do not select cases by neural error or Bayesian success. Retain a separate
full-y Bayesian analysis to measure compression loss. Keep the current 24
different-layout challenges separate; this proposed base test would not certify
challenge or real-observation transfer.

At 95% nominal coverage, 256 IID sources would have a rough binomial standard
error of 0.0136 (about ±0.027 for a normal 95% interval); correlated paired
realizations do not double the independent-source count. A 64-source quartile
has much wider uncertainty (roughly ±0.053). These are IID reference planning
approximations, **not validated error bars for a randomized Sobol bank**,
promised precision, multiplicity corrections or a power guarantee. The
protected set is for final evaluation, not to repair insufficient training data.

## Today's decision

A fresh protected bank is **not necessary for today's decision**: the one frozen
thermal128 remedy failed on existing development validation sources and is not
promoted; the canonical checkpoint remains unchanged. Existing Bayesian and
predictive checks can be reported with their scope and failures. This proposal
can remain a later approval-gated step. A new protected evaluation would be
needed before claiming fresh blind final performance after further development;
it would still not establish real-nebula readiness.

## Measured cost and transparent estimate — no new generation benchmark

Measurements below come from preserved October21 receipts, not new runs:

- `candidate_gl4_512_01/protected/last_run.json`:524 parents (512 base,12
  challenge),1048 observations;261.540136s generator wall,133.421875s CPU;
  0.499122s wall/parent;407,806,242 new product bytes. This mixed-profile average
  includes per-source bookkeeping.
- `candidate_generation_external_timing_01.json`:267.159675s whole command,
  or0.509847s/parent;5.619539s outside the generator's timer.
- `benchmark_resource_measurements_01.json`:16 Gaussian parents averaged
  0.513129s per commit interval;4 ellipsoid challenge parents averaged2.216063s.
  Maximum observed commit interval2.399964s. Intervals include writing and
  bookkeeping; they are not isolated renderer timings.
- `candidate_inventory_01.json`:409,685,158B total (about390.706MiB); maximum
  parent product777,494B. Fixed/metadata remainder above product bytes1,878,916B.

For256 Gaussian parents, the central generation estimate is
`256 × 0.5131287724 + 5.6195393 = 136.980505s`, about2.3minutes. Linear storage
estimate is `256 × 778256.1870 + 1878916 = 201112500B`, about191.796MiB
(0.1873GiB). The mixed-bank byte average is conservative for the same two-noise-
variant products; snapshots and instrument products contribute fixed costs.
These are extrapolations from this machine/filesystem/versions, not maxima;
no fresh bank timing was performed.

The existing all-parent numerical audit took298.374735s for524 parents, and
inventory/checksum audit33.297838s. Linear references are145.770863s and
16.267646s for256 parents. Thus generation plus these audits has a central
reference near299.0s (5.0minutes). Reserve10–20minutes for generation, audits,
custody/export/receipts and filesystem variability. Generation alone has the
cooperative600s/0.40GiB cap below. Old audit evidence is dataset-bound and cannot
simply be relabelled as verification of a new bank: a custodian must perform and
bind actual new-bank checks before release.

Neural scoring/model loading, Bayesian inference and final reporting are extra.
Propose eight sources for matched-compressed Bayes, first sorted opaque source
IDs before opening measurements/labels, each first realization; add full-y Bayes
for the first two of those eight to measure compression loss. At up to30minutes
per source-and-data-route,10 routes require a5-hour inference allowance including
failed phases; add about40minutes for diagnostics/predictive checks/reporting.
This is a budget based on today's
up-to12-minute initial phase plus possible10-minute refresh, **not measured
future convergence or a guarantee**. With generation/audits, an end-to-end
planning envelope is about6hours, dominated by inference/evaluation rather than
generation. Budget roughly50–150MiB beyond the bank for retained failed/accepted
chains, predictive receipts, model copy and plots (uncompressed9D state arrays
alone are about4MiB/route across14,000 draws/chain phases); this is a storage
allowance, not a measurement. Total single-copy bank-plus-evaluation allowance
is about242–342MiB; reserve at least0.6GiB for that scope. Custodian release
copies/backups need additional space; the generator's0.40GiB cap only covers its
own bank, not later chains or copies. Final
evaluator code/settings and custody procedure must be frozen before release.
No such work is scheduled or authorized here.

Machine-readable arithmetic/evidence paths: `PROTECTED_COST_ESTIMATE.json`.
The proposed config differs from the preserved one only by `challenge_sources=0`;
all physical/observing/noise/nuisance settings and two realizations are retained.
Original train/validation/test bookkeeping fractions remain, but **the entire
new bank is reserved for final protected evaluation regardless of internal
tags**. No new-bank source may fit preprocessing, select a model or debug it.
The evaluator must iterate all256 base sources, not just internally tagged
`test` rows. This explicit reservation avoids pretending the unchanged CLI has
an all-test or IID switch that it does not have.

## Exact proposed generation command — only after explicit approval

Saved config: `PROPOSED_protected_256_config.json`, SHA256
`77783884b33f4d1553dd360788c17a2e02f9c741f174bd8bcde369c55ff5732d`.
Simulator/core digest to recheck:
`0a31e1c6c436ca14d4def031a53fb76ee159db7c2b209308277ffb7fb45f2e6a`.
The current project-wide manifest digest is
`6b03c1b7e3660622839de62caaf490ffd84e5288bc4a448d168cb8a96e852412`:
later `training_data.py`, `visibility_photometry.py` and a preparation tool were
added, and `validate_casa_run.py` changed. All common original physics files
match the historical manifest. These differences predate this campaign and are
recorded in `GENERATOR_ENTRYPOINT_AUDIT.json`. To make the exact generation
replay unambiguous, the proposed command selects the bank's immutable original
source snapshot explicitly. Its module path and0a31... digest were inspected
without planning or generating a bank. `PROPOSAL_VALIDATION.json` records the
config/argument-parser check, with no output directory created.

The future custodian command below runs from this repository root, with escrow
outside the development workspace. Only schema/digest/parser inspection has
run; the generation command **has not run**:

```powershell
$env:PYTHONPATH='output/training_readiness_20261021/candidate_gl4_512_01/protected/source_snapshot/src'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OPENBLAS_NUM_THREADS='2'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
.venv/Scripts/python.exe -B -c "from hii_sim.io import code_digest; assert code_digest()=='0a31e1c6c436ca14d4def031a53fb76ee159db7c2b209308277ffb7fb45f2e6a'"
if ($LASTEXITCODE -ne 0) { throw 'Frozen source digest mismatch; do not generate' }
.venv/Scripts/python.exe -B -m hii_sim.cli generate `
  --config output/ml_initial_20261023/repair_02/PROPOSED_protected_256_config.json `
  --out LOCAL_PATH_REDACTED `
  --sources 256 --limit 256 --max-seconds 600 --max-gib 0.40 `
  --source-headroom-seconds 5 --source-headroom-mib 3
```

Output must be new; existing bank paths are forbidden. Limits are cooperative
checks at source boundaries, not hard preemption; slow sources/finalization can
overrun, which must be reported as failure/incomplete rather than compliant
completion. The5s/3MiB reservations exceed observed parent commit time/size but
are not guarantees. If incomplete, preserve the run and receipt; seek approval
for continuation rather than silently raising caps. Do not release seed-bearing
provenance, plans, labels or truth to the developer. Verify actual counts,
integrity, source disjointness, frozen config/source hashes and custody before
predicting. No generation or fresh evaluation is needed for today's delivery.
