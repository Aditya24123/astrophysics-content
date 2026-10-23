# H II thermal free–free phase-one simulator

This repository generates numerical sky brightness and synthetic radio
measurements for a deliberately simple ionized-hydrogen source family. It does
not contain a learned inverse model, an estimator, or real-object inference.
Read [the physics specification](docs/MODEL.md) and [data contract](docs/DATA_CONTRACT.md)
before using the outputs as training data.

**October 22 initial data entry point:** the fresh 524-parent/1,048-observation
candidate has a [training batch starter](docs/TRAINING_STARTER.md) with train-only
normalization, split-safe original targets and public-only test loading. The
versioned preparation report is `output/training_ready_20261022/initial_02/readiness_report.json`.
This is an initial declared-proxy experiment, not real-data or trained-model
certification. No training is started by the preparation or example commands.

**Audit revision 0.2.0:** the original production bank has confirmed label
leakage and scientific limitations. It is preserved, but the predictor loader
rejects its schema by default. Read [the audit](docs/AUDIT_V2.md),
[instrument scope](docs/INSTRUMENT_AUDIT.md), and
[measurement-only reconstruction](docs/RECONSTRUCTION.md). No training or
new bulk generation was performed for this repair.

The separate CASA path supports [shared coarse measurement-only reconstruction](docs/CASA_RECONSTRUCTION_PATH.md)
for existing Measurement Sets and explicitly selected future simulation runs.
Its 3x MAD multiplier is selected among the bounded development comparisons;
one actual saved-case execution has exact parity. See the
[milestone assessment](docs/CASA_TERMINAL_MILESTONE.md) for the remaining
acceptance step and scientific limits. Earlier outputs are preserved.

## Install and run

Use Python 3.13 with the pinned dependencies in `requirements.txt`:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pip install -r requirements-dev.txt
$env:PYTHONPATH = 'src'
.venv\Scripts\python -m pytest -q --basetemp output/test_rerun_01
.venv\Scripts\python -m hii_sim.audit --config configs/compact_proxy_v2.json --out output/audit_rerun_01 --widefield
```

`hii-sim generate`, `hii-sim plan`, `hii-sim validate`, and `hii_sim.audit`
default to the guarded `configs/compact_proxy_v2.json` scope. Pass
`--config configs/base.json` explicitly only when reproducing the legacy
instrument behavior; that configuration has no short-spacing guard and uses
128 MHz windows.

`generate` creates independent base sources plus the configured challenge
sources. `--limit` bounds each invocation; rerun the exact command to resume.
The output directory records the configuration and source plan, refuses a
changed configuration/code/runtime on resume, commits one physical source at
a time, and rebuilds the public manifest from committed records. Keep the
`protected/` directory private. The `public/` subtree is the only predictor
input surface.

The first `generate` invocation for a dataset must have room for its output.
The bounded 16,384-source production dataset was generated before this audit.
Its old code/provenance cannot be resumed using the revised code; the strict
guard is retained. Use a new destination for any later authorized generation.
[Historical validation](docs/VALIDATION.md) retains the old sizing measurements,
not a certification of the old dataset.

## Two instrument paths

The default is a fast *gridded Fourier development approximation*: a synthetic
Hermitian uv mask, Gaussian taper, primary-beam envelope, correlated complex
noise, shared flux-scale/spectral calibration errors, and a unit-peak dirty
image. It is useful for bounded experiments and is explicitly not CASA, a
VLA observation simulator, or a complete interferometer model.

For an actual measurement-set path, run `prepare-casa` on a protected truth
FITS cube, then run `src/hii_sim/casa_simulate.py` inside CASA 6.7.4. The bridge
uses the named built-in `vla.c.cfg`, 30 min on source, 10 s integrations, a
transit pointing, one 128 MHz continuum window per modeled frequency,
`simobserve` with `tsys-atm`, and `tclean` natural-weighted dirty imaging
(`niter=0`, no primary-beam correction). The Measurement Sets, dirty images,
PSFs, PB maps, and residuals are numerical products. The bridge has not been
run in the current Windows environment: CASA is absent. See the exact
integration command and limitations in [the CASA notes](docs/CASA.md).

## Deliverables currently in the repository

- Physics, source geometry, fast measurement, reconstruction, data I/O,
  predictor/training loaders, bounded dataset generation, and CLI modules in
  `src/hii_sim/`.
- The guarded compact scope in `configs/compact_proxy_v2.json`; the preserved
  `configs/base.json` is for explicit legacy replay.
- Versioned equations, assumptions, and survey evidence in `docs/MODEL.md`.
- Independent numerical validation, reproducible pilot and production
  distribution plots, and pilot/production measurements in `output/validation/`.
- Completed 16,396-source, 32,792-observation dataset in `output/production/`;
  its public subset order exposed generating parameters. Legacy shared
  instrument products had no original checksums. See the audit before reuse.
- Original `output/workflow.md` retained as user-provided context; its generic
  density/magnetic/motion targets should be read as later work, not claims of
  this simulator.
