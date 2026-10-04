# E0: controlled rotation prediction

This experiment owns its implementation, configuration, tests and report. The frozen experiment contract is [`SPEC.md`](SPEC.md); the root [`../../SPEC.md`](../../SPEC.md) tracks the active experiment. No other experiment should depend on this module; promote genuinely shared components into `packages/` when a second use case needs them.

Run from the repository root. All commands use the local environment; no package installation is needed to import experiment code:

```sh
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m pytest -q
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m e0_rotation.run prepare
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m e0_rotation.run smoke --device mps
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m e0_rotation.run campaign --device mps
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m e0_rotation.run evaluate --device mps
```

`prepare` refuses to replace an existing dataset. `campaign` reuses completed runs and resumes incomplete training from the last epoch checkpoint. `evaluate` skips completed score records. Never delete an adverse run to improve the reported result. Make a new configuration and artifact namespace for a new experiment revision.

Generated data and run artifacts are under `../../artifacts/e0_rotation/` (ignored by Git):

- `data/`: learner inputs, evaluator-only targets/geometry, and hashed manifest.
- `runs/`: per-method/seed/rate checkpoints, histories, validation tickets, environment and exact source snapshots.
- `selection.json`: validation-only pilot decision, frozen before test evaluation.
- `campaign.json`: selected five-seed run inventory.
- `evaluation/`: immutable predictions, tickets and per-object errors for all schedules.

Reports and compact summary tables belong in `reports/` and can be versioned; large regenerable files do not. Tickets enforce hashes and ordering inside this experiment runner; they are an audit mechanism, not an adversarial security boundary.

After evaluation finishes:

```sh
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m e0_rotation.audit
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m e0_rotation.analyze
```

The audit verifies dataset hashes, input schema, selected budgets/checkpoints, and prediction/score ordering. Analysis writes compact tables, paired bootstrap arrays and figures into `reports/`. Report generation does not select or retrain models.
