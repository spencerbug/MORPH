# Current state

Updated 2026-10-04. **E0 is complete.** Implementation, 14 full fits, five-seed held-out evaluation, integrity audit, figures and report are finished. No training process remains active. No later experiment has started.

Read [the report](experiments/e0_rotation/reports/REPORT.md). The structured recipe passed all registered numerical gates: 54.8% lower unseen eight-step error, paired 95% interval [35.8%,70.5%], wins in 5/5 seeds. Four unrestricted checkpoints were effectively action-insensitive, so this is not clean evidence of compositional superiority between equally trained models. The proposed next step is an E0b baseline-optimization check before E1; it requires a prospective experiment contract and fresh audit data.

## Repository organization

- `experiments/e0_rotation/`: frozen spec, config, source, six contract tests and compact report artifacts.
- `artifacts/e0_rotation/`: approximately 9.5 GB of generated data, checkpoints, predictions, source/environment records and logs; ignored by Git and preserved locally.
- `packages/`, `demos/`: documented reserved locations; no unnecessary shared framework or demo implemented.
- `.venv/`, `.cache/`, `.tools/`: project-local environment and caches. `requirements-lock.txt` pins dependencies.

The E0 implementation and report are included in this local commit at the user’s request. Nothing has been published. The original architecture capture remains unchanged.

## Exact commands to inspect and reproduce analysis

```sh
cd /Users/spencerneilan/workspace/MORPH
cat AGENTS.md
cat experiments/e0_rotation/reports/REPORT.md
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m pytest -q
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m e0_rotation.audit
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m e0_rotation.analyze
```

Completed-fit inventory: `artifacts/e0_rotation/campaign.json`. Validation choice: `artifacts/e0_rotation/selection.json`. Machine-readable outcome: `experiments/e0_rotation/reports/summary.json`. Logs: `artifacts/e0_rotation/{prepare,smoke,campaign,evaluate,audit,analysis,tests}.log`.

These commands reuse the completed fits/scores. Training also supports resuming an incomplete fit at its last saved epoch:

```sh
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m e0_rotation.run campaign --device mps
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m e0_rotation.run evaluate --device mps
```

Do not regenerate the existing dataset or overwrite completed experiments. A new scientific intervention belongs in a new experiment directory and artifact namespace with a registered contract. Environment recreation commands are in README.md.

## Verification

Six contract tests passed. Audit checked 133 dataset-file hashes, 96 distinct object definitions, ten selected fit budgets/checkpoints and 2,048 committed prediction/score pairs. Report figures were visually inspected; local Markdown links and source compilation passed. Scientific uncertainty and optimization failures are retained in the report and ledger.

## Latest documentation change

The E0 report now explains the one-image-to-rotated-images task before presenting results. It explicitly defines unseen objects as shapes excluded from training across all views, distinguishes pixel error from recognition, and identifies the rotation structure supplied by the experiment. This documentation-only revision did not rerun or alter E0.
