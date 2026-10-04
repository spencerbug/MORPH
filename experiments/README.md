# Experiment registry

| ID | Question | Contract | State |
|---|---|---|---|
| E0 — [e0_rotation](e0_rotation/README.md) | Does known SO(2) latent structure improve held-out image prediction? | [Frozen E0-v1.1](e0_rotation/SPEC.md) | Completed; [report](e0_rotation/reports/REPORT.md), numerical pass with baseline-quality qualification |

Each experiment owns `SPEC.md`, `configs/`, `src/`, `tests/` and `reports/`. Large generated files go to `artifacts/<experiment-id>/` at the repository root. Keep a completed experiment reproducible when introducing its successor; do not repurpose E0's directory for E1. The root SPEC.md points to the current active contract, while Experiments.md records decisions across experiments.

Share a component through `packages/` only when reuse is demonstrated. A larger experiment may consume several packages; a demonstration belongs in `demos/` and names the exact validated checkpoints it presents. Experiment success is not automatically demonstration readiness.
