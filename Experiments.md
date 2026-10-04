# Experiment ledger and staged plan

E0 has now been executed; see the completed run record below. Earlier design entries are historical and their proposed values are not results. Active contract: [SPEC.md](SPEC.md). Broader architecture: [capture](architecture/capture.md).

## 2026-10-04 — Design record D0

Repository initially contained empty AGENTS.md and README.md and an empty experiments directory. Read the complete research post and retained its source revision locally. Created architecture capture and an E0 specification. Used the existing uppercase `AGENTS.md` as the operating-instruction file; do not create a conflicting case-only `agents.md` on macOS.

Decision: begin before the article's multi-mechanism prototype with controlled SO(2) prediction. This removes correspondence, occlusion, unknown centers, independent motion, memory and migration as confounds. It also gives the system exact action semantics and a centered view, so findings must remain limited to that favorable setting.

Decision: use a 16-dimensional harmonic pose-bearing code, not a supervised scalar angle, to leave room for symmetric objects. Identity descriptor is 32-dimensional; neither size is established by the article. Representation capacity is a revisable assumption, with revisions recorded before rerunning a fresh evaluation.

Decision: E0 uses known group operators. Learned generators and motor-to-motion mappings are deferred. No pose or identity training labels. No confirmation-confidence claims from pixel MSE.

## E0 — Prescribed rotation and predictive transfer (original design)

Hypothesis, exact configuration, interfaces, splits and acceptance criteria are in SPEC.md; do not duplicate mutable numeric settings here. Observations are one complete grayscale image; action is an exact angular displacement. Compare structured S, unrestricted U, persistence P and known pixel warp W. Train on two-step sequences and evaluate longer compositions and new geometries without intermediate observations.

Metrics: primary unseen-object eight-step foreground-union pixel MSE; full-image and one-step MSE; action sensitivity; collapse diagnostics; symmetry performance; paired intervals over seeds/objects. Negative result: failure of the preregistered predictive gain gate after integrity checks, even if group composition is mathematically exact.

Configuration file: not created. Dataset: not generated. Commands: not implemented. Results: not run. Failures: none observed experimentally. Next decision: implement E0 and its information/coordinate controls, then pilot on validation only. Do not add memory to rescue a failed predictive comparison.

## E1 — Translation plus rotation (conditional; not active)

Question: does the E0 advantage survive motion whose order matters and whose center is no longer fixed? Hypothesis: a declared SE(2) action model predicts held-out rotate/translate orderings better than equally informed generic dynamics.

Observe a single fully visible object's whole image and commanded rigid transform. Supply exact execution and a calibrated image frame; allow varying centers but keep objects in bounds. Do not supply absolute object pose. This isolates noncommuting composition before multi-object grouping. Baselines: E0-style unrestricted transition and known pixel warp; same data, tuning budget and decoder policy. Compare translation→rotation with rotation→translation, inverse and long-rollout sensor error, broken down by center and displacement. Measure reconstruction and cropping separately.

Negative result: structured advantage vanishes or predictions require hidden center/pose truth; exact matrix composition without sensor accuracy is insufficient. Before activation specify latent representation, transform frame/order, shapes, dataset sizes and numeric gates in a new SPEC revision. SO(2)'s commutativity cannot answer this experiment.

## E2 — Fast memory with a frozen predictive substrate (conditional; not active)

Question: can future predictions distinguish candidate identities better than static matching without changing the encoder? Observe single-object episodes with re-entry, new objects and deliberately similar appearances; no ID labels. Use passive predetermined actions first. Assume full visibility and known action execution, retaining E1's proven regime. Memory allocates provisional records from prior evidence, performs exact nearest-neighbor retrieval and keeps UNKNOWN. No ANN scaling or encoder plasticity yet.

Hypothesis: future sensor challenges reduce false merges at a fixed false-split/unknown rate compared with the same frozen representation and nearest-prototype matching. Compare static prototype, frozen prediction with legal future evidence, and an explicitly labeled same-evidence confirmation control. All methods receive the same stream and memory budget. Evaluation uses hidden physical IDs to measure merges, splits, re-entry, provisional promotion accuracy, abstention and latency; thresholds tuned on validation. Track evidence IDs to audit repeated credit.

Negative result: no identity benefit at matched coverage, memory explosion, or confidence increase without new evidence. Identical-looking instances with identical action consequences are intentionally unidentifiable; correct behavior is ambiguity. Before activation define the null likelihood or use nonprobabilistic thresholds without calling them calibrated posterior odds; define promotion/merge lifecycle and numeric gates. Active action selection is a subsequent paired random-policy comparison, not bundled into this test.

## Later gates — not implementation commitments

Only after the predictive and frozen-memory controls work:

1. Add patch crossing and then multiple objects, comparing inferred grouping to diagnostic oracle grouping. Hidden centers and independent motion expose false coherent-motion assumptions.
2. Add candidate shared learning with replay, then explicit migration. Compare frozen substrate, replay-only updates, full-gallery re-encoding and Q transport on identical streams. Require new utility plus historical identity and decoded dynamics retention; report backfill, missing features, compute and repeated-version drift. Fit Q and review it on disjoint anchors.
3. Test temporal escrow under recurrent mistaken associations and plasticity, beyond E2's frozen case. Hold observations/compute fixed so improvement is not merely less training. Report persistent false accepts and calibration, including correlated evidence and duplicate-event stress tests.
4. Introduce a restricted residual only for a specified non-group failure. Compare group-only, residual-only and combined paths using the staged frozen-group schedule; measure takeover.

Hierarchy, language, contact, planning and unrestricted 3-D scenes stay outside this plan until prerequisite evidence exists. These gates are a roadmap, not fully specified active experiments.

## Required entry for each actual run

Create an append-only entry with:

- Run ID, date, status (pilot/completed/failed/invalid), hypothesis/spec revision and reason for run.
- Exact command, project-local working directory, code revision plus dirty diff if needed, config and dataset hashes, environment/device, seed and compute budget.
- Information available to each method, any diagnostics with extra supervision, split manifests and checkpoint-selection rule.
- Artifact paths, paired baseline metrics with uncertainty, qualitative failures and integrity-check outcomes.
- Interpretation: supported, negative or invalid against the registered gate; limits of that conclusion.
- Next decision and any prospective spec change. Preserve crashes, abandoned runs and adverse results; do not replace them with the best run.

## 2026-10-04 — Implementation record D1 (before model results)

User authorized implementing/running E0 and reporting results, with a super-repository organization. Registered E0-v1.1 clarifications in SPEC.md and config before training. Code is local to `experiments/e0_rotation`; generated artifacts are ignored under `artifacts/e0_rotation`. Local Python 3.12 environment and a dependency lock were created. No hypothesis, epoch budget, loss, split size or numerical scientific gate changed.

## 2026-10-04 — E0 completed (E0-v1.1)

Report: [experiments/e0_rotation/reports/REPORT.md](experiments/e0_rotation/reports/REPORT.md). Frozen contract: [experiments/e0_rotation/SPEC.md](experiments/e0_rotation/SPEC.md). All work used the local repository and MPS device; environment is pinned in requirements-lock.txt.

Commands from the repository root (each prefixed with `PYTHONPATH=experiments/e0_rotation/src .venv/bin/python`): `-m e0_rotation.run prepare`, `-m pytest -q`, `-m e0_rotation.run smoke --device mps`, `-m e0_rotation.run campaign --device mps`, `-m e0_rotation.run evaluate --device mps`, `-m e0_rotation.audit`, `-m e0_rotation.analyze`. Logs live under `artifacts/e0_rotation/`. Each fit contains exact config, source snapshot, environment, checkpoint hashes and epoch history. The experiment README provides copyable commands.

Executed: two smoke fits, six seed-zero learning-rate pilots, and eight additional fits completing five paired seeds (14 full fits, 700 epochs, 89,600 steps; 783.5 seconds summed fit time). Validation selected S=0.001 and U=0.0003. All fits completed; no runs were dropped. Three pilot fits gave near-black predictions. Several selected-seed fits showed delayed or unstable optimization.

Results: unseen H8 foreground-union MSE S=0.048574, U=0.107351, P=0.288143, W=0.003700. Relative reduction 54.8%, paired 95% bootstrap interval [35.8%,70.5%], wins in 5/5 seeds. All registered numerical gates pass, including one-step and full-image checks. Six contract tests and the artifact audit passed (133 files, 2,048 ticket/score pairs).

Interpretation: positive for the registered recipe comparison, qualified as mechanistic evidence. U seeds 1–4 are effectively action-insensitive and reconstruct poorly. The known pixel warp is substantially stronger than either learned model. Do not claim clean compositional generalization over an equally trained baseline, or any support yet for MORPH memory/migration.

Next decision: propose E0b to improve/check baseline optimization before E1. A single possible intervention is zero-initializing U's residual output layer; register it and fresh audit data before execution. E0b has not been implemented or run. Preserve E0 without changing its criteria or excluding failed seeds.
