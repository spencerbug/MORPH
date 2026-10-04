# Research operating instructions

- Work only in `/Users/spencerneilan/workspace/MORPH`. Keep environments, caches, data, artifacts and code here; do not modify global configuration or publish externally.
- Read `STATUS.md`, `SPEC.md`, then the latest entries in `Experiments.md` before work. `SPEC.md` is the active contract; `architecture/capture.md` records the broader intent and unresolved questions.
- Separate article claims, provisional assumptions, implementation facts and measured results. Do not claim an experiment ran without saved commands and artifacts.
- Implement the smallest experiment that can reject a hypothesis. Add one mechanism at a time; do not build the complete architecture ahead of evidence. Register changes to hypotheses, splits, metrics and thresholds before inspecting test results.
- No ground-truth identity, absolute pose, ownership or evaluator masks in learner inputs/losses unless an explicitly labeled diagnostic condition permits it. Maintain identical information and comparable budgets across baselines.
- Predict and commit before revealing target evidence; score before learning. Never credit replay, reconstruction, a recurrent message or a revised prediction as fresh confirming evidence. Preserve UNKNOWN and ambiguity when memory is introduced.
- Declare tensor shapes, frames, units, action direction, composition order, version and symmetry conventions at interfaces. A motor command is not generally measured object motion.
- Keep physical transformations, representation migration, dynamics residuals, new capacity and migration error distinct. Test decoded sensor predictions; algebraic consistency alone is insufficient.
- Freeze evaluation models and test splits. Fit/tune only on training/validation data; use disjoint migration-fit and review anchors later. Preserve failed runs and negative results. Never silently alter acceptance thresholds.
- Verify meaningful contracts: renderer/action semantics, information separation, frozen tickets, model versions and baseline fairness. Record seeds, environment, code/config hashes, commands, metrics and representative failures.
- After substantive work, update `Experiments.md` with decisions/results and `STATUS.md` with current state and exact runnable resume commands. Clearly mark planned commands and unimplemented features. Do not fabricate empty success records.
- Keep this file short. Use `SPEC.md` for active details and `Experiments.md` for history. Do not commit, deploy or start later experiment stages merely because their designs are documented.
- Organize this as a super-repository: experiments own their code/config/tests/reports under `experiments/<id>/`; generated files go under ignored `artifacts/<id>/`; move code to `packages/` only after demonstrated reuse. Put later demonstrations in `demos/`. Preserve completed experiment specifications and reports when activating a new experiment.
