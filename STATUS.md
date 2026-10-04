# Current state

Updated 2026-10-04. Phases 1–2 (architecture capture and operating/spec documents) complete. Phase 3's initial predictive experiment and conditional sequence are designed. Implementation and scientific evaluation have not started.

Present: architecture/capture.md, a local source snapshot, AGENTS.md, SPEC.md, Experiments.md and this status file. E0 is the only active experiment. No package, dependencies, environment, dataset, training CLI, checkpoints or empirical results exist. Nothing has been committed or published by this task.

## Resume now

These commands are runnable now and only inspect this local project:

```sh
cd /Users/spencerneilan/workspace/MORPH
cat AGENTS.md
cat SPEC.md
cat Experiments.md
git status --short
```

Next work: implement E0's renderer, split manifests, encoder/decoder, S/U transitions and evaluator with the explicit information boundary. First verify action direction, shapes, composition, inverse and frozen prediction behavior. Then run the one-seed smoke test and validation pilot before the five-seed comparison. Record the selected environment and exact training/evaluation commands here once those entry points exist.

There is no truthful training-resume command yet. Do not treat a proposed CLI as implemented. Place code, virtual environment, dependency caches, datasets and run artifacts under this repository; no external project changes.

## Known limitations / pending decisions

The numeric architecture, dataset and thresholds in SPEC.md are provisional choices, not extracted claims. E0 supplies a centered view and exact action execution; it tests known SO(2) prediction only. Compute feasibility, capacity and optimization behavior remain unmeasured. Later experiment designs must become explicit SPEC revisions before execution. No full MORPH claim is currently supported by local results.

## Documentation verification

Checked all local Markdown links and fenced blocks, source snapshot hash, presence of all 18 numbered article sections in the extracted text, and trailing whitespace in authored documents. Checks passed. These are documentation checks, not model tests or experimental results.
