# Active experiment specification

Revision: E0-v1, 2026-10-04. Status: designed, not implemented or run.
Source interpretation: [architecture/capture.md](architecture/capture.md).
Run ledger and later experiments: [Experiments.md](Experiments.md).

## Question and boundary

**E0: Does a prescribed SO(2) latent action improve open-loop image prediction on unseen rotation compositions and unseen objects compared with an unrestricted learned transition?**

This is a test of an imposed compositional inductive bias. It does not test discovery of a group, learned motor semantics, open-ended identity, migration, or uncertainty calibration. The smallest useful system is a renderer, shared encoder/decoder, two transition choices, immutable predictions and an evaluator. No memory, ANN, residual head, learned grouping, active policy, continual updates, pretrained weights or identity classifier.

Hypothesis H0: the structured model reduces mean foreground-union pixel MSE by at least 10% relative to the unrestricted model on unseen-object, eight-step monotone rollouts, without materially degrading one-step performance. Exact algebra alone does not support H0.

## Observations, world and actions

One rigid planar object, fully visible, centered, fixed size and illumination, on a black background. Observation x is float32 [B,1,32,32], intensity in [0,1]. No observation history (L=0). One instance per episode supplies temporal association without identity labels. The learner receives pixels and commanded increments only. No masks, shape parameters, absolute angle, symmetry label or simulator IDs enter model inputs or losses.

Coordinates: pixel-center x=(column−15.5)/16; y=(15.5−row)/16. The camera is fixed. Positive α rotates the object counterclockwise about (0,0). Action float32 [B,1] is a displacement in radians over one step, not a velocity. Execution equals command; no slip, delay, noise, translation or independent motion. Future frames are rendered from the underlying shape at the accumulated angle, never by repeatedly resampling the previous image.

Proposed deterministic renderer: union of three ellipses on a 128×128 supersampled canvas, average-pool 4×4 to 32×32. Sample centers independently in [−0.25,0.25]^2, semiaxes in [0.08,0.22], ellipse orientations in [0,2π), and intensities in [0.4,1]. Pixel intensity is the maximum over covering ellipses, else zero. All shapes remain within the image during rotation. Sample a canonical shape once; rotate all centers and ellipse orientations together. Generator seed and parameter records belong to the evaluator. Reject empty or effectively circular shapes: evaluation-only image difference between canonical and 45° view must exceed 0.002 whole-image MSE. This is dataset construction, not a learner input. Log rejection counts and rendered examples.

This synthetic distribution has no unseen surfaces; success is not evidence of 3-D completion. Add separate diagnostic objects made by rotating one sampled offset ellipse into 2 or 4 identical copies with identical intensity; do not subject these to the asymmetry rejection rule. Exact continuous raster symmetry is approximate, so measure renderer error before interpreting equivariance error.

## Dataset and separation

Use independent RNG streams: geometry seed 100, training trajectories 200, validation 300, test 400, model seeds 0–4. Split 96 canonical asymmetric objects into 64 train, 16 validation, 16 test **before** rendering. Never split adjacent frames randomly between train and test. Geometry, not just trajectory, is disjoint across these splits; publish parameter hashes and audit exact duplicates. Validation objects may select hyperparameters, test objects may not.

Training: 128 independently initialized two-transition episodes per training object (8,192 total), θ0 uniform [0,2π); each increment independently chosen from {−15°,−5°,0°,5°,15°}, converted once to radians. Each episode is [x0,x1,x2] plus [a0,a1]. All absolute orientations occur in training; the generalization target is longer temporal composition and new geometry, not unseen global angle. Two-step training is permitted equally for both learned models.

Validation and test each have 32 initial-angle draws per object and each schedule below. Keep initial angles paired across methods and schedules. Report seen-object held-out trajectories separately using the 64 training geometries and independent episode draws. Diagnostic symmetry objects: 8 of each order, disjoint seeds 500 and 600, same evaluation schedules; never used for model selection.

Evaluation schedules:

1. One-step: each of the five training increments; interpolation diagnostics at ±10° and extrapolation diagnostics at ±30°, reported separately.
2. Primary: eight successive +15° or eight successive −15° increments, equal weight. Intermediate observations are withheld. Evaluate horizons 1,2,4,8.
3. Mixed: eight increments drawn independently from the training action set, sampled once in the test manifest.
4. Inverse: [+15°, +5°, −5°, −15°]. Final state should return to initial image up to rendering tolerance.

Known action sequences are supplied at rollout start to both models. Only x0 is encoded during rollout; no teacher forcing or target encoding in the prediction path. Training targets may be encoded for losses after predictions are formed. Test targets stay outside the predictor interface.

## Tensor and model interfaces

- `encode(x: [B,1,32,32]) -> (u: [B,32], p: [B,16])`
- `step(u, p, action: [B,1]) -> (u_next, p_next)`
- `decode(u, p) -> x_hat: [B,1,32,32]`
- `rollout(x0, actions: [B,T,1]) -> predictions: [B,T,1,32,32]`
- `commit(input_ids, action_sequence, model_hash, predictions, scoring_rule) -> ticket`
- `score(ticket, target_events) -> immutable metrics_record`

Encoder: three Conv2d layers, channels 1→16→32→64, kernel 4, stride 2, padding 1, each followed by ReLU; flatten 64×4×4; linear 1024→48. First 32 outputs are u, remaining 16 are p. No batch normalization, dropout, coordinate channels or skip connections. Decoder: linear 48→1024 + ReLU; reshape 64×4×4; transposed convolutions 64→32→16→1, kernel 4, stride 2, padding 1; ReLU between layers, sigmoid output. No direct input-image or action shortcut to decoder.

Structured S: p reshapes [B,4,2,2]. For k=1,2,3,4 and each of two copies, apply R(kα) to the final 2-vector; u_next=u. R(β)=[[cosβ,−sinβ],[sinβ,cosβ]]. C(u,p) is concatenation; joint action diag(I32,ρp). No learned action parameters. Shapes flatten in harmonic/copy/component order. Use column-vector mathematics and test row-batch transpose convention explicitly.

p is a 16-dimensional pose-bearing code, not an angle. Do not unit-normalize each block: block amplitudes may vanish for symmetry. No absolute-pose supervision. Global latent phases are gauge freedom. Identity leakage into block amplitudes is possible; this experiment does not establish complete disentanglement. u is encouraged to be invariant but is not an individual-ID label.

Unrestricted U: same encoder/decoder and u_next=u; p_next=p+MLP([u,p,α]), with dimensions 49→64→16 and ReLU hidden activation. This baseline retains the same latent size and identity assumption but has extra transition parameters. Record exact parameter counts, FLOPs estimate and wall time; do not call it parameter-matched. It is a conservative capacity comparison because U has more learned parameters. A parameter-budget-matched comparison, if needed, is a separately registered follow-up, not a retroactive alteration of E0.

Persistence P: predict x0 at every horizon, no training. Known pixel warp W: rotate x0 using known actions and the declared center with bilinear sampling; compute each horizon from x0 using accumulated angle. W uses explicit image geometry and is an engineering control, not a learned baseline or guaranteed oracle. A renderer-truth replay control verifies evaluation bookkeeping but is never a competing model.

## Training and prediction discipline

For each training triple encode each observed frame zt=[ut;pt]. Starting only from z0, roll step twice to obtain zhat1, zhat2. Decoder predicts xhat1,xhat2. Losses (each MSE averages over all its elements):

- Lrec = mean over t=0,1,2 of MSE(D(zt),xt).
- Lfuture = mean over h=1,2 of MSE(D(zhath),xh).
- Llatent = mean over h=1,2 of MSE(zhath,stopgrad(zh)).
- Linvariant = mean over h=1,2 of MSE(u0,uh).
- L = Lrec + Lfuture + 0.1 Llatent + 0.1 Linvariant.

All image losses use the complete image; no evaluator masks in training. Decoder reconstruction gradients reach each frame's encoder; latent targets are detached only in Llatent. Both S and U use identical losses, data order, batch size and optimization budget. Losses permit bad minima: inspect reconstruction, latent variance and action intervention results rather than claiming noncollapse from the objective alone.

Defaults: Adam, learning rate 0.001, betas (0.9,0.999), no weight decay, batch size 64, 50 epochs, gradient norm clipped at 5. Pair initialization seeds for shared modules between S and U. Select checkpoint by lowest validation eight-step foreground-union MSE, ties by earliest epoch. Evaluation masks can select validation checkpoints but cannot enter training gradients. No test-driven tuning. Record library versions, device, determinism settings and actual optimizer steps.

Run a one-seed smoke test first; then a three-value learning-rate pilot {0.0003,0.001,0.003} for each method on validation only, same budget. Select the learning rate by validation score for seed 0 and freeze it before the five-seed comparison (seed 0 may reuse its completed pilot fit). Record all pilot failures. These pilot choices are part of the reported compute cost.

Evaluation bundles are frozen. Commit predictions and model hash before giving targets to scorer. Only score afterwards; no evaluation-time fitting. Deterministic rendering means future targets could theoretically be regenerated from private state: enforce API separation so the learner never receives that state. A ticket establishes chronology, not statistical independence or identity confidence.

## Metrics, checks and decision rules

Primary: foreground-union MSE at horizon 8 on unseen test objects under the two monotone schedules. Foreground union uses evaluator masks of initial and target object support (nonzero supersampled occupancy); average squared pixel error on this union. Using both masks penalizes a predicted object left at its original location. Report full-image MSE as well to expose off-mask artifacts. Aggregate equally per schedule, then per object, then per model seed. Never treat pixels or correlated horizons as independent replicates.

Secondary: one-step and horizons 2/4 sensor errors, seen/new-object gap, reconstruction MSE, symmetry-set error, action-shuffled error, identity within-episode variance, p variance and group composition/inverse error. Report absolute error and relative gain (MSE_U−MSE_S)/MSE_U; if MSE_U<1e−8, relative gain is undefined and H0's gain gate cannot pass.

Compose test compares two sequential latent steps with one summed increment on the same z. Inverse test applies α then −α. For S these are implementation checks at tolerance 1e−5 maximum absolute error in float32 on bounded synthetic codes, not evidence of learning. For U report the same errors diagnostically without forcing it to satisfy that tolerance. Swap action sign at prediction time on asymmetric examples: if changing actions has negligible effect, inspect decoder bypass or pose collapse.

Compute paired 95% bootstrap intervals for primary relative gain by resampling model seeds and test-object IDs (2,000 draws, RNG seed 700); preserve paired method outputs and episode groups. Report per-seed results, dispersion and the interval. Five seeds support a preliminary decision only. Save arrays so intervals are reproducible.

Implementation acceptance before scientific interpretation:

- Positive 90° sends a right-of-center renderer landmark upward; identity, inverse and composition controls pass. Direct rendering versus warp raster error documented.
- Inputs, outputs and action units match signatures; split hashes and evaluator-only fields audited.
- Predictions are insensitive to changing withheld target payloads; frozen-ticket mutation and duplicate-score attempts fail or return the original score idempotently.
- S and U receive the same frames/actions and budgets; no hidden true pose/ID/mask inputs.
- Training produces finite outputs; report actual reconstruction and variance, not just declining aggregate loss.

Provisional scientific go/no-go rule (fixed before inspecting test results): S must have at least 10% primary relative gain over U, paired interval lower bound above zero, and positive gain in at least 4 of 5 seeds. S must beat persistence on primary MSE and have one-step MSE no worse than 1.05×U + 0.0001. Report full-image error; a gain confined to the masked metric with worse full-image error requires investigation and does not justify expansion.

A validated run missing any scientific gate is **negative for E0's registered hypothesis**. No advantage, collapse, action insensitivity, gains only on familiar objects, or exact latent algebra with poor decoded predictions all weaken this recipe. Leakage, broken rendering, or unequal inputs invalidate the comparison rather than refute H0. If both learned models fail reconstruction, diagnose capacity/optimization before claiming a group-theory conclusion. Strong W performance reveals how easy known image geometry makes this task; S need not beat W, and matching W does not demonstrate identity learning.

## Deliverables and stop condition

Next implementation should create a local package, explicit config files, renderer/data manifests, tests for the contracts above, training/evaluation CLI and run artifacts under `experiments/runs/<run_id>/`. Each run stores config, source revision/diff, environment, seed, checkpoint hash, split hashes, tickets, predictions, metrics, learning curves and representative target/prediction/error images. Choose and document exact CLI commands when implemented; none exists yet.

Stop after E0 results and a recorded decision. Add at most one mechanism in the next experiment. No automatic full MORPH implementation or long compute sweep is implied by this specification.
