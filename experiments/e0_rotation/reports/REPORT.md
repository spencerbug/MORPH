# E0 report: predicting rotated images from one observed view

2026-10-04 · Completed locally · Contract: [E0-v1.1](../SPEC.md)

## What this experiment asks

**Given one image of an object and a sequence of rotation commands, can the model predict what the object will look like after those rotations?**

For example, the model receives one grayscale image and the instruction “rotate counterclockwise by 15° eight times.” It must predict the resulting images, including the final view after 120°, without seeing any intermediate images. The evaluator then compares its predictions with images rendered from the actual object.

This is **pixel prediction**. There is no object-name output, identity lookup, recognition score or persistent object memory in E0.

### What the model learns—and what we supply

The learned path is:

```text
One 32×32 image → encoder → 48-number latent representation
                                      ↓ rotation commands
                           updated latent representation
                                      ↓ decoder
                              predicted rotated images
```

The encoder learns to turn raw pixels into a representation; the decoder learns to turn that representation back into pixels. In the structured model, we supply the mathematical rule for how rotation changes 16 of those latent coordinates. The other 32 coordinates are encouraged to retain information that stays stable across rotation. “SO(2)” is the mathematical name for this family of 2-D rotations.

The model therefore learns a representation useful for predicting rotated views. We have **not** established that it learns an explicit geometric model, that its latent coordinates correspond to physical points, or that its stable coordinates uniquely identify an object. In particular, the rotation rule is supplied rather than discovered. The shapes rotate within the image plane; no hidden back side or 3-D surface needs to be inferred.

### What “seen,” “unseen,” and “MSE” mean

| Term in this report | Meaning |
|---|---|
| **Unseen object** | An object's shape was excluded from training in every view. At evaluation, the model receives one initial image of that new object and predicts its rotated views. |
| **Seen object** | Its shape appeared during training. Evaluation uses fresh starting angles and action sequences, still requiring pixel prediction. |
| **Unseen view** | Not the split used to define the main result. Training angles span the full circle; we did not reserve an orientation range as the main test. |
| **H8 / eight-step prediction** | The predicted image after eight rotation commands, computed from the initial image without intermediate observations. |
| **MSE** | Mean squared error: square each difference between a predicted pixel and its target pixel, then average. Pixels range from 0 to 1; zero error means an exact match. Lower is better. It is not recognition accuracy. |
| **Foreground-union MSE** | Pixel error averaged over locations occupied by the object in either the initial image or the target image. This prevents the large black background from dominating the score. The evaluator supplies these regions only for scoring. |
| **Full-image MSE** | Pixel error averaged over the entire image, including background. This also catches incorrect predictions outside the scored foreground region. |

Thus, **“unseen-object MSE” means how accurately the model predicts rotated pixels for a shape it never trained on**, after receiving one image of that shape. It does not mean how well the model recognizes a familiar object from an unfamiliar view.

### What the comparison tests

| Method | How it predicts the next image |
|---|---|
| **Structured S** | Learns the encoder/decoder; uses a supplied rotation rule to update the latent representation. |
| **Unrestricted U** | Learns the same encoder/decoder design and a neural network that updates the latent representation from the rotation command. |
| **Persistence P** | Repeats the initial image, ignoring rotation. |
| **Known pixel warp W** | Rotates the initial pixels directly using the known angle and pivot, with no learned latent representation. |

The question is whether supplying rotation structure helps the learned predictor under the same training budget. The pixel-warp control shows how well a direct geometric solution can do in this deliberately simple setting.

## Outcome and decision

**E0 met all numerical acceptance criteria fixed before evaluation.** For objects excluded from training, the structured model made **54.8% less squared pixel error** after eight rotation steps than the unrestricted model: **0.048574 versus 0.107351**. The paired 95% bootstrap interval for relative reduction is **35.8%–70.5%**. Structured prediction won in **5/5 paired model seeds** (five training runs with different random initializations). This is an error reduction, not a 54.8% recognition rate.

**The result does not yet establish why the structured model wins.** Four of the five selected unrestricted checkpoints are effectively insensitive to action sign and reconstruct substantially worse than the structured models. This comparison demonstrates an advantage for the specified structured training recipe under the fixed budget. We therefore cannot yet separate a benefit from combining rotations correctly over many steps from a benefit in simply getting the model to learn successfully. The known image-warp control is much more accurate than either learned system.

Decision: preserve E0 as a completed positive result with a baseline-quality qualification. **Do a baseline optimization check before adding translation, memory, grouping or migration.** Do not remove unfavorable seeds or change E0's outcome retrospectively.

## What was executed

A single fully visible planar object rotates about a known image-center pivot. Each object is a union of three ellipses. The learner observes one 32×32 grayscale image and signed angular increments; it never receives absolute pose, geometry, object IDs or evaluator masks. Shapes are rendered directly from their canonical geometry at 128×128 and downsampled, avoiding cumulative image-resampling error in targets.

The split contains 64 training, 16 validation and 16 unseen test objects, with separate 2-fold and 4-fold symmetric diagnostic objects. There are 8,192 two-transition training episodes. Evaluation uses 32 initial angles per object per schedule, with identical inputs/actions across methods. Absolute orientations are already covered during training: this is a test of new shapes and longer action sequences, not a claim of unseen absolute-angle generalization.

Both learned models use the same convolutional encoder/decoder, 32-dimensional approximately invariant block, 16-dimensional pose-bearing block, losses, data order and 50-epoch budget. Structured S applies prescribed SO(2) harmonic rotations to the pose block. Unrestricted U predicts a residual pose update with an MLP. Both keep the 32-coordinate stable block unchanged during prediction. The architecture calls this the “identity block,” but E0 does not test whether it identifies individual objects. No future observations are fed back during prediction.

Six seed-zero pilot fits compared three learning rates per method. Validation selected **S: 0.001** and **U: 0.0003**; these choices were frozen before the remaining seeds and test evaluation. The selected seed-zero fits were reused. Thus the campaign contains **14 full fits, 700 epochs and 89,600 optimizer steps**, plus two one-epoch smoke fits. No full fit was discarded or retried. Selected checkpoints minimize validation horizon-eight error; they are not necessarily final-epoch checkpoints.

Environment: local Python 3.12.13, PyTorch 2.14.1, NumPy 2.5.3, Apple MPS device, deterministic-algorithm mode, four CPU threads. Versions are pinned in [requirements-lock.txt](../../../requirements-lock.txt). Summed full-fit training/validation wall time was **783.5 seconds (13.1 minutes)**; this excludes environment setup, rendering, final evaluation and analysis. Checkpoints, epoch histories, source snapshots and commands are retained locally under [artifacts](../../../artifacts/e0_rotation/).

S has **181,969** parameters; U has **186,209**. U has more trainable capacity and was not advertised as parameter-matched. A dense multiply-add estimate gives approximately **20.94 million FLOPs** for an eight-step S rollout and **21.01 million** for U: one encoder pass, eight decoder passes, and U's MLP transitions. These estimates count multiply-add as two operations, ignore activation/trigonometric/indexing costs and include transposed-convolution boundary work approximately; they are not measured device FLOPs.

## Primary and control results

All entries below measure predicted pixels, not object identification. “New” means excluded from training in every view; “familiar” means included in training. H8 is the eighth predicted step. Lower error is better. Learned-model entries average five seeds, objects, initial angles and both monotone signs with the registered equal weighting. P and W are deterministic controls.

| Model | New objects: H8 foreground MSE | New objects: H8 full-image MSE | New objects: one-step foreground MSE | Familiar objects: H8 foreground MSE |
|---|---:|---:|---:|---:|
| Structured S | 0.048574 | 0.005289 | 0.019683 | 0.046094 |
| Unrestricted U | 0.107351 | 0.011787 | 0.110496 | 0.105222 |
| Persistence P | 0.288143 | 0.027368 | 0.038166 | 0.286829 |
| Known pixel warp W | 0.003700 | 0.000406 | 0.003906 | 0.003768 |

The one-step column averages the five training increments on unseen objects. It is distinct from the +15°/−15° horizon-one points in the rollout figure. Interpolation at ±10°, extrapolation at ±30°, mixed sequences and inverse sequences are retained separately in [schedules.csv](schedules.csv).

| Seed | S H8 MSE | U H8 MSE | Relative reduction |
|---|---:|---:|---:|
| 0 | 0.031197 | 0.076000 | 59.0% |
| 1 | 0.029858 | 0.113973 | 73.8% |
| 2 | 0.034403 | 0.115418 | 70.2% |
| 3 | 0.080190 | 0.114757 | 30.1% |
| 4 | 0.067223 | 0.116608 | 42.4% |

The aggregate reduction is the ratio of mean errors, not the mean of these percentages. The interval uses 2,000 paired bootstrap draws over the five seed indices and sixteen unseen-object indices, preserving method pairing and episode groups. It does not treat pixels or rollout horizons as independent samples. Five seeds and one synthetic object generator give preliminary evidence, not a broad population guarantee. The complete paired arrays and bootstrap draws are in [paired_primary.npz](paired_primary.npz).

![Prediction metrics](prediction_metrics.png)

In this figure, “unseen-object” also means a new shape, not a withheld view of a familiar shape. A rollout horizon is the number of commands applied without another observation.

| Registered decision rule | Observed | Decision |
|---|---|---|
| At least 10% H8 reduction versus U | 54.8% | Pass |
| Paired interval lower bound above zero | 35.8% | Pass |
| Positive reduction in at least four seeds | Five seeds | Pass |
| Beat persistence at H8 | 0.048574 < 0.288143 | Pass |
| One-step error ≤ 1.05×U + 0.0001 | 0.019683 ≤ 0.116121 | Pass |
| No worse full-image error | 0.005289 < 0.011787 | Pass |

## Failures and diagnostic findings

**Optimization is the dominant qualification.** Three seed-zero pilot fits—S at 0.003 and U at 0.001/0.003—produced near-black validation predictions and approximately 0.18646 primary validation error. Their best-checkpoint mean predicted intensities were below 0.00005. These completed but unsuccessful fits remain in the pilot record; they were not implementation crashes. Some later seeds remained poor for many epochs and then recovered, showing why early impressions and last-epoch-only reporting would be misleading.

![Pilot learning curves](pilot_curves.png)

For the selected U checkpoints at seeds 1–4, flipping all action signs changed predicted images by only **1.2×10⁻¹⁰ to 8.8×10⁻⁹ mean squared difference**. Their full-image reconstruction MSE is approximately **0.0123–0.0125**, versus **0.0010–0.0020** across S seeds. U seed 0 is action-sensitive and reconstructs better (0.00209). Thus nonzero latent variance does not establish a useful action-dependent decoder. U's poor one-step performance—even worse than persistence on average—also warns against interpreting this as an isolated long-horizon effect.

All S checkpoints respond to action sign (image difference approximately 0.0197–0.0234). Their maximum latent composition/inverse errors are at most **4.77×10⁻⁷**. This is a correct implementation of the imposed group law, **not a learned scientific result**. Action-shuffle, sign-flip, latent-variance and identity-stability measurements are in [diagnostics.csv](diagnostics.csv). The identity block is only approximately stable; no claim of complete identity/pose disentanglement follows.

![Fixed test example across all model seeds](all_seed_examples.png)

The grid above uses the same predetermined test example across every seed. The nearly unchanged U predictions under opposite actions at seeds 1–4 illustrate the quantitative failure. The separate grid below uses the median and worst S error among seed-zero positive rotations, selected by an explicit error-ranking rule. Thin or separated structures and exact contours remain difficult for the learned decoder.

![Median and worst seed-zero structured predictions](examples.png)

On the two-fold symmetry set, H8 foreground error was **0.054513 for S versus 0.104356 for U**; on the four-fold set, **0.018181 versus 0.055249**. These are image-prediction diagnostics, not pose-accuracy measurements. The model receives no absolute angle and does not force a unique symmetric-object orientation. Direct-render symmetry checks had zero observed MSE for the audited equivalent-angle cases. This does not establish calibrated pose uncertainty.

The known warp's H8 error is **0.003700**, roughly thirteen times lower than S. The experiment gives an exact action and known pivot, so a geometric image-space solution is strong. Its nonzero discrepancy reflects rasterization/interpolation differences and confirms that decoder learning, rather than physical uncertainty, leaves substantial error. No learned superiority over this control is claimed.

## Integrity and reproducibility

Six contract tests passed: coordinate direction and renderer inverse, group/shape interfaces and paired initialization, absence of targets from the predictor interface, immutable/idempotent prediction scoring, geometry uniqueness/symmetry, and ticket-metadata mutation rejection. A separate MPS 90° warp check matched direct rendering exactly.

The artifact audit passed **133 dataset-file hashes**, confirmed **96 distinct geometry hashes**, verified all **10 selected fit budgets/checkpoints**, and checked **2,048 prediction-ticket/score pairs**. Scores follow commitments, and all held-out evaluation commitments follow the frozen learning-rate decision. Learner files contain only image/action arrays; evaluator truth is separate. No dataset geometries were rejected by the registered asymmetry criterion. See [integrity.json](integrity.json).

Tickets provide auditable ordering and mutation checks within this runner; they are not a security boundary and do not imply statistical independence or identity evidence. Training may use observed future targets for its losses. Final evaluation performs no fitting. No segmentation labels, absolute pose, pretrained model, identity labels, residual dynamics, memory or migration were introduced.

From the repository root, these commands reproduce checks and analysis from retained artifacts:

```sh
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m pytest -q
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m e0_rotation.audit
PYTHONPATH=experiments/e0_rotation/src .venv/bin/python -m e0_rotation.analyze
```

Training/evaluation commands are documented in the [experiment README](../README.md). Completed runs are reused; do not delete or overwrite them to conduct a new experiment. Exact source/config/environment records accompany each run, and the compact [summary](summary.json), [metrics](metrics.csv), [schedule results](schedules.csv) and figures can be versioned independently of large generated files.

## What to do next

The smallest useful next experiment is **E0b: baseline optimization**, not a new MORPH mechanism. One concrete candidate is zero-initializing U's final residual-transition layer so its initial dynamics are identity-preserving, keeping all other architecture/loss/data-budget choices fixed and reporting all seeds. This is a proposed intervention, not an established fix or an executed experiment.

Before running E0b, register its baseline-health measurements (reconstruction and action response), selection rules and a fresh final audit set. Retain E0's original test results; do not use repeated tuning on this test set as independent confirmation. If a healthy unrestricted predictor still loses at longer horizons, the compositional interpretation becomes stronger. If its performance catches up, the present result was primarily an optimization advantage.

Only after that comparison should E1 introduce noncommuting SE(2) motion. E0 provides no evidence yet for learned motor semantics, correspondence under occlusion, physical individual identity, memory promotion, active disambiguation, representation migration or 3-D embodiment.
