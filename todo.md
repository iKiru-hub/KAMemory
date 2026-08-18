# KAMemory: checklist to finish the project and paper

This is the execution order for completing the modeling work, freezing the
results, and writing the paper. Work from top to bottom. Do not begin the next
phase until the current phase's **Done when** condition is satisfied.

## Current next action

- [x] Complete E5: explain E2 recency and forgetting from exact synaptic trace
  survival under subsequent instructive signals.
- [x] Complete E6: cross-validated spatial, cue, and conjunctive CA1 cell
  classification on the factorial task.
- [x] Complete E7: one prespecified quantitative comparison with published CA1
  data and use its quantitative mismatch to define one bounded limitation test.
- [x] Complete E8: rule-specific matched baselines on disjoint development and
  held-out seeds.
- [x] Complete E9: test whether cue-independent background plateau
  heterogeneity repairs the E7 overcoupling without sacrificing EC readout.
- [ ] Freeze the simulation package and move to figure/manuscript assembly; do
  not add another simulation unless a frozen claim fails audit.

---

## Phase 0 — Freeze the question and scope

- [ ] Adopt one primary claim:
  - A content-aware entorhinal instructive signal aligns rapid CA3→CA1
    plasticity with a fixed downstream decoder, preserving memory-content
    decodability during continual learning.
- [ ] Decide the biological framing:
  - [ ] **Recommended:** call the current rule "BTSP-inspired" and test an
    explicit temporal BTSP variant for robustness.
  - [ ] Alternative: make a mechanistic BTSP model, in which case explicit
    eligibility traces, plateau events, timing dependence, and bidirectional
    plasticity are required in the main model.
- [ ] Freeze the four planned main results:
  1. Model architecture, learning rule, and biological interpretation.
  2. Causal effect of IS–decoder alignment on decodability.
  3. Retention, capacity, interference, and degraded-cue retrieval.
  4. Quantitative CA1 spatial/context/sensory result and prediction.
- [ ] Move every other idea to the optional backlog at the end of this file.

**Done when:** the primary claim and four-result structure can be stated on one
page without adding another model component.

---

## Phase 1 — Make the current result reproducible and trustworthy

### 1.1 Environment and entry point

- [x] Create a working local environment from a documented dependency file.
- [ ] Add every imported runtime dependency to the dependency specification,
  including PyTorch, SciPy, Matplotlib, tqdm, Pillow, W&B when used, and
  optional Numba behavior.
- [x] Replace machine-specific path hacks in the definitive backend with paths resolved from the project
  root.
- [x] Create one headless command for running the central experiment.
- [x] Confirm that the command works from a fresh shell at the repository root.
- [x] Record Python, package, device, and operating-system metadata with each
  run.

### 1.2 Determinism and provenance

- [x] Seed Python, NumPy, and PyTorch from one configuration value.
- [x] Select autoencoder checkpoints by stable name, not `os.listdir` index.
- [x] Save the full experiment configuration beside every result.
- [x] Save the Git commit, seed, checkpoint name, and output schema version.
- [x] Never silently reuse old panels or results in a final experiment run.

### 1.3 Model correctness

- [x] Separate `forward`, `learn`, and `recall` operations.
- [x] Ensure evaluation/recall cannot update `W_ca3_ca1`.
- [x] Ensure reset creates a fresh learned weight matrix and clears recordings.
- [x] Stop replacing an `nn.Parameter` object inside every forward pass; update
  state explicitly and safely.
- [ ] Test `sparsemoid` for boundary cases (`K=0`, `K=1`, `K=N`, ties).
- [ ] Investigate `make_equal_tuning`:
  - [ ] Verify whether it is intended to return indices or a weight matrix.
  - [ ] Replace it with an explicit, documented EC→CA3 projection if needed.
  - [ ] Test its dimensions, value range, sparsity, and seed reproducibility.
- [ ] Consolidate or retire the duplicated `MTL`, `MTLev`, and `MTLexp`
  implementations.
- [ ] Consolidate the duplicated `testing_mod` and sensory-generator functions.
- [ ] Fix or remove broken/dead routines such as `train_for_weight_plot`.

### 1.4 Minimal tests

- [x] Test the learning-rule equation against a hand-computed update.
- [x] Test that `alpha=0` leaves weights unchanged.
- [x] Test that learning mode changes weights and recall mode does not.
- [x] Test that identical seeds reproduce identical inputs, weights, and scores.
- [x] Test that checkpoint loading by name is stable.
- [x] Test that aligned, permuted, and rescue conditions use the same base model
  and input patterns.
- [x] Test exact numerical parity with the validated notebook dataset generator,
  autoencoder training loop, MTL storage updates, and recall outputs.

### 1.5 Reproduce the preliminary effect

- [x] Reproduce the existing aligned-IS result from scratch.
- [x] Reproduce the existing shuffled-IS result from scratch.
- [x] Verify the analytical or simulated chance level for the chosen metric.
- [x] Compare the newly generated values with the saved preliminary results.
- [x] Save raw per-memory and per-seed values, not only means or images.

**Done when:** one clean command recreates the preliminary aligned-versus-
shuffled result and all correctness tests pass.

---

## Phase 2 — Establish the central causal result

Use the same memories, CA3 projection, autoencoder checkpoint, initialization,
and paired seed for every condition.

### 2.1 Implement the control conditions

- [x] **Aligned IS:** use the learned content code `c = E(x)`.
- [x] **Fixed-permutation IS:** use `c' = P c` with one fixed permutation per
  run.
  - [x] Permute the complete encoded signal, including the effect of encoder bias.
  - [x] Verify that sparsity and marginal activity statistics are preserved.
- [x] **Random matched IS:** match the aligned condition's sparsity and firing
  rates while removing its relationship to memory content.
- [x] **No-plasticity control:** keep `W_ca3_ca1` fixed.
- [ ] **No-instructive-signal control:** remove or clamp the IS.
- [ ] **Matched local-learning baseline:** implement a simple Hebbian/delta
  control with comparable weight bounds and effective update magnitude.
- [x] **Decoder rescue:** combine `c' = P c` with `D' = D P^T`.
- [x] Verify mathematically and at runtime that the rescue decoder uses
  the correct permutation orientation.

### 2.2 Run and analyze the experiment

- [x] Run a five-seed smoke test for all conditions.
- [x] Inspect activity, sparsity, weight ranges, and output distributions.
- [x] Fix failures before launching a large run.
- [x] Predefine the primary endpoint: final chance-corrected decodability after
  storing a fixed number of memories.
- [x] Predefine E1 secondary endpoints: reconstruction loss, exact/top-K
  recovery, and CA1-code similarity.
- [x] Add retention by memory age as the E2 endpoint.
- [x] Run 20 paired final seeds.
- [x] Plot every seed, paired differences, and confidence
  intervals.
- [x] Use the network/data seed as the independent statistical unit; do not
  treat all memories from one network as independent replicates.

### 2.3 Decision gate

- [x] Confirm that aligned IS outperforms fixed-permutation, random, and
  no-plasticity controls.
- [x] Confirm that the matched decoder substantially rescues the fixed-
  permutation condition.
- [x] Confirm that the result is not explained by different activity levels,
  sparsity, output norms, or autoencoder reconstruction quality.

**Done when:** the decoder-rescue experiment demonstrates that coordinate
alignment between the IS code and the fixed decoder causally determines memory
decodability.

If rescue fails, stop and diagnose the central mechanism before continuing.

---

## Phase 3 — Show that the model performs memory retrieval

### 3.1 Retention and interference

- [x] Measure each memory immediately after storage.
- [x] Re-test all memories after every subsequent storage event.
- [x] Plot recall as a function of memory age.
- [x] Plot recent, middle-aged, and remote memories separately.
- [x] Quantify catastrophic versus gradual forgetting.
- [x] Compare all Phase 2 conditions with paired seeds.
- [x] Derive the exact survival coefficient of an earlier CA3 contribution
  under subsequent row-specific instructive signals.
- [x] Test whether cumulative IS/CA3 interference predicts memory-level recall
  beyond memory age alone.
- [x] Plot memory-quality distributions and survival probabilities across age.
- [x] Relate forgetting half-life to learning rate using paired seeds.

### 3.2 Degraded-cue retrieval

- [x] Define separate encoding inputs, retrieval cues, and reconstruction
  targets.
- [x] Test masking at prespecified levels (0%, 25%, 50%, 75%).
- [x] Test bit flips at prespecified levels (0%, 5%, 10%, 20%).
- [ ] Test additive noise when using continuous spatial inputs.
- [x] Test correlated memories with controlled pairwise overlap.
- [x] Test unseen lures and measure false retrieval.
- [x] Report both content reconstruction and memory-identity discrimination.
- [ ] Compare with chance, nearest-neighbor, and local-learning baselines.
  Chance and nearest-neighbor comparisons are complete; the matched local-
  learning baseline remains pending.

### 3.3 Metrics

- [x] Keep cosine similarity for continuity with preliminary results.
- [x] Add a chance-corrected metric with a clearly derived baseline.
- [x] Add exact/top-K active-bit recovery or precision/recall/F1.
- [x] Add reconstruction loss where appropriate.
- [x] Verify that inactive dimensions do not artificially inflate performance
  using top-K recovery and memory-identity discrimination.

**Done when:** aligned IS improves retention and retrieval from incomplete or
noisy cues, not only reconstruction from the original complete input.

---

## Phase 4 — Characterize capacity and robustness

Use a small smoke-test grid first, then freeze a feasible final grid.

- [x] Define capacity before running the final sweep.
- [x] Sweep number of stored memories from 1–60 in E2a.
- [ ] Sweep CA1 population size.
- [x] Sweep CA3 population size.
- [ ] Sweep input, CA3, and CA1 sparsity.
  CA3 and CA1/IS sparsity are complete in E4; input sparsity remains optional.
- [x] Sweep learning rate `alpha`.
- [ ] Sweep activation sharpness/temperature parameters.
- [x] Sweep IS–decoder alignment using controlled fixed permutations.
- [x] Sweep memory overlap/correlation in E2b.
- [x] Repeat every final E4 grid point across 20 independent paired seeds.
- [x] Plot capacity normalized by network size in E2a.
- [x] Identify a broad working region rather than one optimized point.
- [x] Reserve final seeds and parameter settings that were not used for model
  selection.
- [x] Compare robustness with aligned, fixed-permutation, matched-decoder-rescue,
  and random-matched controls.

**Done when:** we can state how capacity scales, which variables control it, and
where the mechanism fails.

---

## Phase 5 — Produce one rigorous CA1 result

The goal is one quantitative biological connection, not a collection of
qualitative heatmaps.

### 5.1 Repair the track/task design

- [x] Separate spatial and sensory input components explicitly.
- [x] Decide and document which components project to CA3 and which contribute
  to the instructive signal.
- [x] Randomize cue identity independently of cue position.
- [x] Present the same cue at multiple positions.
- [x] Present different cues at the same position.
- [x] Include cue-free and no-plasticity trials.
- [x] Split laps into selection/training and held-out evaluation sets.
- [x] Select neurons using only the training laps.
- [x] Evaluate tuning and decoding only on held-out laps.
- [x] Check that the protocol does not directly prescribe the claimed
  over-representation through its learning-rate profile.

### 5.2 Quantify the prediction

- [x] Reproduce the legacy notebook's same-cue remapping from one clean
  command, including the place-field and effective-input panels.
- [x] Verify the legacy field shift across independent seeds and save the raw
  representative simulation plus seed-level metrics.
- [x] Measure CA1 place-field density across track position.
- [x] Measure cue selectivity within position.
- [x] Decode cue identity across unseen positions.
- [x] Decode context while controlling for position.
- [x] Track receptive-field formation and stability across laps.
- [x] Compare aligned, permuted, random, rescue, and no-plasticity conditions.
- [x] Repeat across cue layouts and independent seeds.
- [x] Classify position-only, position-invariant cue, additive-mixed, and
  conjunctive CA1 cells from training laps with FDR and effect-size control.
- [x] Confirm training-defined selectivity and quantify cue-evoked remapping
  on held-out laps without circular cell selection.
- [x] Compare mixed-selectivity profiles across aligned, random-matched, and
  no-plasticity conditions using the network seed as the independent unit.
- [x] Compare the simulated population profile with an appropriate published
  CA1/EC3 result using a prespecified similarity or error metric.
- [x] Test a minimal heterogeneous-plateau explanation for the quantitative
  E7 mismatch on 20 fresh paired seeds.
- [x] Report the complete plateau-rate sweep and the biological-agreement /
  stable-readout Pareto frontier, including the negative result.

### 5.3 Target conclusion

- [x] Determine whether content-aware IS produces context/sensory selectivity
  that generalizes across spatial position.
- [x] State a concrete experimental prediction about EC3/plateau-related
  activity and subsequent CA1 selectivity.

**Done when:** the CA1 claim holds on held-out laps, across seeds and cue
layouts, and cannot be explained by cue-position confounding or circular cell
selection.

**E9 boundary result:** independently timed, randomly matched background
plateaus do not repair the categorical cue–spatial coupling. Rare events make
only a small improvement in the primary E7 error and no tested rate meets the
prespecified quantitative-repair rule; sufficiently frequent events degrade
EC-output decoding. Treat this as a frozen limitation and do not tune the
event rate post hoc.

---

## Phase 6 — Validate or limit the BTSP interpretation

### 6.1 Rule-specific functional control

- [x] Compare the simplified target-gated overwrite against bounded
  potentiation-only Hebbian storage and a bounded local delta rule.
- [x] Match early immediate recall on disjoint development seeds before the
  held-out comparison.
- [x] Repeat the alignment, fixed-permutation, and matched-decoder conditions
  for every rule.
- [x] Quantify sequential retention, ongoing acquisition, forgetting time,
  and weight saturation on 20 paired held-out seeds.

**Result:** coordinate alignment and decoder rescue generalize across all
three rules. The target-gated rule avoids Hebbian saturation and preserves
new-memory acquisition, while the explicit error-correcting delta rule trades
lower ongoing acquisition for longer retention. This supports a distinct
stability–plasticity interpretation, not a claim of algorithmic optimality.

### 6.2 Temporally explicit BTSP reference

- [ ] Implement a temporally explicit reference model with:
  - [ ] Presynaptic eligibility trace.
  - [ ] Plateau/instructive trace or event.
  - [ ] Seconds-long interaction window.
  - [ ] Timing-dependent potentiation and depression.
  - [ ] Dependence on initial synaptic strength.
- [ ] Verify one-trial receptive-field formation.
- [ ] Verify a characteristic temporal signature, such as asymmetric field
  formation or speed-dependent field width.
- [ ] Compare the explicit rule with the simplified update using identical
  tasks and seeds.
- [ ] Repeat the central alignment and decoder-rescue experiment with the
  explicit rule.
- [ ] Decide final terminology based on the result:
  - [ ] "Simplified BTSP model" if the explicit validation succeeds.
  - [ ] "BTSP-inspired rule" if only the functional principles are retained.

**Done when:** the biological terminology used in the title, abstract, figures,
and discussion is supported by a direct model comparison.

---

## Phase 7 — Freeze final simulations and statistics

- [ ] Freeze all primary metrics, parameters, seeds, and exclusion rules.
- [ ] Run the complete experiment suite from clean configurations.
- [ ] Save raw data for every seed and every main figure panel.
- [ ] Save aggregate tables separately from raw data.
- [ ] Calculate effect sizes and confidence intervals.
- [ ] Correct for multiple comparisons where applicable.
- [ ] Display individual independent runs in summary figures.
- [ ] Check that all conclusions survive the reserved final seeds.
- [ ] Run sensitivity analyses around the reported parameter values.
- [ ] Record failed and negative controls in the experiment log.
- [ ] Generate a source-data table for every figure.
- [ ] Archive the frozen configuration and checkpoint manifest.

**Done when:** no main result depends on a single seed, selected example, stale
cache, or undisclosed parameter choice.

---

## Phase 8 — Finalize the figures

### Figure 1 — Model and rule

- [ ] Architecture with pathways labeled accurately.
- [ ] Simplified learning-rule equation.
- [ ] BTSP/reference-rule validation.
- [ ] One representative learning example.

### Figure 2 — Central causal result

- [ ] Aligned versus fixed-permutation IS.
- [ ] Matched decoder rescue.
- [ ] Random/no-plasticity/local-learning controls.
- [ ] Decodability versus measured IS–decoder alignment.

### Figure 3 — Memory function

- [ ] Retention by memory age.
- [x] Rule-specific baselines and stability–plasticity trade-off.
- [ ] Capacity/scaling.
- [ ] Degraded-cue retrieval.
- [ ] Correlated-memory or lure-discrimination result.

### Figure 4 — CA1 result and prediction

- [x] Task design with cue and position independently varied.
- [x] Held-out CA1 receptive-field/context result.
- [x] Quantitative comparison across conditions.
- [x] Testable biological prediction.

### Figure quality checks

- [ ] Generate every panel from frozen raw data.
- [ ] Use consistent colors, labels, units, fonts, and condition names.
- [ ] Include sample sizes and uncertainty definitions in captions.
- [ ] Avoid illustrative BTSP kernels that are not generated by or explicitly
  connected to the implemented rule.
- [ ] Check legibility at final publication size.
- [ ] Create only necessary supplementary figures for robustness and controls.

**Done when:** the complete scientific argument can be understood from the four
figures and their captions.

---

## Phase 9 — Material-freeze package

For every main panel, create a short result record containing:

- [ ] Scientific question.
- [ ] Compared conditions.
- [ ] Independent sample size.
- [ ] Primary metric and chance baseline.
- [ ] Effect size and confidence interval.
- [ ] Statistical test, if used.
- [ ] Exact configuration and seed list.
- [ ] Raw-data path.
- [ ] Figure-generation command.
- [ ] One-sentence result.
- [ ] One-sentence limitation.

Project-wide freeze:

- [ ] Run the full pipeline from a clean checkout/environment.
- [ ] Confirm that all figures match their source-data tables.
- [ ] Confirm that no final notebook depends on hidden execution state.
- [ ] Tag or commit the frozen simulation version.
- [ ] Stop adding experiments unless an existing claim fails verification.

**Done when:** all simulation material needed for writing can be recovered
without inspecting notebook history or asking how a panel was produced.

---

## Phase 10 — Write the paper in one pass

Write in this order:

- [ ] Results section from the final figure sequence.
- [ ] Complete figure captions.
- [ ] Methods from the frozen configurations and implementation.
- [ ] Discussion, limitations, and experimental predictions.
- [ ] Introduction and relationship to adjacent BTSP memory models.
- [ ] Abstract.
- [ ] Title.
- [ ] Supplementary methods and results.
- [ ] Data/code availability statement.
- [ ] Author contributions, acknowledgements, and funding.
- [ ] Complete bibliography and verify every factual claim/citation.

Manuscript-specific corrections:

- [ ] Replace the placeholder title, authors, and abstract.
- [ ] Rewrite the existing incomplete Results section.
- [ ] Write the currently empty Discussion.
- [ ] Correct the Methods: `alpha`, not `beta`, is the learning-rate parameter.
- [ ] Describe the IS as the encoded CA1 target/content code used in the model,
  rather than equating it directly with the raw EC input.
- [ ] Make the learning-rule equation match the implementation exactly.
- [ ] Add the missing bibliography database.
- [ ] Ensure anatomical labels and biological claims match the final model.

**Done when:** a reader can reproduce every result from the Methods, configs,
source data, and code release.

---

## Phase 11 — Internal review and release

- [ ] Perform a claim-by-claim audit against figures and source data.
- [ ] Have collaborators review the scientific framing before copyediting.
- [ ] Address major scientific comments first.
- [ ] Proofread equations, terminology, captions, and references.
- [ ] Build the manuscript and inspect every rendered page.
- [ ] Prepare a clean public repository or archival snapshot.
- [ ] Remove tracked caches, local W&B internals, and machine-specific files
  from the release package.
- [ ] Include a concise README with install, smoke-test, and reproduction
  commands.
- [ ] Archive code and source data with a persistent identifier when ready.
- [ ] Submit the preprint/manuscript.

**Done when:** the manuscript, figures, source data, and reproduction package
are ready for an external reviewer with no private setup knowledge.

---

## Existing useful assets

- [x] Initial Python model and training code.
- [x] Saved autoencoder checkpoints.
- [x] Preliminary aligned-versus-shuffled simulations.
- [x] Preliminary parameter searches.
- [x] Draft Figure 1 and Figure 2 notebooks/panels.
- [x] COSYNE abstract with the core biological motivation.
- [ ] Consolidate exploratory notebooks into final experiment scripts.
- [ ] Regenerate and save final paper data after the model is frozen.
- [ ] Replace preliminary group-distance/statistical plots with the prespecified
  paired analyses above.
- [ ] Replace the preliminary accuracy-versus-CA1/IS-similarity plot with the
  controlled alignment sweep and decoder-rescue analysis.

---

## Optional backlog — must not block the paper

- [ ] Analytical capacity theory.
- [ ] Recurrent CA3 dynamics.
- [ ] Multicompartment or conductance-based CA1 neurons.
- [ ] Multiple sensory modalities.
- [ ] Multi-day consolidation or systems-level replay.
- [ ] Reinforcement-learning controller.
- [ ] Large evolutionary hyperparameter searches.
- [ ] Matching additional CA1 findings beyond the one selected main result.
- [ ] Hardware or neuromorphic implementation.

Only promote an optional item into the main checklist if a required result
cannot be interpreted without it.
