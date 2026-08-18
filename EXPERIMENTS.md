# Experiments Plan

**Central Thesis:**
> Content-aware instructive signals place rapidly learned CA1 representations into coordinates that remain readable by a stable downstream decoder.

## Summary Table

| ID | Experiment | Main question | Priority |
|---|---|---|---|
| E0 | Backend and model validation | Is the implementation trustworthy? | Required (Methods) |
| E1 | IS–decoder alignment and rescue | Does coordinate alignment causally determine decodability? | Central Result |
| E2 | Sequential memory function | Does the mechanism support retention, capacity, and retrieval? | Central Result |
| E3 | CA1 spatial–sensory task | Does it produce a meaningful CA1-like prediction? | Biological Result |
| E4 | Focused robustness | Does the conclusion survive reasonable parameters and seeds? | Required Support |
| E5 | Interference mechanism | Does the rule quantitatively explain recency and forgetting? | Finishing Analysis |
| E6 | CA1 mixed selectivity | Which spatial, cue, and conjunctive cell classes form? | Finishing Analysis |
| E7 | Published CA1 comparison | Does one simulated population profile match empirical data? | Biological Anchor |
| E8 | Rule-specific baselines | Is alignment rule-general, and what does the overwrite rule contribute? | Mechanistic Control |
| E9 | Plateau heterogeneity | Can cue-independent background plateaus repair cue–spatial overcoupling without losing the stable readout? | Limitation Test |

## Detailed Experiment Checklist

### [x] E0 — Model Validation
**Proposed File:** `experiments/00_model_validation.py`  
**Goal:** Verify that the implementation is trustworthy. This serves as a methods/supplementary figure.

- [x] Autoencoder learns a decodable CA1 representation.
- [x] `retrieve()` never changes CA3→CA1 weights.
- [x] `store()` matches a hand-computed BTSP-inspired update.
- [x] `alpha=0` produces no change.
- [x] Identical seeds reproduce inputs, weights, and outputs.
- [x] Checkpoints and configs work independently of the working directory.
- [x] Weight ranges and activity sparsities remain sensible during storage.

Implemented in `src/experiments/00_model_validation.py`. The deterministic E0
run saves its machine-readable report and validation figure in
`src/experiments/plots/`. Dataset generation, autoencoder minibatch order,
storage updates, and recall outputs have exact numerical parity tests against
the validated notebook backend.

---

### [x] E1 — Alignment Causality and Decoder Rescue
**Proposed File:** `experiments/01_alignment.py`  
**Goal:** The decisive experiment. Establish if coordinate alignment causally determines decodability.

**Conditions:**
- [x] 1. Aligned IS: $c = E(x)$
- [x] 2. Fixed permutation: $c' = Pc$
- [x] 3. Fixed permutation + decoder rescue: $D' = DP^T$
- [x] 4. Random matched IS (same sparsity and marginal activity, no content relationship)
- [x] 5. No plasticity

**Protocol:**
- [x] Use paired seeds, identical memories, identical CA3 projections, and identical initial states.

**Endpoints:**
- **Primary:**
    - [x] Chance-corrected content decodability after storing a fixed number of memories.
- **Secondary:**
    - [x] Top-K precision/recall/F1.
    - [x] Reconstruction MSE.
    - [x] Cosine similarity.
    - [x] CA1–target-code similarity.
    - [x] Activity sparsity and output norms.

**Expected Causal Pattern:**
`aligned > fixed permutation ≈ random/no plasticity`  
`fixed permutation + matched decoder ≈ aligned`

*Note: If matched-decoder rescue fails, diagnose the model before expanding.*

Implemented in `src/experiments/01_alignment.py` using 20 paired seeds. Raw
arrays, summary statistics, and the E1 figure are saved in
`src/experiments/plots/`. The run also reproduces the saved notebook comparison
to numerical precision before evaluating the corrected controls. The main run
uses the same EC→CA3-initialization-before-dataset order and row-by-row NumPy
sampling protocol as the validated notebooks.

---

### [x] E2 — Sequential Memory Function
Split into two steps: Retention/Capacity and Degraded-cue Retrieval.

#### [x] E2a — Retention and Capacity
**Proposed File:** `experiments/02_retention.py`  
**Goal:** Convert existing capacity figures into a rigorous retention experiment.

- [x] Store memories sequentially.
- [x] Test every previous memory after each storage event.
- [x] Plot performance against memory age and total memory load.
- [x] Compare: Aligned, Fixed-Permutation, Rescue, and No-Plasticity.
- [x] Define capacity using a chance-corrected threshold before final seeds.
- [x] Report capacity normalized by CA3 or CA1 population size.

Implemented in `src/experiments/02_retention.py` with 20 paired held-out
seeds, loads from 1–60 memories, a full triangular store/recall matrix, and
the pre-existing notebook threshold (raw cosine 0.8) expressed in
chance-corrected coordinates. Raw arrays and provenance are saved with the
figure.

#### [x] E2b — Degraded-cue Retrieval
**Proposed File:** `experiments/03_degraded_cues.py`  
**Goal:** Test if the system acts as associative memory.

**Setup:** Use distinct encoding inputs, retrieval cues, and reconstruction targets.
**Conditions to test:**
- [x] 0%, 25%, 50%, and 75% masking.
- [x] Bit-flip noise.
- [x] Small controlled overlap series between memories.
- [x] Unseen lure patterns.

**Metrics:**
- [x] Reconstruction of memory content.
- [x] Identification of which stored memory was retrieved.

Implemented in `src/experiments/03_degraded_cues.py` using separate clean
storage inputs, corrupted retrieval cues, and clean reconstruction targets.
The aligned network retrieves content and identity above controls and chance,
and matched-decoder rescue reproduces aligned performance. A nearest-neighbor
reference remains stronger under corruption, so the conservative claim is
degraded-cue retrieval rather than optimal pattern completion.

---

### [x] E3 — CA1 Spatial–Sensory Result
**Proposed File:** `experiments/04_ca1_track.py`  
**Goal:** Produce a meaningful CA1-like prediction from a factorial track design.

**Protocol (Factorial Track Design):**
- [x] Same cue at multiple positions.
- [x] Different cues at the same position.
- [x] Cue-free laps.
- [x] Held-out evaluation laps.
- [x] Neurons selected using training laps only.

**Primary Measurements:**
- [x] Cue decoding while controlling for position.
- [x] Position decoding while controlling for cue.
- [x] Place-field density across the track.
- [x] Receptive-field formation across laps.
- [x] Stability of fields on held-out laps.
- [x] EC-output content decoding.

*Note: Focus on Aligned vs. Random-matched IS and No Plasticity for CA1 selectivity.*

Implemented in `src/experiments/04_ca1_track.py` using the validated mixed
spatial/sensory autoencoder and BTSP backend. Cue identity is fully crossed
with four presentation positions across two layouts. Plasticity is gated at
cue events but has the same profile for both cue identities and all controls.
All cell selection uses training laps; every reported endpoint uses separately
generated held-out laps. Fixed permutation preserves intrinsic CA1 cue and
position information but breaks the stable EC readout, which the matched
decoder rescues.

Before the paper rerun, the autoencoder was retrained on independent samples
from this exact factorial-track generator rather than the older generic
sensory generator. The validation-selected checkpoint
`ae_factorial_paper_v1` improves held-out autoencoder MSE from 0.0159 to
0.00335 and sensory cosine from 0.843 to 0.976 relative to `ae_6`. With the new
checkpoint, aligned held-out cue accuracy is 1.0, position accuracy is 0.875,
field stability is 0.554, and EC-output cosine is 0.884; fixed permutation
reduces output cosine to 0.114 and the matched decoder restores 0.884.

**Legacy notebook reproduction:** `src/experiments/04b_legacy_remapping.py`
replays the earlier one-cue remapping experiment from a clean command. It pins
the notebook's `ae_6` checkpoint, legacy stimulus RNG order, pre-update CA1
recording convention, constant effective learning rate, and cell-selection
threshold. The old place-field and effective-receptive-field panels are
regenerated alongside a 12-seed summary. Moving the cue from position 10 to
30 shifts the selected CA1 population-field peak by 20 bins in every paired
seed. This is retained as backward-compatible descriptive evidence; the
factorial held-out experiment above carries the causal claim.

---

### [x] E4 — Focused Robustness
**Proposed File:** `experiments/05_sensitivity.py`  
**Goal:** Show a broad working regime via a small, interpretable sweep.

**Sweep Parameters:**
- [x] Learning rate.
- [x] Number of stored memories.
- [x] CA3 size.
- [x] CA1/CA3 sparsity.
- [x] Degree of IS–decoder misalignment.

**Execution Plan:**
- [x] Use a small paired smoke run for development.
- [x] Run 20 paired seeds for the final selected settings.

Implemented in `src/experiments/05_sensitivity.py`. The reference configuration
is frozen in `src/configs/optimized_memory.json` from the repository's explicit
`src/optim_wb/best_params.yaml` artifact: checkpoint `ae_8`, alpha 0.208794,
beta 54, 18 active CA1 units, 22 active CA3 units, and 50 CA3 units. E4 uses
new seeds that were not used for E1–E3 and holds memories and model randomness
paired across settings. Aligned IS outperforms the fixed and random controls
throughout the tested grids; matched-decoder rescue is numerically coincident
with aligned performance. Performance degrades smoothly with memory load,
overly dense CA3 activity, very small CA3 populations, and coordinate
misalignment, identifying interpretable limits rather than a single optimum.

---

### [x] E5 — Synaptic Trace Survival and Forgetting
**Proposed File:** `experiments/06_interference_analysis.py`  
**Goal:** Explain E2 recency and interference directly from the implemented update rule.

- [x] Derive the exact survival coefficient of each stored CA3 contribution
  under subsequent instructive signals.
- [x] Replay the validated E2 protocol with paired seeds and record IS/CA3
  overlap, predicted trace survival, CA1-code similarity, and output recall.
- [x] Test whether predicted trace survival explains measured forgetting beyond
  memory age alone.
- [x] Plot recall distributions and survival curves, not only means.
- [x] Relate the measured forgetting half-life to learning rate.
- [x] Save raw memory-level values and use the network seed as the independent
  unit for inferential summaries.

Implemented in `src/experiments/06_interference_analysis.py` using the frozen
E2 source arrays and the same 20 seeds. The exact contribution decomposition
reconstructs final weights with zero error and recall within 1.8e-7. Surviving
trace correlates positively with recall (mean within-seed r=0.718), whereas
crosstalk correlates negatively (r=-0.711). Age captures most of the smooth
decay; trace and crosstalk add a modest but consistent mean delta-R-squared of
0.0030. Increasing alpha from 0.12 to 0.55 shortens measured recall half-life
from 34.9 to 8.3 subsequent stores, providing the mechanistic link between the
learning rule, E2 recency, and the E4 learning-rate effect.

---

### [x] E6 — Cross-validated CA1 Mixed Selectivity
**Proposed File:** `experiments/07_ca1_mixed_selectivity.py`  
**Goal:** Resolve spatial, position-invariant cue, and conjunctive CA1 coding.

- [x] Fit cell-wise position, cue-identity, and position×cue effects using
  training laps only.
- [x] Evaluate classified cells and effect sizes on held-out laps.
- [x] Compare aligned, random-matched, and no-plasticity conditions.
- [x] Test whether cue tuning predicts relocation/remapping strength without
  selecting cells on the evaluation data.
- [x] Avoid object-vector terminology unless relative distance and direction
  are manipulated explicitly.

Implemented in `src/experiments/07_ca1_mixed_selectivity.py` by replaying the
frozen E3 inputs and reproducing every analyzed E3 final weight bit-for-bit.
Cell classes are assigned from training laps using factorial partial-F tests,
BH-FDR control, and a prespecified minimum partial eta-squared of 0.05; all
effect sizes and cue-evoked remapping measurements use held-out laps. Aligned
plasticity produces a sparse tuned population (8.16%): 7.33% conjunctive,
0.575% position-only, and 0.254% position-invariant cue cells, with no
detectable additive-mixed class. Random-matched plasticity creates more
conjunctive cells (10.09%) but fewer position-only (0.463%) and no
position-invariant cue cells. No-plasticity produces no classified cells.
Training cue tuning strongly predicts held-out cue-evoked remapping within
the aligned population (mean seed-level Spearman rho=0.996). Because
the factorial input combinations are deterministic, held-out target effect
sizes saturate at partial eta-squared=1 for classified cells; this result is
best interpreted as exact generalization of sparse response classes, not as
an estimate of biological trial-to-trial reliability.

---

### [x] E7 — Prespecified Published CA1 Comparison
**Proposed File:** `experiments/08_ca1_data_comparison.py`  
**Goal:** Anchor one simulated CA1 population prediction to one empirical profile.

- [x] Select one primary paper/dataset and one population profile before
  examining model agreement.
- [x] Document digitization, normalization, inclusion, and alignment rules.
- [x] Compare model and data with one prespecified similarity/error metric and
  uncertainty across model seeds.
- [x] Report disagreement and limitations as well as agreement.

Implemented in `src/experiments/08_ca1_data_comparison.py` using the exact CA1
counts reported by Symanski, Bladon et al. (eLife, 2022): spatial fields in
71/108 odor-responsive cells and 273/477 odor-inactive cells. The reference
profile, extraction rules, model mapping, primary metric, and limitations are
frozen in `src/experiments/reference_data/symanski_2022_ca1_profile.json`; no
figure digitization is used. The published data show a modest positive
cue–space association (65.7% versus 57.2%; continuity-corrected odds ratio
1.43). Every aligned-model seed matches that direction, but the model predicts
96.6% versus 0.62% and a geometric-mean odds ratio of 4,041. Thus the current
sparse deterministic CA1 activity overcouples cue and spatial selectivity by
roughly 2,835-fold in odds-ratio terms. E7 supplies a biological directional
anchor and, more importantly, a clear quantitative limitation rather than a
claim of numerical biological realism.

---

### [x] E8 — Rule-Specific Baselines
**Proposed File:** `experiments/09_rule_baselines.py`  
**Goal:** Separate the rule-general coordinate-alignment claim from the
specific stability–plasticity properties of the target-gated overwrite.

- [x] Implement the validated target-gated update, bounded potentiation-only
  Hebbian storage, and a bounded local delta rule as pure tested functions.
- [x] Give every rule the same CA3 activity, content-bearing CA1 target,
  initial weights, memories, storage order, encoder, and decoder.
- [x] Select alternative learning rates on disjoint development seeds by
  matching early immediate recall and then maximizing integrated retention.
- [x] Evaluate aligned, fixed-permutation, and matched-decoder conditions for
  every rule on 20 new paired seeds.
- [x] Compare full sequential retention, ongoing immediate recall, and weight
  saturation on the held-out seeds.

Implemented in `src/experiments/09_rule_baselines.py` with configuration in
`src/configs/rule_baselines.json`. The validated target-gated learning rate is
fixed at 0.35; development seeds select 0.75 for bounded Hebbian storage and
0.20 for the delta rule. Held-out early immediate recall is matched across the
three rules (0.903, 0.894, and 0.903, respectively). Coordinate permutation
reduces endpoint recall to chance for every rule and the matched decoder
restores the aligned value numerically, showing that alignment is a
rule-general representational constraint.

The sequential-memory profiles differ. The target-gated rule preserves the
ability to encode the next memory across load (mean age-0 recall 0.943) and
avoids weight saturation. Potentiation-only Hebbian storage saturates every
weight and collapses to age-0 recall 0.314 and integrated recall 0.141. The
explicit error-correcting delta rule has lower ongoing age-0 recall (0.782)
but a longer half-life (24.3 versus 13.2 stores) and higher integrated recall
(0.483 versus 0.439). Thus the current rule is not claimed to dominate an
error-driven algorithm: it realizes a distinct high-plasticity operating
point without requiring a signed CA1 prediction-error signal.

Raw arrays, development sweeps, seed-level paired effects, the JSON report,
and the E8 figure are saved in `src/experiments/plots/`. The original update
remains the backend default, and all 37 tests, including exact legacy-equation
and notebook-parity tests, pass.

---

### [x] E9 — Cue-Independent Plateau Heterogeneity
**Proposed File:** `experiments/10_is_heterogeneity.py`  
**Goal:** Test a minimal explanation for the E7 cue–spatial overcoupling.

- [x] Preserve E3 content-aligned cue events and its weak non-cue baseline.
- [x] Add strong plateau events only outside cue presentations.
- [x] Draw background signals from a randomly matched version of the same CA1
  code bank, preserving signal statistics while removing coincident content.
- [x] Use nested masks so every rate is paired within seed and layout.
- [x] Sweep from no background events through an explicit failure regime
  (0%, 1%, 2%, 5%, 10%, 20%, 40%, and 80% per non-cue track bin).
- [x] Reuse the frozen E7 cell mapping and primary absolute log-odds-ratio
  error without refitting its threshold.
- [x] Require quantitative biological agreement (absolute log-OR error at
  most 1.0) while retaining at least 80% of the rate-zero EC-output cosine.
- [x] Run 20 fresh paired seeds across both E3 cue layouts and report the full
  biological-agreement/readout frontier.

Implemented in `src/experiments/10_is_heterogeneity.py` with the frozen sweep
in `src/configs/is_heterogeneity.json`. Rate zero numerically reproduces direct
E3 aligned training. Across seeds 2300–2319, the rate-zero population is 96.7%
spatial among cue-responsive cells versus 0.62% among cue-inactive cells, and
the 80% background condition is 99.8% versus 1.38% (published: 65.7% versus
57.2%). The primary absolute log-odds-ratio error is 7.96 at rate zero. Rare
events reduce it only slightly, to 7.92 at 5%, before it rises to 9.36 at 80%.

The stable readout is initially insensitive to rare background events, then
falls from cosine 0.884 at rate zero to 0.837 at 40% and 0.669 at 80%. No rate
meets the frozen quantitative-repair rule. Thus independently timed,
content-independent plateau heterogeneity alone is not a sufficient repair:
the model needs a mechanism that creates stable spatial coding outside the
cue-defined population (or a less deterministic activity/readout model), not
simply more unmatched plateau events. This is a useful negative result and a
clear model boundary, not evidence against the central coordinate-alignment
result.

Raw cell masks, seed-level contingency tables, biological errors, output
metrics, event schedules, provenance hashes, JSON report, and the E9 figure
are saved in `src/experiments/plots/`. All 44 backend and experiment tests
pass.

---

## Thesis Figure Structure

1. **Model & Validation:** Architecture, learning rule (BTSP-inspired), and validation.
2. **Alignment:** Manipulation and matched-decoder rescue.
3. **Memory:** Retention, rule-specific trade-offs, capacity, and degraded-cue retrieval.
4. **Biology:** Spatial–sensory CA1 results and biological prediction.
5. **Appendix/Supplement:** Sensitivity plots.

## Out of Scope

- Evolutionary parameter optimization.
- Exhaustive high-dimensional sweeps.
- Multiple alternative plasticity models.
- Full mechanistic BTSP timing traces.
- Multiple environments or behavioral tasks.
- Maintaining `MTL`, `MTLev`, and `MTLexp` as separate scientific models.
- A large comparison of learning rules.
