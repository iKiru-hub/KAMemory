# KAMemory paper tracker

This document is the control record for the **paper**, not a thesis and not a
general repository backlog. It freezes the intended scientific story, maps
claims and figures to source data, and tracks the work required for a
submission-ready manuscript.

## Paper status

| Item | Status |
|---|---|
| Simulation scope | **Refrozen** after distribution-matched AE retraining and E3/E6/E7/E9 rerun |
| Article type | Focused computational neuroscience / modeling paper |
| Central claim | Frozen provisionally; wording below |
| Literature review | Focused novelty audit complete; broader Discussion audit remains |
| Main-figure composition | Four-figure draft complete; captions and source data recorded |
| Results text | Not started |
| Methods text | Legacy draft exists but requires a complete rewrite |
| Introduction | Four-paragraph cited draft written; revise after Results captions freeze |
| Discussion | Not started |
| Abstract and title | Working versions only |
| Reproducibility package | Experiments complete; paper-level freeze audit pending |

### Current next action

- [x] Complete a targeted literature review for the Introduction and
  Discussion using the review brief below.
- [x] Build a verified claim–citation table before writing literature-dependent
  prose.
- [ ] Revisit the working title and Introduction outline after the review, but
  do not change the central simulation claim without recording the reason in
  the decision log.

---

## Working identity of the paper

### Recommended working title

**Decoder-aligned instructive signals support rapid associative memory in a
BTSP-inspired hippocampal network**

Alternative titles to reconsider after the literature review:

1. **Content-aligned instructive signals preserve decodability during rapid
   hippocampal plasticity**
2. **Representational alignment links rapid CA3–CA1 plasticity to stable
   entorhinal readout**
3. **A coordinate-alignment principle for instructive-signal-guided
   hippocampal memory**

### One-sentence question

How can rapid plasticity at CA3→CA1 synapses create content-rich CA1
representations that remain interpretable by a stable downstream decoder?

### Central claim

> Content-aware instructive signals place rapidly learned CA1 representations
> into coordinates that remain readable by a stable downstream decoder.

### Strongest causal statement supported by the simulations

With memories, initialization, CA3 activity, storage order, and decoder held
fixed, permuting the instructive-signal coordinates abolishes downstream
content decoding; applying the matched inverse permutation to the decoder
restores it. Coordinate compatibility between the instructive signal and the
decoder therefore causally determines decodability in this model.

### Paper type and audience

- A focused computational/modeling article about a representational principle
  for rapid hippocampal learning.
- Relevant to readers working on hippocampal memory, CA1 representations,
  behavioral-timescale plasticity, associative memory, and neural
  representational geometry.
- Not positioned as a biophysical reconstruction of BTSP or as a quantitative
  fit to CA1 population data.

### Contribution hierarchy

1. **Primary contribution:** a decoder-alignment principle for
   instructive-signal-guided rapid plasticity, supported by a matched-decoder
   causal rescue.
2. **Functional contribution:** the aligned mechanism supports sequential
   retention and retrieval from degraded cues, with an explicit interference
   mechanism.
3. **Mechanistic control:** coordinate alignment generalizes across three
   local learning rules, while the rules occupy different
   stability–plasticity regimes.
4. **Biological connection:** a factorial spatial–sensory task produces stable
   spatial, cue, and conjunctive CA1-like responses and remapping.
5. **Boundary result:** the model captures the direction but not the magnitude
   of a published CA1 cue–space association, and unmatched background plateau
   heterogeneity does not repair the discrepancy.

---

## Scope guardrails

### Claims this paper may make

- Coordinate alignment between a content-bearing instructive signal and a
  fixed decoder is sufficient and necessary for decodable storage under the
  controlled model manipulations.
- The matched-decoder rescue identifies representational coordinates, rather
  than signal sparsity or update magnitude, as the causal factor.
- The model supports associative-memory functions under the tested masking,
  corruption, overlap, and load protocols.
- The implemented target-gated rule has an interpretable recency/interference
  mechanism and a distinct stability–plasticity profile relative to the
  matched baselines.
- Aligned plasticity produces held-out spatial, cue, and conjunctive response
  structure in the factorial task.
- The model agrees with the direction of one published CA1 population
  association but strongly overestimates its magnitude.
- Cue-independent randomly matched background plateaus, as implemented in E9,
  are insufficient to resolve that overcoupling.

### Claims this paper must not make

- That the implemented update is a complete or quantitatively realistic BTSP
  mechanism.
- That the model implements experimentally measured eligibility or plateau
  time courses, seconds-long timing kernels, bidirectional BTSP, dendritic
  compartments, or speed-dependent field formation.
- That the instructive signal is simply the raw entorhinal input. In the model
  it is the encoded CA1 target/content code `c = E(x)`.
- That the target-gated rule is universally superior or optimal. The delta
  baseline retains old memories longer under the matched comparison.
- That degraded-cue retrieval is optimal pattern completion. The
  nearest-neighbor reference remains stronger under some corruption regimes.
- That the model quantitatively reproduces CA1 population statistics or
  explains the Symanski–Bladon dataset neuron by neuron.
- That cue remapping is object-vector coding; distance and direction from an
  object were not independently manipulated.
- That individual cells, memories, laps, or layouts are independent
  inferential replicates. The network/data seed is the independent unit.

### Terminology to use consistently

| Use | Avoid or qualify |
|---|---|
| BTSP-inspired update/rule | BTSP model, unless explicitly qualified |
| content-aware or decoder-aligned instructive signal | raw EC input as the IS |
| CA1 target/content code | ground-truth biological CA1 code |
| stable EC-output decoder/readout | stable cortical readout, unless motivated |
| degraded-cue retrieval | optimal pattern completion |
| cue-evoked remapping | object-vector response |
| artificial CA1/CA3 units | biological neurons without qualification |
| directional biological agreement | quantitative biological match |
| network/data seed, `n = ...` | cells or memories as independent samples |

### Exact update to report

For CA3 activity `x_t`, encoded CA1 target `c_t`, plastic weights `W_t`, and
learning rate `alpha`, the validated target-gated update is

```text
W_(t+1) = (1 - alpha c_t) ⊙ W_t + alpha c_t x_t^T
```

where the first product is row-wise through broadcasting. `alpha`, not
`beta`, is the learning rate. `beta` controls activation sharpness in the
sparse activation function. The Methods must define all dimensions,
broadcasting, sparsity operations, and the stable encoder/decoder explicitly.

---

## Narrative arc

### Abstract logic — six-sentence skeleton

1. Rapid hippocampal plasticity must create new representations without making
   them unreadable to downstream circuits that change more slowly.
2. We formulate this as a coordinate-alignment problem in a BTSP-inspired
   EC–CA3–CA1–EC network with a pretrained stable CA1 decoder.
3. Content-aligned instructive signals support decodable storage, whereas
   matched coordinate permutations abolish it and a matched decoder rescues
   it.
4. The mechanism supports sequential retention and degraded-cue retrieval;
   exact trace analysis and learning-rule controls expose its interference and
   stability–plasticity properties.
5. In a factorial spatial–sensory task it produces stable spatial, cue, and
   conjunctive CA1-like responses, but substantially overcouples cue and
   spatial selectivity relative to a published CA1 contingency.
6. These results identify decoder alignment as a general computational
   constraint on instructive-signal-guided learning while defining the limits
   of the current BTSP-inspired activity model.

### Results outline and provisional headings

1. **A content-bearing instructive signal defines decoder-compatible CA1
   coordinates**
   - Introduce the two-stage architecture and exact learning rule.
   - Establish that storage and retrieval are separated and the pretrained
     autoencoder supplies the fixed representational coordinate system.

2. **Instructive-signal–decoder alignment causally determines memory
   decodability**
   - Aligned, fixed-permutation, random-matched, no-plasticity, and
     matched-decoder-rescue conditions.
   - Lead with the chance-corrected endpoint and paired seed effects.
   - Use the rescue to distinguish intrinsic CA1 information from readable
     downstream content.

3. **Aligned storage supports sequential retention and degraded-cue
   retrieval**
   - Full retention matrix, memory age, capacity, masking, bit flips, overlap,
     and lures.
   - State that the nearest-neighbor comparison limits any optimal-pattern-
     completion claim.

4. **Synaptic trace survival explains forgetting and differs across local
   learning rules**
   - Exact contribution/survival decomposition and crosstalk.
   - Learning-rate/half-life relationship.
   - Rule-general alignment rescue plus rule-specific
     stability–plasticity trade-offs.

5. **Aligned instructive signals generate stable spatial–sensory CA1 response
   structure**
   - Factorial cue × position design, cue-free laps, held-out evaluation.
   - Spatial, cue, and conjunctive effects; remapping relationship.
   - Legacy place-field shift only as descriptive continuity, preferably in
     the supplement.

6. **Published CA1 data reveal a quantitative boundary of the current model**
   - Prespecified Symanski, Bladon et al. population contingency.
   - Directional agreement and magnitude disagreement.
   - E9 negative result: unmatched plateau heterogeneity does not resolve the
     overcoupling and eventually damages the stable readout.

### Discussion outline

1. Restate the coordinate-alignment principle without repeating all results.
2. Explain why matched-decoder rescue is the central causal result.
3. Relate content and index views of hippocampal representations.
4. Discuss what is BTSP-like in the rule and what is deliberately absent.
5. Interpret the stability–plasticity comparison without declaring an
   algorithmic winner.
6. Interpret the CA1 result and make a testable prediction about the relation
   between instructive/plateau activity and later CA1 selectivity.
7. Treat the E7/E9 mismatch as evidence for missing spatially structured
   mechanisms, activity variability, or additional learning pathways.
8. List limitations: abstract units, fixed decoder, artificial data,
   deterministic sparse responses, no recurrent CA3, no timing kernels, and
   task mismatch in the empirical comparison.
9. End with the broader implication: rapid local learning is useful only when
   the learned representation remains coordinated with its consumers.

---

## Proposed paper figures

Existing experiment figures are analysis records, not automatically final
paper figures. Compose final panels from their raw `.npz` source data with one
consistent visual language.

### Figure 1 — Model and representational problem

**Purpose:** establish the architecture, stable coordinate system, and exact
plasticity operation.

- [x] A. EC input → CA3 → CA1 → EC output architecture, with pretrained
  EC→CA1→EC autoencoder pathway and frozen/plastic connections distinguished.
- [x] B. Encoded CA1 target `c = E(x)`, CA3 activity, and row-gated weight
  update.
- [x] C. Aligned versus permuted instructive-signal coordinates relative to a
  fixed decoder.
- [x] D. One representative validated storage/retrieval example.
- [x] State clearly that timing traces are not implemented.

Primary sources: E0, architecture assets, and backend equations.

### Figure 2 — Alignment causally controls decodability

**Purpose:** carry the paper's primary claim.

- [x] A. Five controlled conditions.
- [x] B. Primary chance-corrected decodability with all 20 paired seeds and
  95% confidence intervals.
- [x] C. Top-K recovery or reconstruction endpoint confirming the same result.
- [x] D. Matched-decoder rescue schematic and paired rescue effect.
- [x] E. Controlled partial-misalignment dose response from E4.
- [x] Keep activity/sparsity matching diagnostics in the supplement unless
  required to interpret the main panel.

Primary sources: E1 and the E4 misalignment sweep.

### Figure 3 — Memory function and interference

**Purpose:** show that the alignment principle supports memory behavior rather
than reconstruction of only the current complete input.

- [x] A. Sequential retention matrix.
- [x] B. Memory-age curves for the four core conditions.
- [x] C. Capacity across load, with the threshold defined in the caption.
- [x] D. Masked/bit-flipped cue retrieval and memory identity.
- [x] E. Exact trace survival and crosstalk relationship to recall.
- [x] F. Learning-rate versus forgetting half-life.
- [x] Move the full rule-specific stability–plasticity comparison (E8) to
  Supplementary Figure S4 to keep the main figure focused.

Primary sources: E2a, E2b, E5, and E8.

### Figure 4 — CA1 spatial–sensory representations and biological boundary

**Purpose:** connect the computational mechanism to a controlled CA1-like
task and report both agreement and failure.

- [x] A. Fully crossed cue × position task with cue-free and held-out laps.
- [x] B. Representative held-out CA1 fields/response profiles selected only
  from training laps.
- [x] C. Spatial, cue, and conjunctive population fractions with percentages.
- [x] D. Cue tuning versus held-out remapping.
- [x] E. Published versus model conditional spatial prevalence and log odds
  ratio.
- [x] F. Compact E9 plateau-rate result; retain the full sweep in the
  supplement.

Primary sources: E6, E7, and E9. E3 task validation and legacy field-shift
continuity remain supplementary.

### Proposed supplementary figures

- [ ] S1. Backend validation, deterministic parity, activity, and weight
  diagnostics (E0).
- [ ] S2. Full E1 secondary metrics and matched signal statistics.
- [ ] S3. Capacity/sparsity/size/learning-rate robustness grids (E4).
- [ ] S4. Full rule-baseline development and held-out comparison (E8), if not
  retained in Figure 3.
- [ ] S5. Legacy same-cue field-shift reproduction (E3 legacy).
- [ ] S6. Complete mixed-selectivity classifications and random/no-plasticity
  controls (E6).
- [ ] S7. Full published-data mapping, contingency uncertainty, and E9
  plateau-heterogeneity sweep.

---

## Frozen result register

These values are writing aids, not substitutes for reading the corresponding
JSON report and raw arrays when producing a panel or statistical sentence.

| Result | Independent `n` | Frozen fact | Source |
|---|---:|---|---|
| E1 alignment | 20 paired seeds | Chance-corrected decoding: aligned 0.533, fixed permutation 0.0019, random matched 0.0159; decoder rescue 0.533 | `src/experiments/plots/e1_alignment.{json,npz}` |
| E2a retention | 20 paired seeds | Mean aligned load capacity 8.2 at the frozen threshold; fixed and no-plasticity capacity 0; rescue matches aligned | `src/experiments/plots/e2a_retention.{json,npz}` |
| E2b degraded cues | 20 paired seeds | At 50% masking, aligned identity accuracy 0.594; rescue matches; fixed 0.122 and no-plasticity 0.125 | `src/experiments/plots/e2b_degraded_cues.{json,npz}` |
| E3 factorial track | 12 seeds × 2 paired layouts | With `ae_factorial_paper_v1`, aligned held-out cue accuracy 1.0 and position accuracy 0.875; field stability 0.554; output cosine 0.884 versus fixed 0.114; rescue 0.884 | `src/experiments/plots/e3_ca1_track.{json,npz}` |
| E4 robustness | 20 paired final seeds | Alignment advantage survives the frozen focused parameter grids; use grid-specific values from the report | `src/experiments/plots/e4_sensitivity.{json,npz}` |
| E5 interference | 20 seeds | Trace survival–recall `r = 0.718`; crosstalk–recall `r = -0.711`; recall half-life falls from 34.85 to 8.25 stores as `alpha` rises from 0.12 to 0.55 | `src/experiments/plots/e5_interference_analysis.{json,npz}` |
| E6 selectivity | 12 seeds × 2 paired layouts | Aligned tuned fraction 8.16%: 7.33% conjunctive, 0.575% position-only, 0.254% position-invariant cue; cue-tuning/remapping `rho = 0.996` | `src/experiments/plots/e6_ca1_mixed_selectivity.{json,npz}` |
| E7 empirical anchor | 12 seeds | Published spatial prevalence 65.7% cue-active versus 57.2% cue-inactive; model 96.6% versus 0.62%; OR 1.43 versus 4,041 | `src/experiments/plots/e7_ca1_data_comparison.{json,npz}` |
| E8 rule controls | 20 held-out paired seeds | Alignment/rescue holds for all rules; target-gated ongoing immediate recall 0.943, Hebbian 0.314, delta 0.782; delta half-life 24.25 versus target-gated 13.2 | `src/experiments/plots/e8_rule_baselines.{json,npz}` |
| E9 heterogeneity | 20 fresh seeds × 2 layouts | Low rates slightly reduce error but no rate meets the frozen quantitative-repair rule; output cosine falls from 0.884 at 0% to 0.837 at 40% and 0.669 at 80% | `src/experiments/plots/e9_is_heterogeneity.{json,npz}` |

### Reporting rules

- [ ] Verify every number against the JSON/NPZ immediately before inserting it
  into prose or a caption.
- [ ] Report sample size as independent network/data seeds; describe paired
  layouts, memories, or conditions separately.
- [ ] Report what the interval represents and how it was calculated.
- [ ] Use paired seed-level effects where conditions share seeds.
- [ ] Do not convert cell counts or memories within a seed into independent
  inferential samples.
- [ ] Distinguish prespecified primary endpoints from secondary/descriptive
  analyses.
- [ ] Report negative checks and failed quantitative agreement alongside
  positive findings.
- [ ] Keep displayed precision proportional to uncertainty; usually two or
  three significant digits is enough.

---

## Literature-review brief

The review should support the problem formulation and interpretation, not
retrofit citations to every model component.

### Review questions

#### A. Behavioral-timescale synaptic plasticity

- [ ] What experimental findings define BTSP?
- [ ] Which properties are essential: plateau potentials, eligibility traces,
  seconds-long windows, one-trial field formation, bidirectional weight
  change, initial-weight dependence, asymmetry, or speed dependence?
- [ ] What evidence supports EC layer III involvement in CA1 plateau/instructive
  signals, and how strong is the regional/anatomical specificity?
- [ ] Which claims from the legacy Introduction are oversimplified or wrong?

#### B. CA1 representations and instructive signals

- [ ] How are sensory, contextual, reward, and spatial variables represented
  in CA1?
- [ ] What evidence links dendritic plateaus or EC3 activity to subsequent
  place-field formation and remapping?
- [ ] Which experiments separate cue responsiveness from spatial coding in a
  way comparable to E3/E6?
- [ ] What biological mechanisms could create stable spatial coding outside a
  cue-defined population, as required by the E7/E9 limitation?

#### C. Index and content theories of hippocampal memory

- [ ] How do hippocampal-index accounts describe CA3 and CA1 roles?
- [ ] Which models distinguish content-bearing representations from pointers
  or indices?
- [ ] Where does the current EC–CA3–CA1–EC architecture agree with or depart
  from complementary-learning-systems and associative-memory models?

#### D. Stable readout and representational alignment

- [ ] What theoretical and experimental literature addresses how downstream
  circuits read out changing neural representations?
- [ ] Is “representational alignment,” “decoder compatibility,” “credit
  assignment,” or another term best established for our central problem?
- [ ] What work studies stable behavior despite representational drift, and
  what is genuinely analogous versus merely adjacent?

#### E. Learning-rule and memory comparisons

- [ ] Which local or biologically motivated rules are the fairest conceptual
  comparators for the target-gated overwrite?
- [ ] What literature frames the stability–plasticity dilemma in rapid
  hippocampal learning?
- [ ] Which claims about capacity, interference, and pattern completion require
  established benchmarks or qualification?

### Search and evidence standards

- [ ] Prefer original experimental papers for biological claims and original
  modeling/theory papers for formal claims.
- [ ] Use reviews to map the field, then trace decisive claims to primary
  sources.
- [ ] Record DOI/URL, exact supported claim, relevant figure/page, model or
  species/task, and important caveats.
- [ ] Separate direct evidence from interpretation and from analogy to our
  model.
- [ ] Verify anatomical pathway statements carefully; do not inherit the
  legacy draft's LEC/MEC/EC3 assignments without checking primary sources.
- [ ] Do not cite a paper merely because it uses “BTSP,” “remapping,” or
  “representational drift”; record why it is relevant to a specific sentence.
- [ ] Prefer a compact set of decisive citations over an exhaustive catalogue.

### Seed references already present in the repository — all require review

- Bittner et al. (2017), behavioral-timescale plasticity/place-field formation.
- Milstein et al. (2021), BTSP mechanisms and weight dependence.
- Symanski, Bladon et al. (2022), exact CA1 cue-responsive/spatial-field
  contingency used in E7; DOI `10.7554/eLife.79545`.
- Ito and Schuman (2012) and anatomy references cited in the legacy draft.
- Schapiro et al. (2017), complementary learning systems.
- Pang and Recanatesi (2025), non-Hebbian episodic-memory coding; final
  Science Advances DOI `10.1126/sciadv.ado4112` supersedes the 2024 preprint.

Presence in this list is not approval for citation. Bibliographic metadata,
claims, relevance, and current publication status must be verified during the
review.

### Literature-review deliverables

- [x] `LITERATURE.md` containing an annotated, thematic bibliography.
- [x] An initial claim–citation matrix with one row per literature-dependent manuscript
  claim.
- [ ] A four-paragraph Introduction outline grounded in the verified sources.
- [ ] A short related-model comparison table for the Discussion or supplement.
- [ ] A list of claims removed or weakened after verification.
- [ ] A bibliography file suitable for the final manuscript source.

### Claim–citation intake table

Populate this during the review.

| Manuscript claim | Evidence needed | Candidate source | Direct support? | Caveat | Status |
|---|---|---|---|---|---|
| BTSP can form CA1 place fields after one/few experiences | Primary experiment | TBD | TBD | Species/task/timing | Open |
| BTSP uses seconds-long interactions between eligibility and instructive signals | Primary experiment/model | TBD | TBD | Exact definition | Open |
| EC3 activity contributes to CA1 plateau/field formation | Primary manipulation | TBD | TBD | Pathway specificity | Open |
| CA1 combines spatial and non-spatial variables | Primary population studies | TBD | TBD | Task dependence | Open |
| Downstream readout can remain stable despite changing representations | Primary/theory work | TBD | TBD | Analogy to our model | Open |
| Rapid learning faces a stability–plasticity trade-off | Theory/review plus original models | TBD | TBD | Scope | Open |

---

## Manuscript production checklist

Work in this order unless the decision log records a reason to change it.

### M0 — Freeze the article identity

- [x] Choose a paper rather than thesis format.
- [x] Freeze the simulation scope after E9.
- [x] Adopt “BTSP-inspired” as the default terminology.
- [x] Define the central claim and its strongest causal wording.
- [ ] Select target journal/category and record its length, figure, data, and
  formatting requirements.
- [ ] Confirm authors, affiliations, author order, corresponding author, and
  contribution expectations.
- [ ] Select the final working title after the literature review.

**Done when:** the title, audience, claim, authorship, and submission format fit
on one page without changing the scientific scope.

### M1 — Literature review and Introduction evidence

- [ ] Complete review questions A–E.
- [x] Create the initial annotated bibliography and claim–citation matrix.
- [ ] Verify every anatomical and BTSP-mechanism statement from the legacy
  draft.
- [x] Identify the closest computational models and state the precise novelty
  relative to each.
- [x] Decide whether “representational alignment” is the best established term.
- [x] Produce the four-paragraph Introduction outline and initial prose.

**Done when:** every literature-dependent statement planned for the
Introduction has a verified source or has been removed.

### M2 — Freeze figures and source data

- [x] Finalize which E8 and E9 panels are main versus supplementary.
- [x] Draw the Figure 1 architecture directly from the implemented model.
- [x] Compose Figures 2–4 from frozen raw arrays, not existing raster panels.
- [x] Apply one condition-color dictionary, typography, panel-label, and
  uncertainty style across all figures.
- [x] Show individual seed values where space permits.
- [x] Put sample size and uncertainty definition in every caption.
- [x] Create one machine-readable source-data table per figure.
- [x] Record the command, configuration, and raw-data path for every panel.
- [x] Inspect figures at final print size and in grayscale/colorblind-safe form.

Figure files, panel-level provenance, draft captions, and the single rebuild
command are recorded in `article/figures/README.md`; exact input/output hashes
are recorded in `article/figures/main_figure_manifest.json`.

**Done when:** the complete paper argument is understandable from four figures
and their captions alone.

### M3 — Write Results and captions

- [ ] Write Results subsection 1 from Figure 1.
- [ ] Write Results subsection 2 from Figure 2.
- [ ] Write Results subsections 3–4 from Figure 3.
- [ ] Write Results subsections 5–6 from Figure 4.
- [ ] Lead each subsection with the scientific question and end with the
  supported conclusion.
- [ ] Report effect sizes, uncertainty, `n`, controls, and negative findings.
- [ ] Avoid methods detail that belongs in Methods and interpretation that
  belongs in Discussion.
- [ ] Write complete figure captions immediately after each Results section.

**Done when:** every Results sentence points to a panel/source-data record and
does not exceed the claims allowed above.

### M4 — Rewrite Methods from the frozen implementation

- [ ] Model architecture and dimensions.
- [ ] Sparse activation and all parameter meanings.
- [ ] Autoencoder training data, objective, optimizer, checkpoint selection,
  and frozen weights.
- [ ] CA3 projection and activity generation.
- [ ] Exact target-gated, Hebbian, and delta update equations.
- [ ] Aligned, permutation, random-matched, no-plasticity, and decoder-rescue
  conditions.
- [ ] E1 primary/secondary metrics and chance correction.
- [ ] Sequential retention, capacity threshold, and degraded-cue protocols.
- [ ] Exact trace-survival/interference analysis.
- [ ] Factorial track, held-out split, cell selection, FDR, effect sizes, and
  remapping metrics.
- [ ] Published-data extraction and model-to-data mapping.
- [ ] E9 event schedule and quantitative-repair rule.
- [ ] Seeds, paired designs, confidence intervals, statistical unit, hardware,
  software, and deterministic settings.
- [ ] Data/code availability and reproduction commands.

**Done when:** a reader could reconstruct every figure without reading a
notebook or guessing a parameter.

### M5 — Write Discussion

- [ ] Follow the Discussion outline above.
- [ ] Compare against the closest verified models rather than a broad list.
- [ ] Explain the decoder-rescue result in computational and biological terms.
- [ ] State what E8 says about rule-generality and what it does not say.
- [ ] Treat E7/E9 as a central limitation, not a footnote.
- [ ] State at least one concrete experimental prediction.
- [ ] Separate limitations fixable by parameter/model extensions from those
  requiring a different experimental question.

**Done when:** the Discussion explains importance, relationship to prior work,
predictions, and limitations without making a stronger BTSP or CA1 claim than
the implementation supports.

### M6 — Write Introduction, abstract, and title

- [x] Paragraph 1: rapid learning versus stable readout problem.
- [x] Paragraph 2: BTSP/instructive signals as biological motivation.
- [x] Paragraph 3: missing representational/decoder-alignment question and
  relationship to prior models.
- [x] Paragraph 4: model, causal tests, principal results, and scope.
- [ ] Write the abstract from the six-sentence skeleton after Results and
  Discussion stabilize.
- [ ] Choose the final title and short title.
- [x] Check that novelty language is supported by the completed literature
  review.

**Done when:** the Introduction motivates exactly the problem solved by the
figures, and the abstract contains no claim absent from the Results.

### M7 — Paper-level reproducibility and statistics audit

- [x] Run all 44 tests from the documented environment after the
  distribution-matched autoencoder refreeze.
- [ ] Run the complete final experiment suite from clean commands or verify
  every archived frozen artifact against its configuration and provenance.
- [ ] Confirm all paper panels against raw arrays and source-data tables.
- [ ] Confirm every `n`, confidence interval, threshold, and paired comparison.
- [ ] Audit for pseudoreplication and circular selection.
- [ ] Check decoder permutation notation/orientation against code and a runtime
  identity.
- [ ] Check equations and parameter symbols against the implementation.
- [ ] Freeze a commit/tag and record it in the manuscript and archive manifest.

**Done when:** every number, panel, equation, and conclusion has an auditable
path to frozen code and source data.

### M8 — Internal review and submission package

- [ ] Claim-by-claim audit by the modeling authors.
- [ ] Biological/anatomical review by a domain expert.
- [ ] Statistical and visualization review.
- [ ] Revise scientific issues before copyediting.
- [ ] Check references, nomenclature, abbreviations, and figure callouts.
- [ ] Prepare cover letter, highlights/significance statement if required,
  author contributions, acknowledgements, funding, competing interests, and
  data/code statements.
- [ ] Render and inspect the final manuscript and supplement page by page.
- [ ] Prepare a clean public repository/archive with a persistent identifier.

**Done when:** an external reviewer can read, evaluate, and reproduce the work
without private context.

---

## Paper asset map

| Asset | Role |
|---|---|
| `EXPERIMENTS.md` | Detailed simulation record and experiment-level conclusions |
| `TODO.md` | Repository-wide completion and release checklist |
| `MANUSCRIPT.md` | Paper scope, claims, figures, literature brief, and writing tracker |
| `src/experiments/*.py` | Definitive headless experiment entry points |
| `src/configs/*.json` | Frozen experiment/model configurations |
| `src/experiments/plots/*.json` | Machine-readable reports and provenance |
| `src/experiments/plots/*.npz` | Raw source arrays for paper panels |
| `src/experiments/plots/*.png` / `*.pdf` | Experiment-level analysis figures |
| `article/figures/make_main_figures.py` | Single command that composes the four main figures from frozen arrays |
| `article/figures/figure_*_source_data.csv` | Tidy panel-level source data for each main figure |
| `article/figures/main_figure_manifest.json` | Input/output paths, hashes, and representative-example provenance |
| `article/figures/README.md` | Panel rationale, draft captions, statistics, and main/supplement decisions |
| `src/experiments/reference_data/` | Prespecified external quantitative anchor |
| `tests/` | Backend, parity, metric, and protocol tests |
| `notes/paper/` | Legacy draft; source of ideas only, not authoritative Methods/Results |

---

## Decision log

Record scope-changing choices here so the paper does not drift during writing.

| Date | Decision | Reason | Consequence |
|---|---|---|---|
| 2026-08-16 | Write a focused paper, not a thesis | The completed experiment suite supports one coherent computational article | Remove thesis-style breadth and organize around four main figures |
| 2026-08-16 | Freeze simulations after E9 | Central causal, memory, rule-control, CA1, empirical-anchor, and boundary results are complete | New simulations require a failed claim audit, not curiosity alone |
| 2026-08-16 | Use “BTSP-inspired” by default | The rule lacks explicit eligibility/plateau timing traces and biophysical detail | Limit title, abstract, and Discussion claims accordingly |
| 2026-08-16 | Keep the E7/E9 mismatch visible | Quantitative disagreement defines model scope and prevents biological overclaiming | Include in Figure 4 or a clearly referenced supplement |
| 2026-08-17 | Use “decoder compatibility” for the main principle and reserve “coordinate alignment” for the intervention | Representational alignment is overloaded, and vectorized instruction, BTSP memory, and fixed-readout failure each have prior art | Frame novelty around the permutation plus matched-decoder rescue; do not use an absolute first claim |
| 2026-08-17 | Keep compact E9 in Figure 4 and move full E8 to Supplementary Figure S4 | E9 closes the biological-boundary argument; E8 supports generality but overloads the memory figure | Four main figures now form a complete causal→functional→mechanistic→biological sequence |
| 2026-08-17 | Retrain the spatial–sensory autoencoder before rerunning its dependent experiments | `ae_6` was trained on the older generic sensory generator and underperformed on held-out factorial sensory content | Freeze `ae_factorial_paper_v1` at validation-selected epoch 74; rerun E3, E6, E7, and E9; retain `ae_8` for the separate sparse-pattern suite |

## Open decisions

- [ ] Target journal or article category.
- [ ] Final title.
- [ ] Author list, order, affiliations, and corresponding author.
- [x] Move the full E8 comparison to Supplementary Figure S4.
- [x] Keep a compact E9 boundary panel in Figure 4 and the full sweep in the
  supplement.
- [ ] Final manuscript source format after the initial Markdown drafting pass.
