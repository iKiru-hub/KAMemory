# KAMemory novelty and literature audit

**Status:** focused audit complete, 2026-08-17  
**Purpose:** freeze the article's defensible contribution before writing the
Results and Discussion.  
**Scope:** the closest primary BTSP experiments, hippocampal memory models,
stable-readout work, and vectorized-instruction work. This is a targeted
novelty audit, not a claim that every paper using related terminology has been
exhaustively reviewed.

## Executive conclusion

The paper has a specific, defensible contribution, but it should not claim the
invention of BTSP-based memory, one-shot associative storage, content-addressed
recall, target-like entorhinal instruction, vectorized teaching signals, or the
general distinction between encoded information and fixed-readout performance.
Each of those ingredients has a clear precedent.

The contribution supported by the implemented controls is narrower and
cleaner:

> In a BTSP-inspired hippocampal memory model with a pretrained decoder, the
> coordinate compatibility of a content-bearing instructive signal with that
> decoder causally determines whether rapidly stored content can be read out.
> A coordinate permutation impairs the fixed readout without changing the
> information supplied during learning, and a matched decoder transformation
> restores performance while leaving the learned plastic weights unchanged.

A targeted search of the primary literature reviewed below found work on all
of the constituent ideas, but no prior study using this complete causal design:
a content-bearing instructive signal, a pre-existing fixed decoder, an
instructive-coordinate permutation, and a matched decoder rescue that holds
the learned representation fixed. The manuscript should phrase this as the
specific gap addressed by the study, not as an absolute "first" claim.

## Frozen language

### Recommended primary claim

> Decoder compatibility is a causal determinant of content decodability in
> this BTSP-inspired hippocampal memory model.

### Recommended novelty statement

> Previous studies have separately shown that BTSP-like rules can support
> rapid associative storage, that target-like or neuron-specific instructive
> signals can shape neural activity, and that representational change can
> degrade fixed readouts. Here these ideas are connected by a matched causal
> control that changes the coordinate relationship between instruction and a
> pretrained decoder while preserving the learned plastic weights.

### Terminology decision

Use **decoder compatibility** for the central computational principle and
**instructive-signal coordinate alignment** for the operational manipulation.
Use **representational alignment** only with a definition: in other literatures
it can refer to manifold registration, cross-subject hyperalignment,
communication subspaces, or drift relative to a coding subspace. It is
therefore too overloaded to name the principal result by itself.

### Claims to avoid

- "The first vectorized instructive signal." Francioni et al. provide direct
  cortical evidence for neuron-specific vectorized dendritic instruction.
- "The first BTSP memory model" or "the first BTSP model with downstream
  decoding." Wu and Maass already demonstrate one-shot content-addressable
  memory and generic downstream classification.
- "The first stable-readout account." Rule et al. explicitly distinguish
  information retained in a drifting population from performance of a fixed
  decoder.
- "A complete model of BTSP." The present rule omits measured eligibility and
  plateau time courses, dendritic compartments, and full weight-dependent
  bidirectionality.
- "The model proves that biological CA1 uses a pretrained decoder coordinate
  system." The stable decoder and encoded target are modeling assumptions.
- "Coordinate alignment is universally necessary for memory." Necessity is
  established only under the controlled architecture and manipulations tested.

## Audit of the implemented causal experiment

The primary implementation is
[`src/experiments/01_alignment.py`](../src/experiments/01_alignment.py). The
cross-rule replication is
[`src/experiments/09_rule_baselines.py`](../src/experiments/09_rule_baselines.py).

Let the pretrained encoder produce a target CA1 code `c = E(x)` and let `D`
be the frozen CA1-to-output decoder. With the indexing convention used in the
code, a coordinate permutation is `c' = c[perm]`.

| Condition | Signal used for storage | Decoder | Interpretation |
|---|---|---|---|
| Aligned | `c` | `D` | Instruction and readout use the same coordinates. |
| Fixed permutation | `c[perm]` | `D` | Information and signal statistics are preserved, but coordinates do not match the readout. |
| Matched decoder rescue | `c[perm]` | `D[:, perm]` | Storage is identical to fixed permutation; only the readout is transformed consistently. |
| Random matched signal | Another memory's code, with no self-matches | `D` | Preserves the code distribution while breaking content identity. |
| No plasticity | None | `D` | Establishes the untrained retrieval floor. |

The code checks the decoder orientation at runtime. For a probe vector `z`, it
verifies that decoding `z` with `D` equals decoding `z[perm]` with
`D[:, perm]`. It also verifies numerically that fixed-permutation and rescue
conditions finish with identical CA3-to-CA1 plastic weights. Thus the rescue
does not improve storage; it changes only which CA1 coordinate each fixed
decoder weight reads.

In the frozen E1 run (`n = 20` independent network/data seeds, 28 memories per
seed), mean chance-corrected content cosine was 0.533 for aligned instruction,
0.002 for the fixed permutation, 0.016 for random matched instruction, and
0.533 for the matched decoder rescue. The aligned--fixed effect was positive
in the archived paired checks, signal statistics were matched, the
fixed/rescue weights were identical, and the legacy notebook result was
reproduced to a maximum absolute error of `4.26e-9`. These values are recorded
in [`src/experiments/plots/e1_alignment.json`](../src/experiments/plots/e1_alignment.json).

### What the rescue establishes

1. The failure under permutation is not caused by less informative or less
   sparse instructive signals: a permutation is an invertible relabeling.
2. It is not caused by different plasticity between fixed and rescue
   conditions: their inputs, targets, and final plastic weights are identical.
3. It is not merely a metric artifact in CA1 space: applying the matched
   transformation to the actual downstream decoder restores output content.
4. Under these controls, the remaining causal difference is compatibility
   between the coordinates installed by learning and those read by the
   decoder.

### What the rescue does not establish

1. It does not show that biological instructive pathways literally transmit
   autoencoder coordinates or permutations.
2. It does not identify which anatomical circuit learns or stabilizes the
   decoder.
3. It does not test arbitrary nonlinear reparameterizations; the clean causal
   intervention is a coordinate permutation.
4. It does not reproduce the seconds-long BTSP kernel or its molecular and
   dendritic mechanisms.
5. It does not show that a decoder could not adapt. Rule et al. show that local
   decoder plasticity can compensate gradual drift; this paper instead asks
   what rapid storage can accomplish while the decoder is held fixed.

## Closest-paper comparison

### Bittner et al. (2017)

**Contribution.** In vivo and slice experiments established that dendritic
plateau potentials can produce CA1 place fields in one trial by modifying
inputs active over seconds, including inputs that were not coincident with
postsynaptic spiking. The work introduced BTSP as a non-Hebbian rule suited to
storing behavioral sequences.

**Most relevant evidence.** Figures 1--3: abrupt in-vivo field formation,
seconds-long and asymmetric plasticity, and the eligibility/instructive-signal
account; Figure 3 includes the slice induction measurements.

**Relation to this project.** It motivates rapid, plateau/instruction-gated
CA3-to-CA1 learning.

**Critical difference.** It does not ask whether the induced CA1 coordinates
are compatible with a pre-existing content decoder.

**Claim it supports.** The update may be described as **BTSP-inspired** and
one-shot, not as a faithful BTSP implementation.

**Source.** [Science, DOI 10.1126/science.aan3846](https://doi.org/10.1126/science.aan3846).

### Milstein et al. (2021)

**Contribution.** Experiments and modeling showed that BTSP bidirectionally
reshapes existing place fields: weak inputs potentiate, strong inputs depress,
and the magnitude and direction depend on timing and initial synaptic weight
rather than postsynaptic firing alone.

**Most relevant evidence.** Figures 1--3 establish field translocation and
bidirectional changes; Figures 5--6 define and validate the weight-dependent
eligibility/instructive-signal model; Figure 7 explores population adaptation
when plateau probability reflects mismatch between local output and target
feedback.

**Relation to this project.** It is the closest mechanistic precedent for a
target-directed BTSP population model.

**Critical difference.** Its target-feedback proposal controls plateau
probability and place-field adaptation, not placement in a pretrained content
decoder's coordinate system. It has no coordinate permutation or matched
readout rescue.

**Claim it supports.** Target-directed BTSP is biologically and computationally
motivated, but this project's simpler update omits major measured features.

**Source.** [eLife, DOI 10.7554/eLife.73046](https://doi.org/10.7554/eLife.73046).

### Grienberger and Magee (2022)

**Contribution.** CA1 reward over-representation developed with BTSP-like
signatures; perturbing EC3 prevented it. EC3 activity tracked salient
environmental structure, and a comparator model proposed that plateau
probability reflects the difference between an EC3 target and CA1 feedback.

**Most relevant evidence.** Figure 1, CA1 reward over-representation; Figure
2, abrupt field formation, backward shifts, speed--width relation, and
pharmacology; Figure 3, EC3 necessity; Figures 4--5, EC3 activity structure;
Figure 6, target/comparator account.

**Relation to this project.** This is the strongest hippocampal precedent for
an entorhinal signal that directs CA1 learning toward a target population
profile.

**Critical difference.** The target is a task-dependent spatial activity
profile, not an explicitly learned content code tied to a stable decoder, and
the paper does not perform a coordinate/readout rescue.

**Claim it supports.** Entorhinal target-like instruction is a plausible
biological motivation. It does not justify identifying the model's `E(x)` with
raw EC3 activity.

**Source.** [Nature, DOI 10.1038/s41586-022-05378-6](https://doi.org/10.1038/s41586-022-05378-6).

### Wu and Maass (2025)

**Contribution.** A simplified stochastic BTSP model with binary synapses
supports one-shot, content-addressable memory, partial/noisy-cue retrieval,
and robust memory traces. The work also evaluates generic downstream linear
classification.

**Most relevant evidence.** Figure 1, simplified rule and architecture;
Figure 2, storage and cue-based recall; Figure 3, comparisons with random
projections/Hopfield-style alternatives and downstream classification.

**Relation to this project.** It is the closest BTSP-inspired associative
memory comparator and prevents novelty claims based only on one-shot storage,
degraded cues, or downstream decodability.

**Critical difference.** Plateaus select memory traces rather than provide a
content-target vector in a pretrained decoder's coordinates. Downstream
classification is not used to test instruction/readout compatibility.

**Claim it supports.** The novel element here is the matched decoder causal
control, not BTSP-inspired content-addressable memory itself.

**Source.** [Nature Communications, DOI 10.1038/s41467-024-55563-6](https://doi.org/10.1038/s41467-024-55563-6).

### Pang and Recanatesi (2025)

**Contribution.** A presynaptic-only, one-factor rule stores an episode as a
path through a pre-existing high-dimensional world-model representation. A
familiarity signal can guide retrieval, including arbitrary serial-order and
item--position associations.

**Most relevant evidence.** Figure 2, path storage and retrieval in an
existing representational space; Figure 5, serial-order and item--position
associations.

**Relation to this project.** Both studies ask how rapid local plasticity can
exploit a pre-existing representation for episodic memory.

**Critical difference.** The rule has no content-bearing postsynaptic target,
and its path-following decoder is assumed rather than subjected to a fixed
coordinate mismatch and inverse rescue.

**Claim it supports.** Pre-existing representational structure can organize
rapid memory without establishing this paper's decoder-compatibility result.

**Source.** [Science Advances, DOI 10.1126/sciadv.ado4112](https://doi.org/10.1126/sciadv.ado4112).

### Chandra et al. (2025)

**Contribution.** Vector-HaSH factors high-dimensional content storage from an
error-correcting spatial scaffold, yielding high-capacity associative,
sequential, spatial, and memory-palace functions.

**Most relevant evidence.** Figure 1, architecture and computational
challenges; Figures 2--3, scaffold dynamics and content-addressable memory;
Figure 5, episodic sequences; Figures 6--7, hippocampal phenomena and memory
palaces.

**Relation to this project.** It is a strong contemporary example of
heteroassociative hippocampal memory that explicitly separates content from an
organizing latent structure.

**Critical difference.** It uses a broader scaffolded circuit and learned
heteroassociations rather than a BTSP-like instructive signal. It does not
isolate compatibility with a frozen content decoder by coordinate rescue.

**Claim it supports.** The project should be framed as a focused
representational/readout principle, not as a comprehensive hippocampal memory
architecture.

**Source.** [Nature, DOI 10.1038/s41586-024-08392-y](https://doi.org/10.1038/s41586-024-08392-y).

### Rule et al. (2020)

**Contribution.** Posterior parietal task information remained available
despite representational drift, but single-day fixed decoders degraded across
days. Multi-day readouts were more stable, modest decoder weight changes
improved performance, and an online local least-mean-square rule could track
the drift.

**Most relevant evidence.** Figure 3, degradation of single-day fixed readouts
and multi-day decoders; Figure 4, drift relative to behavior/noise subspaces
and the plasticity--accuracy trade-off; Figure 5, local adaptive-decoder
compensation.

**Relation to this project.** It is the closest conceptual precedent for the
distinction between information present in a population and compatibility
with a fixed downstream readout.

**Critical difference.** It analyzes gradual cortical drift across days and
compensatory plasticity in the decoder. This project manipulates the teaching
coordinates during rapid hippocampal storage and holds the decoder fixed,
then uses a matched transformation as a causal rescue.

**Claim it supports.** Decodability by a newly fitted or adaptive decoder does
not imply readability by a particular fixed decoder.

**Source.** [eLife, DOI 10.7554/eLife.51121](https://doi.org/10.7554/eLife.51121).

### Dorian et al. (2026)

**Contribution.** Rare plateau-like calcium events preceded stable CA1 odor
responses in an odor-cued task, and holographic induction in single neurons
causally generated new odor responses. MEC and LEC manipulations differentially
affected event occurrence and odor-field formation.

**Most relevant evidence.** Figure 1, spontaneous events preceding odor
fields; Figure 2, causal single-cell induction; later figures, learning-stage
and entorhinal-pathway analyses.

**Relation to this project.** It provides the strongest direct bridge from
spatial BTSP to the model's non-spatial and mixed-content motivation.

**Critical difference.** It establishes causal formation of sensory responses,
not their compatibility with a stable downstream content decoder.

**Claim it supports.** A BTSP-inspired account need not be restricted to place
fields, although the model is not a quantitative reconstruction of this task.

**Source.** [Nature Communications, DOI 10.1038/s41467-026-71503-y](https://doi.org/10.1038/s41467-026-71503-y).

### Francioni et al. (2026)

**Contribution.** In a retrosplenial-cortex neurofeedback task, soma--dendrite
residual signals contained reward and error information with neuron-specific
signs determined by each neuron's causal role. Targeted optogenetic disruption
impaired learning, providing direct evidence for vectorized dendritic teaching
signals.

**Most relevant evidence.** Figures 1--3, task and soma/dendrite residual
analysis; Figure 4, reward and outcome information; Figure 5, cell-specific
error signs and causal perturbation.

**Relation to this project.** It makes neuron-specific, vectorized instruction
a biologically grounded concept rather than merely an artificial-network
analogy.

**Critical difference.** It concerns cortical credit assignment and target/error
signals, not rapid hippocampal associative storage, a pretrained content
decoder, or coordinate permutation and rescue.

**Claim it supports.** The manuscript may motivate a vector-valued instruction,
but must not claim novelty for vectorization itself.

**Source.** [Nature, DOI 10.1038/s41586-026-10190-7](https://doi.org/10.1038/s41586-026-10190-7).

## Feature matrix

Legend: **yes** = explicit central test; **partial** = related but not the same
operation; **no** = absent from the reported study.

| Study | BTSP / rapid local storage | Content- or target-bearing instruction | Pre-existing decoder | Fixed-readout failure | Coordinate permutation | Matched decoder rescue |
|---|---:|---:|---:|---:|---:|---:|
| Bittner 2017 | yes | no | no | no | no | no |
| Milstein 2021 | yes | partial | no | no | no | no |
| Grienberger & Magee 2022 | yes | partial | no | no | no | no |
| Wu & Maass 2025 | yes | no | partial | no | no | no |
| Pang & Recanatesi 2025 | yes | no | partial | no | no | no |
| Chandra et al. 2025 | yes | no | partial | no | no | no |
| Rule et al. 2020 | no | no | yes | yes | no | no |
| Dorian et al. 2026 | yes | partial | no | no | no | no |
| Francioni et al. 2026 | no | yes | no | no | no | no |
| KAMemory | BTSP-inspired | yes | yes | yes | yes | yes |

The matrix exposes the synthesis accurately: the components are not new, but
their combination into a controlled decoder-compatibility test appears to be.

## Claim-to-source map for the Introduction

| Introduction statement | Direct source | Boundary applied in the draft |
|---|---|---|
| Fast hippocampal learning must coexist with slower knowledge | Marr 1971; McClelland et al. 1995; Schapiro et al. 2017 | Theory framing, not evidence that the model's decoder is anatomically fixed. |
| BTSP forms fields rapidly over seconds | Bittner et al. 2017 | The model is called BTSP-inspired. |
| BTSP is timing- and weight-dependent and bidirectional | Milstein et al. 2021 | These omitted features are acknowledged rather than attributed to the implementation. |
| EC3 can direct CA1 learning toward salient target profiles | Grienberger & Magee 2022 | `E(x)` is not equated with raw EC3 activity. |
| Plateau-like events can create non-spatial CA1 responses | Dorian et al. 2026 | Used as motivation, not a claimed quantitative match. |
| Dendritic instruction can be neuron-specific/vectorized | Francioni et al. 2026 | Novelty is not claimed for vectorization. |
| BTSP-like and hippocampal models already support memory | Wu & Maass 2025; Pang & Recanatesi 2025; Chandra et al. 2025 | Novelty is restricted to decoder compatibility and rescue. |
| Information can persist while fixed-readout performance falls | Rule et al. 2020 | Conceptual analogy from PPC drift, not hippocampal evidence for this mechanism. |
| The present permutation/rescue isolates decoder compatibility | This study, E1 and E8 | Causal only within the tested model. |

## Search outcome and uncertainty

The audit explicitly searched for combinations of the terms *instructive* or
*teaching signal*, *fixed decoder/readout*, *coordinate permutation*, *matched
decoder*, *inverse rescue*, *hippocampus*, and *BTSP*. It also followed the
closest primary citations found from BTSP target-learning, associative-memory,
representational-drift, and vectorized-instruction papers. No exact predecessor
for the complete control was found.

Negative literature searches cannot prove nonexistence. Before submission, the
corresponding author or a domain expert should repeat this focused search and
check citing/cited-by networks for the five closest conceptual papers:
Grienberger and Magee (2022), Wu and Maass (2025), Pang and Recanatesi (2025),
Rule et al. (2020), and Francioni et al. (2026). Unless an exact predecessor is
found, the manuscript should retain the conservative wording "we address" or
"we test," not "we are the first."

## Consequences for the paper

1. Keep the working title's emphasis on **decoder-aligned instructive signals**.
2. Let Figure 2/E1 carry the novelty: aligned, fixed permutation, random
   matched, no-plasticity, and matched decoder rescue must be visible together.
3. Explain explicitly that fixed and rescue conditions learn identical
   CA3-to-CA1 weights. This is the logic that turns a performance comparison
   into a causal readout test.
4. Treat E2/E3 and degraded-cue results as functional consequences, not the
   primary novelty.
5. Use E8 to show rule-generality of the alignment effect while reporting
   differences in stability--plasticity between rules.
6. Keep the E7/E9 biological mismatch prominent so that the abstract cannot be
   read as a quantitative CA1 model claim.
7. Put the detailed related-model table in the Discussion or supplement; use
   only the compact conceptual gap in the Introduction.

## Files produced or updated by this audit

- [`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md): frozen novelty, comparison table,
  causal interpretation, and claim boundaries.
- [`maintex/main.tex`](maintex/main.tex): four-paragraph Introduction based on
  the verified citation spine.
- [`maintex/references.bib`](maintex/references.bib): added the 2026 Francioni
  et al. record without cleaning or removing existing entries.
- [`LITERATURE.md`](LITERATURE.md) and [`MANUSCRIPT.md`](MANUSCRIPT.md): tracker
  updates marking this focused audit and Introduction draft complete.

## Next manuscript task

Freeze the four-figure panel map and source-data manifest (M2), starting with
the E1 causal figure. The Introduction should remain provisional until the
Results captions have fixed the exact reported numbers and terminology.
