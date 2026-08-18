# KAMemory literature map

This is a working literature record for the paper described in
[`MANUSCRIPT.md`](MANUSCRIPT.md). It is not intended to be an exhaustive review
of hippocampal memory. Its purpose is to identify which papers support each
part of the argument, distinguish central references from background reading,
and record gaps in [`references.bib`](maintex/references.bib) before manuscript
prose is written.

Last focused novelty audit: **2026-08-17**. See
[`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md) for the paper-by-paper comparison and
frozen claim language.

## Reading and citation labels

- **Core** — expected in the Introduction or central Discussion.
- **Support** — useful for a specific biological, computational, or methods
  statement.
- **Context** — relevant background, but probably unnecessary in the short
  Introduction.
- **Add** — relevant paper that is not yet represented correctly in
  `references.bib`.
- **Verified** — metadata and stated contribution checked against a publisher,
  PubMed, or journal page; this does not imply that every analysis in the full
  2026 article has already been independently evaluated.

## 1. Behavioral-timescale synaptic plasticity

These papers define what is genuinely BTSP-like in the project and, equally
importantly, what the present activity-level update does not implement.

- **Core · Verified — Bittner et al. (2017), “Behavioral time scale synaptic
  plasticity underlies CA1 place fields”** (`bittner2017`;
  [DOI](https://doi.org/10.1126/science.aan3846)) — Foundational evidence that a
  single dendritic plateau can create a CA1 place field by modifying inputs over
  seconds, motivating one-shot, non-Hebbian learning in the model.

- **Core · Verified — Milstein et al. (2021), “Bidirectional synaptic plasticity
  rapidly modifies hippocampal representations”** (`milstein2021`;
  [DOI](https://doi.org/10.7554/eLife.73046)) — Shows that BTSP can potentiate
  weak and depress strong inputs as a function of initial synaptic weight and
  timing, making it the key citation for why the implemented target-gated rule
  is BTSP-inspired rather than a complete BTSP rule.

- **Core · Verified — Grienberger and Magee (2022), “Entorhinal cortex directs
  learning-related changes in CA1 representations”** (`grienberger2022`;
  [DOI](https://doi.org/10.1038/s41586-022-05378-6)) — Provides the strongest
  experimental precedent for an entorhinal target-like instructive signal that
  guides CA1 population activity during learning.

- **Core · Verified — Magee (2026), “Behavioral timescale synaptic plasticity:
  properties, elements and functions”** (`magee2026`;
  [DOI](https://doi.org/10.1038/s41593-026-02214-2)) — Current synthesis of
  BTSP’s seconds-long eligibility, plateau-based induction, bidirectionality,
  circuit control, and possible memory functions; use as a map to primary work,
  not as the only evidence for individual claims.

- **Core · Verified — Madar et al. (2025), “Synaptic plasticity rules driving
  representational shifting in the hippocampus”** (`madar2025`;
  [DOI](https://doi.org/10.1038/s41593-025-01894-6)) — Finds that BTSP-like rules
  explain trial-by-trial place-field shifts better than STDP and links rare
  plasticity events to ongoing representational drift, directly informing E3
  and the stability–plasticity discussion.

- **Core · Verified — Vaidya et al. (2025), “Formation of an expanding memory
  representation in the hippocampus”** (`vaidya2025`;
  [DOI](https://doi.org/10.1038/s41593-025-01986-3)) — Longitudinal CA1 data
  suggest that stable task representations can emerge through repeated,
  history-dependent place-field re-formation rather than permanent freezing of
  all synapses, providing a useful biological counterpoint to the fixed decoder.

- **Support · Verified — Li et al. (2024), “Mechanisms of memory-supporting
  neuronal dynamics in hippocampal area CA3”** (`li2024`;
  [DOI](https://doi.org/10.1016/j.cell.2024.09.041)) — Reports symmetric BTSP at
  recurrent CA3 synapses and an entorhinal updating input in an online attractor
  model, supporting the plausibility of BTSP-like learning beyond CA1 while
  underscoring that this project simplifies CA3 dynamics.

- **Support · Add · Verified — Jain et al. (2024), “Dendritic, delayed,
  stochastic CaMKII activation in behavioural time scale plasticity”**
  ([DOI](https://doi.org/10.1038/s41586-024-08021-8)) — Identifies a delayed,
  stochastic dendritic CaMKII process required for BTSP, useful for explicitly
  delimiting molecular mechanisms omitted from the model.

- **Support — Golding et al. (1999), “Dendritic calcium spike initiation and
  repolarization…”** (`golding1999`;
  [DOI](https://doi.org/10.1523/JNEUROSCI.19-20-08789.1999)) — Establishes basic
  CA1 dendritic calcium-spike physiology relevant to plateau generation, but it
  is background rather than evidence for the model’s representational claim.

- **Support — Golding et al. (2002), “Dendritic spikes as a mechanism for
  cooperative long-term potentiation”** (`golding2002`;
  [DOI](https://doi.org/10.1038/nature00854)) — Connects dendritic spikes to
  cooperative LTP and provides mechanistic context for strong dendritic events.

## 2. Non-spatial signals, entorhinal instruction, and CA1 mixed selectivity

This group anchors the spatial–sensory experiments and prevents the paper from
implying that BTSP or CA1 coding is exclusively spatial.

- **Core · Add · Verified — Dorian et al. (2026), “Rapid formation of
  non-spatial hippocampal representations consistent with behavioral timescale
  synaptic plasticity is modulated by entorhinal input”**
  ([DOI](https://doi.org/10.1038/s41467-026-71503-y)) — Plateau-like events and
  causal single-cell stimulation produce stable odor responses in CA1, while
  MEC and LEC differentially affect event frequency and odor-field formation;
  this is the most direct experimental bridge to E3/E6.

- **Core · Add · Verified — Symanski, Bladon et al. (2022), “Rhythmic
  coordination and ensemble dynamics in the hippocampal–prefrontal network
  during odor-place associative memory and decision making”**
  ([DOI](https://doi.org/10.7554/eLife.79545)) — Supplies the exact CA1
  odor-responsive/spatial-field contingency used as the prespecified empirical
  benchmark in E7 and E9; cite it only for the measured association, not as
  evidence for BTSP.

- **Support — Bilash et al. (2023), “Lateral entorhinal cortex inputs modulate
  hippocampal dendritic excitability…”** (`bilash2023`;
  [DOI](https://doi.org/10.1016/j.celrep.2022.111962)) — Shows that LEC inputs
  can gate CA1 dendritic nonlinearities through local disinhibitory and
  inhibitory microcircuits, supporting a cautious sensory-input route to CA1.

- **Support — Henriksen et al. (2010), “Spatial representation along the
  proximodistal axis of CA1”** (`henriksen2010`;
  [DOI](https://doi.org/10.1016/j.neuron.2010.08.042)) — Demonstrates structured
  variation in CA1 spatial coding along the proximodistal axis, warning against
  treating biological CA1 as a homogeneous population.

- **Support — Ito and Schuman (2012), “Functional division of hippocampal area
  CA1 via modulatory gating of entorhinal cortical inputs”** (`ito2012`;
  [DOI](https://doi.org/10.1002/hipo.20909)) — Reviews how entorhinal input and
  inhibition may route distinct information streams through CA1, relevant to
  the model’s spatial and cue pathways.

- **Support — Soltesz and Losonczy (2018), “CA1 pyramidal cell diversity
  enabling parallel information processing in the hippocampus”**
  (`soltesz2018`; [DOI](https://doi.org/10.1038/s41593-018-0118-0)) — Reviews
  anatomical and functional CA1 heterogeneity, a key limitation of the model’s
  exchangeable artificial CA1 units.

- **Support — Mizuseki et al. (2012), “Activity dynamics and behavioral
  correlates of CA3 and CA1 hippocampal pyramidal neurons”** (`mizuseki2012`;
  [DOI](https://doi.org/10.1002/hipo.22002)) — Large-scale recordings establish
  distinct CA3/CA1 firing and place-field properties, supporting the decision
  to discuss the two regions as computationally non-equivalent.

- **Context — Lee and Han (2023), “Activity patterns of individual neurons and
  ensembles correlated with retrieval of a contextual memory…”** (`lee2023`;
  [DOI](https://doi.org/10.1523/JNEUROSCI.1407-22.2022)) — Links learned-context
  CA1 activity and ensemble synchrony to memory strength, useful background for
  interpreting population-level retrieval.

## 3. Fast associative memory and related computational models

These papers locate the model relative to existing theories. The novelty claim
should be about **decoder-coordinate alignment and matched-decoder rescue**, not
about being the first hippocampal associative-memory model or the first
BTSP-inspired memory model.

- **Core · Verified — Wu and Maass (2025), “A simple model for BTSP provides
  content addressable memory with binary synapses and one-shot learning”**
  (`wu2025`; [DOI](https://doi.org/10.1038/s41467-024-55563-6)) — The closest
  direct computational comparator: it derives one-shot content-addressable
  memory from stochastic plateau allocation and binary synapses, whereas this
  project tests compatibility with a separately trained stable decoder.

- **Core · Add · Verified — Pang and Recanatesi (2025), “A non-Hebbian code for
  episodic memory”**
  ([DOI](https://doi.org/10.1126/sciadv.ado4112)) — Shows that a presynaptic-only
  rule can store episodes as decodable high-dimensional path vectors and
  support one-shot sequential and associative recall, making it a close
  rule-level comparator without the present model’s instructive-signal/decoder
  alignment test.

- **Core · Verified — Chandra et al. (2025), “Episodic and associative memory
  from spatial scaffolds in the hippocampus”** (`chandra2025`;
  [DOI](https://doi.org/10.1038/s41586-024-08392-y)) — Vector-HaSH factors
  content storage from an error-correcting spatial scaffold to support
  high-capacity associative and episodic memory, making it an important modern
  comparator with a different architectural solution.

- **Core · Add · Verified — Marr (1971), “Simple memory: a theory for
  archicortex”** ([DOI](https://doi.org/10.1098/rstb.1971.0078)) — Classic
  computational foundation for rapid hippocampal associative storage and
  sparse representations.

- **Core · Add · Verified — Teyler and DiScenna (1986), “The hippocampal memory
  indexing theory”** ([DOI](https://doi.org/10.1037/0735-7044.100.2.147)) —
  Frames hippocampal activity as an index capable of reinstating distributed
  cortical content, directly relevant to distinguishing an address-like index
  from the model’s content-readable CA1 code.

- **Core · Add — McClelland, McNaughton, and O’Reilly (1995), “Why there are
  complementary learning systems in the hippocampus and neocortex”**
  ([DOI](https://doi.org/10.1037/0033-295X.102.3.419)) — Establishes the
  stability–plasticity motivation for fast hippocampal learning alongside slow
  cortical learning, the broad systems problem captured by a plastic memory
  pathway and stable decoder.

- **Support · Add · Verified — Schapiro et al. (2017), “Complementary learning
  systems within the hippocampus…”**
  ([DOI](https://doi.org/10.1098/rstb.2016.0049)) — Models a division in which
  the trisynaptic pathway rapidly stores episodes while the monosynaptic
  entorhinal–CA1 pathway extracts regularities, providing architectural context
  but not a direct precedent for the current decoder-alignment mechanism.

- **Support — Spens and Burgess (2024), “A generative model of memory
  construction and consolidation”** (`spens2024`;
  [DOI](https://doi.org/10.1038/s41562-023-01799-z)) — Combines hippocampal
  autoassociation and replay with cortical generative models, useful for
  contrasting immediate retrieval in this paper with longer-term consolidation.

- **Support — Wen et al. (2024), “One-shot entorhinal maps enable flexible
  navigation in novel environments”** (`wen2024`;
  [DOI](https://doi.org/10.1038/s41586-024-08034-3)) — Shows how fixed
  landmark-to-grid structure and downstream BTSP-like plasticity can jointly
  balance rapidity and representational accuracy, conceptually close to the
  paper’s fixed/plastic division.

- **Context — Hasselmo et al. (2020), “Overview of computational models of
  hippocampus and related structures”** (`hasselmo2020`;
  [DOI](https://doi.org/10.1002/hipo.23201)) — Broad survey useful for orienting
  the Discussion, but too general to support a specific novelty claim.

## 4. Stable readout, representational geometry, and drift

This is the largest conceptual gap in the current bibliography. These papers
do not establish the project’s causal result, but they provide the vocabulary
needed to explain why a representation can contain information yet fail to be
readable by a fixed downstream mapping.

- **Core · Verified — Francioni et al. (2026), “Vectorized instructive signals
  in cortical dendrites”** (`francioni2026`;
  [DOI](https://doi.org/10.1038/s41586-026-10190-7)) — Provides direct cortical
  evidence for neuron-specific dendritic reward/error signals whose signs
  depend on each neuron's causal role, and shows that targeted perturbation
  impairs learning. It prevents a novelty claim for vectorized instruction
  itself; the present contribution is the decoder-compatibility test in a
  BTSP-inspired hippocampal memory model.

- **Core · Add · Verified — Rule et al. (2020), “Stable task information from
  an unstable neural population”**
  ([DOI](https://doi.org/10.7554/eLife.51121)) — Shows that representational
  drift can remain constrained enough for linear decoding while still
  degrading a fixed readout, making it the closest precedent for separating
  information content from fixed-decoder compatibility.

- **Support · Add · Verified — Gallego et al. (2020), “Long-term stability of
  cortical population dynamics underlying consistent behavior”**
  ([DOI](https://doi.org/10.1038/s41593-019-0555-4)) — Finds stable latent
  dynamics despite turnover in recorded motor-cortical neurons, demonstrating
  that coordinate alignment at a population level can matter more than
  single-unit stability.

- **Support · Add · Verified — Schoonover et al. (2021), “Representational
  drift in primary olfactory cortex”**
  ([DOI](https://doi.org/10.1038/s41586-021-03628-7)) — Provides a clear
  empirical example in which sensory representations drift and a fixed
  day-specific classifier loses accuracy, useful as general problem framing
  rather than hippocampus-specific evidence.

- **Support — Zou et al. (2023), “Re-expression of CA1 and entorhinal activity
  patterns preserves temporal context memory at long timescales”** (`zou2023`;
  [DOI](https://doi.org/10.1038/s41467-023-40100-8)) — Human 7-T fMRI links
  later temporal-context memory to reinstatement of CA1/entorhinal patterns,
  supporting the broader relevance of representational compatibility across
  encoding and retrieval.

## 5. Functional CA1/CA3 memory evidence

These sources may support a compact paragraph on hippocampal memory function or
specific Discussion points. They should not crowd the mechanistic opening of
the Introduction.

- **Support — Atucha et al. (2023)** (`atucha2023`;
  [DOI](https://doi.org/10.1016/j.celrep.2023.113317)) — Dissociates CA1 support
  for long-lived gist from CA3 support for memory precision, relevant to the
  project’s CA3-to-CA1 transformation but not a direct validation of it.

- **Support — Kolibius et al. (2023)** (`kolibius2023`;
  [DOI](https://doi.org/10.1038/s41562-023-01706-6)) — Human single-neuron data
  show hippocampal codes for individual episodic memories, supporting the
  premise that content-specific population representations can accompany
  episodic retrieval.

- **Support — Sans-Dublanc et al. (2020)** (`sans-dublanc2020`;
  [DOI](https://doi.org/10.1126/sciadv.aba5003)) — Demonstrates circuit-level
  gating of CA1-dependent contextual memory retrieval and reminds us that a
  fixed feed-forward decoder omits retrieval-state control.

- **Context — Hunsaker et al. (2008)** (`hunsaker2008`;
  [DOI](https://doi.org/10.1016/j.bbr.2007.11.015)) — Lesion results distinguish
  CA3 and CA1 contributions to temporal-context processing.

- **Context — Hoang and Kesner (2008)** (`hoang2008`;
  [DOI](https://doi.org/10.1037/0735-7044.122.1.9)) — Shows that dorsal
  hippocampal, CA3, and CA1 lesions impair temporal sequence completion.

- **Context — Ji and Maren (2008)** (`ji2008`;
  [DOI](https://doi.org/10.1101/lm.794808)) — Provides a CA1/CA3 dissociation in
  contextual encoding and retrieval of extinguished fear.

- **Context — Olarte-Sánchez et al. (2014)** (`olarte-sanchez2014`;
  [DOI](https://doi.org/10.1037/a0037055)) — Identifies distinct
  perirhinal–entorhinal–hippocampal networks for recognition and recency,
  including a prominent CA1 role in temporal discrimination.

- **Context — Albasser et al. (2012)** (`albasser2012`;
  [DOI](https://doi.org/10.1037/a0029754)) — Hippocampal lesions spare simple
  object recognition but impair object-recency judgments, useful only if the
  paper discusses temporal-order memory.

- **Context — Bartsch et al. (2011)** (`bartsch2011`;
  [DOI](https://doi.org/10.1073/pnas.1110266108)) — Focal human CA1 lesions
  impair autobiographical recollection, broadening biological relevance but
  lying far from the model’s tested mechanisms.

- **Context — Jeong et al. (2018)** (`jeong2018`;
  [DOI](https://doi.org/10.1038/s41598-018-28176-5)) — Implicates CA1 in
  incremental value learning; retain only if value or reward learning becomes
  part of the Discussion.

- **Context — Evans et al. (2022)** (`evans2022`;
  [DOI](https://doi.org/10.1038/s41598-022-10947-w)) — Links increased
  neurogenesis, forgetting, and reduced CA1 retrieval activity, but it is
  peripheral to the paper’s synaptic interference mechanism.

## 6. Continual-learning and cognitive background currently in the bibliography

These papers are relevant to nearby questions but are not presently required
to tell the focused story.

- **Context — González et al. (2020)** (`gonzalez`;
  [DOI](https://doi.org/10.7554/eLife.51005)) — Models sleep replay as protection
  against catastrophic forgetting; potentially useful when distinguishing the
  project’s online interference from consolidation-based solutions.

- **Context — Hayes et al. (2020)** (`hayes2020`;
  [DOI](https://doi.org/10.1007/978-3-030-58598-3_28)) — REMIND is a machine
  continual-learning method based on compressed replay and should be cited only
  if the Discussion explicitly compares replay-based engineering solutions.

- **Context — Sikström (2006)** (`sikstrom2006`;
  [DOI](https://doi.org/10.1207/s15516709cog0000_55)) — Uses adaptive LTP/LTD
  thresholds to explain serial-position effects, a possible cognitive analogy
  for recency but not direct evidence for BTSP.

- **Context — Hasselmo (2006)** (`hasselmo2006`;
  [DOI](https://doi.org/10.1016/j.conb.2006.09.002)) — Reviews cholinergic
  modulation of learning and memory; relevant only if neuromodulatory gating is
  developed as a future mechanism.

- **Context — Miller et al. (2025)** (`miller2025`;
  [DOI](https://doi.org/10.1101/2025.02.19.638996)) — A preprint linking
  hippocampal amnesia to reduced representational distinctiveness and stability
  in a broader autobiographical-memory network; do not rely on it for a central
  claim while it remains a preprint.

## 7. Claim-to-citation map for the manuscript

| Planned claim | Primary citations | Qualification needed |
|---|---|---|
| BTSP supports rapid, one-trial CA1 field formation across seconds | Bittner 2017; Milstein 2021; Magee 2026 | The project omits measured temporal kernels, dendrites, and full bidirectionality. |
| Entorhinal activity can act as a target-like instructive input to CA1 | Grienberger & Magee 2022 | The model’s instructive signal is the encoded target `c = E(x)`, not raw EC activity. |
| Instructive signals can be neuron-specific and vectorized | Francioni et al. 2026 | This evidence is from retrosplenial cortex and a BCI task, not hippocampal memory. |
| BTSP-like events can create non-spatial sensory representations | Dorian et al. 2026 | E3/E6 are an abstract factorial task, not a quantitative fit to the odor experiment. |
| Rapid hippocampal storage creates a stability–plasticity problem | McClelland et al. 1995; Vaidya et al. 2025 | A stable pretrained decoder is a modeling assumption, not a claimed anatomical fact. |
| Information can survive while a fixed readout fails | Rule et al. 2020; Gallego et al. 2020 | These are conceptual precedents from cortical datasets, not demonstrations of the present causal mechanism. |
| BTSP-inspired rules can implement associative memory | Wu & Maass 2025; Li et al. 2024 | The paper’s novelty must be decoder alignment/rescue, not one-shot associative memory alone. |
| Spatial scaffolds and content can be computationally factored | Chandra et al. 2025 | Vector-HaSH solves a broader problem with different circuitry and learning assumptions. |
| CA1 combines spatial and task/sensory variables | Symanski et al. 2022; Dorian et al. 2026 | Do not call cue remapping “object-vector coding.” |
| The E7 model–data comparison is directional, not quantitative agreement | Symanski et al. 2022 | Report the exact contingency and uncertainty; do not generalize beyond that dataset. |

## 8. Bibliography hygiene

- [ ] Remove one of `grienberger2022` / `grienberger2022a` (same DOI).
- [ ] Remove one of `li2024` / `li2024a` (same DOI).
- [ ] Remove one of `zou2023` / `zou2023a` (same DOI).
- [ ] Add the missing year **2020** to `gonzalez`.
- [x] Add the final 2026 Nature Communications record for Dorian et al., DOI
  `10.1038/s41467-026-71503-y`; the malformed legacy `zotero-item-7051` remains
  for the later cleanup pass.
- [x] Add a complete BibTeX record for Symanski, Bladon et al. (2022), DOI
  `10.7554/eLife.79545`; it is already a prespecified empirical source in E7.
- [x] Add Marr (1971), Teyler and DiScenna (1986), McClelland et al. (1995),
  Schapiro et al. (2017), Rule et al. (2020), Gallego et al. (2020), Schoonover
  et al. (2021), Jain et al. (2024), and Pang and Recanatesi (2025).
- [x] Add Francioni et al. (2026), DOI `10.1038/s41586-026-10190-7`, after the
  novelty audit identified it as the decisive limit on vectorization claims.
- [ ] Remove absolute Zotero `file` paths before sharing the bibliography; they
  expose a local filesystem and are not useful to readers.
- [ ] Compile the paper after cleanup and verify that every citation key is
  unique and resolves exactly once.

## 9. Next literature tasks

- [x] Triage the existing bibliography by relevance to the frozen paper story.
- [x] Identify a primary citation for the E7/E9 empirical contingency.
- [x] Identify the closest BTSP associative-memory comparator.
- [x] Identify initial stable-readout and representational-drift literature.
- [ ] Read the full methods/results of the **Core** papers and record exact
  claim boundaries, sample sizes, and figure/table locations where needed.
- [x] Search explicitly for prior models in which a teaching/instructive signal
  is aligned to a fixed downstream decoder; this is the decisive novelty audit.
- [ ] Search for communication-subspace and representational-alignment work
  that is more anatomically relevant to hippocampal outputs.
- [ ] Decide whether “hippocampal index” is helpful framing or creates an
  unnecessary content-versus-index detour.
- [x] Convert this map into a sentence-level Introduction claim–citation table.
- [ ] Update `references.bib` only after final records and preferred citation
  keys have been agreed upon.

## Provisional Introduction citation spine

A concise Introduction can probably be built from this sequence:

1. Fast hippocampal learning and the stability–plasticity problem — Marr
   (1971); McClelland et al. (1995).
2. BTSP as a one-trial, seconds-scale CA1 learning mechanism — Bittner et al.
   (2017); Milstein et al. (2021).
3. Entorhinal target-like instruction and non-spatial generality — Grienberger
   and Magee (2022); Dorian et al. (2026).
4. The unresolved readout problem — Rule et al. (2020), with the distinction
   between information in a population and compatibility with a fixed decoder.
5. Closest computational alternatives — Pang and Recanatesi (2025); Wu and
   Maass (2025); Chandra et al. (2025).
6. Vectorized dendritic instruction already exists as an experimental concept
   — Francioni et al. (2026), delimiting the novelty claim.
7. Present question — whether instructive-signal coordinates causally
   determine downstream decodability, tested by permutation and matched-decoder
   rescue.

This spine should remain short: many of the supporting and context papers above
belong in the Discussion or Methods rather than in the opening narrative.
