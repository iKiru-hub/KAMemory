# Main paper figures

This directory contains the four figures that carry the proposed paper's main
argument. They are composed directly from the frozen experiment arrays rather
than copied from notebook panels or existing raster figures.

## Rebuild

From the repository root, run:

```bash
.venv/kamvenv/bin/python article/figures/make_main_figures.py
```

To regenerate the spatial–sensory branch from scratch in a clean checkout,
the dependency order is deliberately explicit:

```bash
.venv/kamvenv/bin/python src/train_autoencoder.py --name ae_factorial_paper_v1
.venv/kamvenv/bin/python src/experiments/04_ca1_track.py --deterministic
.venv/kamvenv/bin/python src/experiments/07_ca1_mixed_selectivity.py --deterministic
.venv/kamvenv/bin/python src/experiments/08_ca1_data_comparison.py
.venv/kamvenv/bin/python src/experiments/10_is_heterogeneity.py --deterministic
.venv/kamvenv/bin/python article/figures/make_main_figures.py
```

The training command intentionally refuses to overwrite an existing named
checkpoint. In a non-clean workspace, choose a new name and pass that same
checkpoint explicitly to E3, E6, and E9.

The command writes PNG previews at 400 dpi, vector PDF/SVG files, a tidy CSV
source-data table for each figure, and `main_figure_manifest.json`. The
manifest records the exact input files, SHA-256 hashes, outputs, and the rule
used to select representative examples. It also records hashes for both
distribution-matched autoencoder checkpoints.

All intervals shown in quantitative panels are mean ± 1.96 SEM across
independent network/data seeds. Lines connecting points indicate paired seeds.
Cells, memories, trials, and the two layouts within a seed are not treated as
independent inferential samples. Condition names and colors are fixed across
figures.

Figures 1–3 use the distribution-matched sparse-pattern checkpoint `ae_8`.
Before regenerating the spatial–sensory branch, the 100→1000→100 autoencoder
was retrained on independent samples from the exact factorial-track generator;
Figure 4 uses the validation-selected checkpoint `ae_factorial_paper_v1`.

## Figure 1 — Model and representational problem

**Claim:** rapid plasticity is useful only when its learned CA1 coordinates are
compatible with the stable decoder.

- **A:** implemented EC→CA3→CA1→EC architecture, distinguishing frozen and
  plastic paths.
- **B:** the exact target-gated update and the scope of the "BTSP-inspired"
  label; no eligibility or instructive timing traces are implemented.
- **C:** aligned, permuted, and matched-decoder coordinate relationships. The
  fixed-permutation and rescue conditions have identical learned weights.
- **D:** one representative E1 memory, selected deterministically using the
  seed nearest the median aligned score and then the memory nearest that
  seed's median score. The rule and selected indices are recorded in the
  manifest.

Quantitative input: `src/experiments/plots/e1_alignment.npz`. The conceptual
panels follow the implemented architecture and update in
`src/kamemory/models.py` and `src/kamemory/plasticity.py`.

**Caption draft.** **Decoder-aligned instructive signals define readable CA1
coordinates.** (A) The implemented model uses a frozen EC encoder and decoder,
a frozen EC→CA3 projection, and plastic CA3→CA1 synapses. (B) For CA3 activity
`h_t` and encoded target `c_t`, the target-gated update overwrites each CA1 row
in proportion to its instructive activity. It is BTSP-inspired but contains no
seconds-long timing traces. (C) Permuting `c_t` changes the coordinate system
relative to the fixed decoder; applying the matched inverse permutation at
readout restores compatibility without changing the plastic weights. (D)
Representative target and reconstructions from E1 (seed 507, memory 7), chosen
by the prespecified median-example rule rather than visual appearance.

## Figure 2 — Alignment causally controls decodability

**Claim:** decoder compatibility, rather than signal magnitude or information
loss, causally determines whether stored content can be decoded.

- **A:** five paired controls: aligned, fixed permutation, matched decoder,
  random matched, and no plasticity.
- **B:** primary chance-corrected cosine after 28 stores.
- **C:** top-K F1 as an independent sparse-content endpoint.
- **D:** the paired fixed-permutation→matched-decoder rescue, with identical
  learned weights.
- **E:** the E4 partial-permutation dose response and full matched rescue.

Inputs: `src/experiments/plots/e1_alignment.npz` and
`src/experiments/plots/e4_sensitivity.npz`.

**Caption draft.** **Instructive-signal–decoder alignment causally determines
content decodability.** (A) Paired controls alter the relationship between the
instructive code and stable decoder while holding the relevant simulated
memories and seeds fixed. (B,C) Seed-level chance-corrected cosine and top-K
recovery after 28 stores. Points are independent network/data seeds (`n=20`);
black symbols show means and 95% normal-approximation confidence intervals.
(D) Applying the matched decoder rescues every fixed-permutation network even
though its learned weights are unchanged. (E) Decodability decreases as a
larger fraction of instructive coordinates is permuted and is restored by the
matched decoder at full permutation (`n=20` paired seeds; lines show seeds,
symbols mean ± 1.96 SEM).

## Figure 3 — Memory function and interference

**Claim:** aligned storage supports sequential retention and degraded-cue
retrieval, while trace survival and crosstalk explain its finite memory span.

- **A:** mean aligned sequential-recall matrix across loads up to 60 memories.
- **B:** memory-age curves for the four core conditions.
- **C:** seed-level contiguous capacity using the frozen chance-corrected
  recall threshold of `0.7778`.
- **D:** identity recovery from masking and bit flips, with nearest-neighbor
  references.
- **E:** within-seed associations of recall with exact trace survival and
  crosstalk.
- **F:** learning rate versus recall half-life.

Inputs: `src/experiments/plots/e2a_retention.npz`,
`src/experiments/plots/e2b_degraded_cues.npz`, and
`src/experiments/plots/e5_interference_analysis.npz`.

**Caption draft.** **Aligned storage supports associative retrieval but has a
finite interference-limited lifetime.** (A) Mean chance-corrected recall after
each sequential storage event in the aligned condition. (B) Recall as a
function of memory age; the matched decoder overlaps the aligned condition.
(C) Contiguous load capacity at the prespecified chance-corrected threshold of
`0.7778`. Points are independent network/data seeds. (D) Correct memory
identity from masked or bit-flipped cues; the dotted line is the
nearest-neighbor reference and chance is `1/8`. (E) Within each seed, trace
survival correlates positively and crosstalk negatively with recall. (F)
Increasing the learning rate shortens recall half-life. For B–F, `n=20`
independent seeds; lines or points show seeds and intervals are mean ± 1.96
SEM. Cells, memories, and corrupted-cue draws are nested within seeds.

## Figure 4 — CA1-like responses and a biological boundary

**Claim:** the model produces stable spatial–sensory response structure but
overcouples cue and spatial selectivity relative to published CA1 data, and
unmatched background plateaus do not repair the discrepancy.

- **A:** fully crossed cue×position design, cue-free probes, and held-out laps.
- **B:** held-out response profiles from cells classified using training laps
  only.
- **C:** tuned fraction among all units and class composition among tuned
  cells, with percentages rendered directly.
- **D:** cue-tuning strength versus held-out cue-evoked remapping.
- **E:** published and model conditional spatial prevalence.
- **F:** E9 background-event rate against biological log-odds-ratio error and
  held-out output quality.

Inputs: `src/experiments/plots/e6_ca1_mixed_selectivity.npz`,
`src/experiments/plots/e7_ca1_data_comparison.npz`, and
`src/experiments/plots/e9_is_heterogeneity.npz`.

**Caption draft.** **Aligned plasticity produces CA1-like spatial–sensory
structure while revealing a quantitative biological mismatch.** (A) The
factorial task crosses two cues with four positions and reserves independent
laps for evaluation; cue-free laps probe spatial responses. (B) Example
held-out responses from position-only, cue-only, and conjunctive cells,
selected and ranked using training classifications only (representative seed
1100). (C) Across all model units, 8.2% are tuned; among tuned cells, 7.0% are
position-only, 3.1% cue-only, 0% additive mixed, and 89.8% conjunctive. Values
pool the two paired layouts within each seed (`n=12` independent seeds). (D)
Cue-tuning strength predicts cue-evoked remapping on held-out laps (mean
seed-level Spearman `rho=0.996`; `n=12`). (E) The model matches the direction
of the published cue–space association but strongly exaggerates it: spatial
effects occur in 57.2% versus 65.7% of published cue-inactive and
cue-responsive cells, compared with 0.6% versus 96.6% in the model. Published
bars are descriptive counts (273/477 and 71/108); model points are independent
seeds (`n=12`). (F) Low rates produce only a small reduction in the
prespecified absolute published log-odds-ratio error, no rate meets the frozen
quantitative-repair rule, and high rates reduce held-out EC-output cosine
(`n=20` fresh seeds, two paired layouts per seed; intervals mean ± 1.96 SEM).

## Main versus supplementary decision

The E9 negative result is retained as Figure 4F because it closes the main
biological-boundary claim. The full E8 rule comparison is assigned to
Supplementary Figure S4: it establishes rule generality and the
stability–plasticity trade-off, but adding it to the already six-panel Figure
3 would obscure the central memory/interference mechanism. E0 backend
validation and E3 legacy field-shift continuity likewise remain supplementary.

## Output inventory

For each figure, prefer PDF or SVG for manuscript assembly and PNG for quick
review. Each `figure_N_source_data.csv` uses the columns `figure`, `panel`,
`series`, `seed`, `x`, `value`, and `detail`. Figure 1's conceptual panels do
not create artificial quantitative rows; its source table records only the
displayed representative data. See `main_figure_manifest.json` for complete
file provenance and checksums.
