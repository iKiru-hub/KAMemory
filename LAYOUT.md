# KAMemory repository layout

This document maps the active, paper-ready code path. The normal flow is:

```text
configuration → distribution-matched autoencoder → headless simulation
→ frozen arrays and report → paper figure → manuscript
```

New work belongs in `src/kamemory/` or `src/experiments/`. Notebooks, flat
legacy modules, and archived searches remain available for continuity but are
not authoritative for new results.

## Top-level map

```text
KAMemory/
├── src/
│   ├── kamemory/              canonical reusable backend
│   ├── experiments/           definitive E0–E9 headless scripts
│   │   ├── plots/             NPZ source arrays, JSON reports, diagnostics
│   │   └── reference_data/    prespecified empirical input
│   ├── configs/               training and experiment configurations
│   ├── data/autoencoders/     named loader-compatible checkpoints
│   └── train_autoencoder.py   autoencoder terminal entry point
├── tests/                     backend, parity, and protocol tests
├── article/                   manuscript and paper-figure assets
├── notebooks/                 exploratory/legacy notebooks
├── notes/                     ideas, meetings, and legacy drafting material
├── media/                     historical/generated visual material
├── README.md                  setup and quick commands
├── EXPERIMENTS.md             experiment-level scientific record
├── TODO.md                    repository completion tracker
└── LAYOUT.md                  this map
```

## Canonical backend: `src/kamemory/`

| File | Role |
|---|---|
| `models.py` | `Autoencoder`, `BTSPMemory`, and the storage/retrieval boundary. |
| `plasticity.py` | Target-gated, Hebbian, and delta local update rules. |
| `data.py` | Sparse patterns, sensory patterns, legacy track input, and factorial-track generation. |
| `training.py` | Small reusable autoencoder and memory training/evaluation loops. |
| `autoencoder_experiment.py` | One-call factorial AE generation, train/validation/test, best-checkpoint restore, and save. |
| `io.py` | Paths, JSON config loading, checkpoint persistence, and runtime metadata. |
| `utils.py` | Seeding, sparsity, similarity, and digest helpers. |
| `checks.py` | Fast backend smoke checks. |

Import new code from this package:

```python
from kamemory import Autoencoder, BTSPMemory, run_autoencoder_experiment
```

`src/models.py`, `src/training.py`, and `src/utils.py` are legacy
notebook-parity references. Do not extend them.

## Autoencoders and stimulus distributions

The project intentionally keeps two distribution-matched autoencoder families.

| Used by | Input distribution | Checkpoint |
|---|---|---|
| E1, E2a, E2b, E4, E5, E8; paper Figures 1–3 | Exactly K-hot 50-dimensional sparse patterns | `ae_8` |
| E3, E6, E7, E9; paper Figure 4 | 100-dimensional factorial spatial–sensory track | `ae_factorial_paper_v1` |

Do not use the spatial–sensory checkpoint for random sparse-pattern experiments
or the sparse checkpoint for spatial–sensory experiments. The autoencoder must
be trained on the distribution used by its downstream simulation.

### Train the factorial spatial–sensory autoencoder

- Trainer/API: `src/kamemory/autoencoder_experiment.py`
- Terminal wrapper: `src/train_autoencoder.py`
- Default config: `src/configs/autoencoder_factorial.json`

```bash
.venv/kamvenv/bin/python src/train_autoencoder.py --name ae_factorial_paper_v1
```

The trainer creates independent train, validation, and held-out test sessions
using the exact `generate_factorial_track` distribution. It saves:

```text
src/data/autoencoders/<checkpoint-name>/
├── autoencoder.pt   # PyTorch state dictionary
└── info.json        # settings, data hashes, history, metrics, runtime metadata
```

Load a checkpoint with:

```python
from kamemory import load_autoencoder_session
info, autoencoder = load_autoencoder_session("ae_factorial_paper_v1")
```

Named checkpoints are never overwritten. Pick a fresh name for a new run and
pass it explicitly into E3, E6, and E9.

## Configurations: `src/configs/`

| File | Purpose |
|---|---|
| `autoencoder_factorial.json` | Factorial AE architecture, data distribution, optimizer, validation, and save settings. |
| `optimized_memory.json` | Frozen E4 robustness reference. |
| `rule_baselines.json` | E8 rule-comparison protocol. |
| `is_heterogeneity.json` | E9 plateau-rate sweep; defaults to `ae_factorial_paper_v1`. |
| `base_configs.json`, `lap_configs.json` | Older schemas retained for legacy continuity. |

## Definitive simulations: `src/experiments/`

Each script runs from the repository root and writes its `.npz`, `.json`,
`.png`, and `.pdf` outputs under `src/experiments/plots/`.

| ID | Entry point | Purpose | Dependency |
|---|---|---|---|
| E0 | `00_model_validation.py` | Update equation, determinism, and mutation-boundary validation. | Backend only. |
| E1 | `01_alignment.py` | Alignment/permutation/matched-decoder causal test. | `ae_8`; sparse patterns. |
| E2a | `02_retention.py` | Sequential retention and capacity. | `ae_8`; sparse patterns. |
| E2b | `03_degraded_cues.py` | Masked and bit-flipped retrieval. | `ae_8`; sparse patterns. |
| E3 | `04_ca1_track.py` | Factorial cue×position CA1 task and held-out laps. | `ae_factorial_paper_v1`. |
| E3 legacy | `04b_legacy_remapping.py` | Notebook-compatible same-cue field shifts. | Pinned historical protocol. |
| E4 | `05_sensitivity.py` | Robustness and partial-misalignment sweep. | `ae_8`; optimized config. |
| E5 | `06_interference_analysis.py` | Trace survival, crosstalk, and forgetting. | E2 arrays and `ae_8`. |
| E6 | `07_ca1_mixed_selectivity.py` | Training-only classification and held-out remapping. | Replays E3; `ae_factorial_paper_v1`. |
| E7 | `08_ca1_data_comparison.py` | Prespecified model-versus-published CA1 comparison. | E6 arrays plus reference data. |
| E8 | `09_rule_baselines.py` | Rule-specific controls. | `ae_8`; sparse patterns. |
| E9 | `10_is_heterogeneity.py` | Background plateau-rate limitation sweep. | `ae_factorial_paper_v1`; fresh track seeds. |

### Spatial–sensory dependency order

```text
train autoencoder → E3 → E6 → E7 → E9 → compose Figure 4
```

From a clean checkout:

```bash
.venv/kamvenv/bin/python src/train_autoencoder.py --name ae_factorial_paper_v1
.venv/kamvenv/bin/python src/experiments/04_ca1_track.py --deterministic
.venv/kamvenv/bin/python src/experiments/07_ca1_mixed_selectivity.py --deterministic
.venv/kamvenv/bin/python src/experiments/08_ca1_data_comparison.py
.venv/kamvenv/bin/python src/experiments/10_is_heterogeneity.py --deterministic
```

Do not mix a new E3 checkpoint with old E6/E7/E9 arrays; rerun the dependency
chain before rebuilding Figure 4.

## Outputs: `src/experiments/plots/`

For every `eN_*` prefix:

- `.npz` — raw numerical source arrays for downstream analysis and paper panels.
- `.json` — configuration, checkpoint metadata, statistics, protocol checks, and provenance.
- `.png` / `.pdf` — experiment-level diagnostics, not automatically final paper panels.

E7's empirical anchor is
`src/experiments/reference_data/symanski_2022_ca1_profile.json`.

## Manuscript and paper figures: `article/`

| Path | Contents |
|---|---|
| `article/maintex/main.tex` | LaTeX manuscript source. |
| `article/maintex/references.bib` | Bibliography database. |
| `article/MANUSCRIPT.md` | Claims, scope, result register, figure plan, and paper checklist. |
| `article/LITERATURE.md` | Annotated literature list. |
| `article/NOVELTY_AUDIT.md` | Closest-model and novelty audit. |
| `article/figures/make_main_figures.py` | Composes the four paper figures from frozen arrays. |
| `article/figures/figure_*.{png,pdf,svg}` | Main-figure deliverables. |
| `article/figures/figure_*_source_data.csv` | Tidy plotted source data. |
| `article/figures/main_figure_manifest.json` | Input/output and checkpoint hashes. |
| `article/figures/README.md` | Figure captions, provenance, and full regeneration order. |

Rebuild all main figures:

```bash
.venv/kamvenv/bin/python article/figures/make_main_figures.py
```

Figure mapping:

1. Model and rule — E1.
2. Alignment causality — E1 and E4.
3. Memory/interference — E2a, E2b, and E5.
4. Spatial–sensory CA1 result and biological boundary — E6, E7, and E9.

## Tests

The 44 tests under `tests/` cover backend mechanics, legacy parity, the new
factorial-autoencoder save/reload path, factorial-task protocol, and
experiment-specific statistics.

```bash
.venv/kamvenv/bin/python -m unittest discover -s tests -q
```

## Historical and exploratory material

- `notebooks/` — original exploration and figure-development notebooks.
- `notes/` — ideas, meetings, diagrams, and legacy draft material.
- `media/` — earlier generated figures, animations, and visual assets.
- `src/other/`, `src/optim_wb/`, and `src/.archive/` — exploratory searches
  and archived code/artifacts.

These directories are useful context, but paper Methods, Results, and
reproducibility should be based on the canonical backend, headless
experiments, frozen reports, and article assets above.
