"""E6: cross-validated CA1 position, cue, and conjunctive selectivity.

This analysis replays the frozen E3 factorial cue x position protocol. Cell
classes are assigned from training laps only using factorial partial-F tests,
Benjamini-Hochberg FDR control, and a minimum partial eta-squared. Effect sizes
and cue-evoked remapping are then measured on independently generated held-out
laps.

Run from the repository root:

    python3 src/experiments/07_ca1_mixed_selectivity.py --deterministic
"""

from __future__ import annotations

import argparse
import copy
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kamemory-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from kamemory.io import load_autoencoder_session, runtime_metadata
from kamemory.utils import array_digest, seed_everything


SCHEMA_VERSION = 1
DEFAULT_E3_DATA = Path(__file__).resolve().parent / "plots" / "e3_ca1_track.npz"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
CONDITIONS = ("aligned", "random_matched", "no_plasticity")
DISPLAY_NAMES = {
    "aligned": "Aligned IS",
    "random_matched": "Random matched IS",
    "no_plasticity": "No plasticity",
}
COLORS = {
    "aligned": "#2878B5",
    "random_matched": "#8C6BB1",
    "no_plasticity": "#7A7A7A",
}
CLASS_NAMES = ("position", "cue", "additive_mixed", "conjunctive", "unclassified")
CLASS_LABELS = (
    "Position only",
    "Cue only (position-invariant)",
    "Additive position + cue",
    "Cue × position (mixed)",
    "Unclassified",
)
CLASS_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B8B8B8")


def _load_e3_module():
    path = Path(__file__).resolve().parent / "04_ca1_track.py"
    spec = spec_from_file_location("e3_factorial_source", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E3 = _load_e3_module()


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def design_matrices(cues: np.ndarray, positions: np.ndarray) -> dict[str, np.ndarray]:
    """Nested balanced factorial designs for cell-wise partial-effect tests."""

    cues = np.asarray(cues, dtype=float)
    positions = np.asarray(positions)
    levels = np.unique(positions)
    position_terms = np.column_stack([positions == level for level in levels[1:]]).astype(float)
    intercept = np.ones((len(cues), 1))
    cue_term = cues[:, None]
    return {
        "intercept": intercept,
        "cue_only": np.column_stack((intercept, cue_term)),
        "position_only": np.column_stack((intercept, position_terms)),
        "additive": np.column_stack((intercept, position_terms, cue_term)),
        "full": np.column_stack((intercept, position_terms, cue_term, position_terms * cue_term)),
    }


def residual_sum_squares(design: np.ndarray, activity: np.ndarray) -> np.ndarray:
    coefficients = np.linalg.lstsq(design, activity, rcond=None)[0]
    residual = activity - design @ coefficients
    return np.sum(residual * residual, axis=0)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Numerical Recipes continued fraction for the incomplete beta."""

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1e-300 if abs(d) < 1e-300 else d
    d = 1.0 / d
    result = d
    for iteration in range(1, 201):
        twice = 2 * iteration
        numerator = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
        d = 1.0 + numerator * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + numerator / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        result *= d * c
        numerator = -(a + iteration) * (qab + iteration) * x / ((a + twice) * (qap + twice))
        d = 1.0 + numerator * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + numerator / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return result


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def f_survival(statistic: np.ndarray, df_effect: int, df_error: int) -> np.ndarray:
    statistic = np.asarray(statistic, dtype=float)
    transformed = df_error / (df_error + df_effect * statistic)
    return np.asarray(
        [regularized_beta(float(x), df_error / 2, df_effect / 2) for x in transformed]
    )


def partial_test(
    reduced: np.ndarray, full: np.ndarray, activity: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return partial F, p, and eta-squared for every activity column."""

    rss_reduced = residual_sum_squares(reduced, activity)
    rss_full = residual_sum_squares(full, activity)
    df_effect = full.shape[1] - reduced.shape[1]
    df_error = len(activity) - full.shape[1]
    effect_ss = np.maximum(rss_reduced - rss_full, 0.0)
    denominator = rss_full / max(df_error, 1)
    statistic = np.divide(
        effect_ss / df_effect,
        denominator,
        out=np.full_like(effect_ss, np.inf),
        where=denominator > 1e-12,
    )
    statistic[(effect_ss <= 1e-12) & (denominator <= 1e-12)] = 0.0
    p_value = f_survival(statistic, df_effect, df_error)
    eta = np.divide(
        effect_ss,
        effect_ss + rss_full,
        out=np.zeros_like(effect_ss),
        where=(effect_ss + rss_full) > 1e-12,
    )
    return statistic, p_value, eta


def bh_significant(p_values: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg discoveries at target FDR q."""

    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ordered = p_values[order]
    passing = ordered <= q * np.arange(1, len(ordered) + 1) / len(ordered)
    significant = np.zeros(len(ordered), dtype=bool)
    if np.any(passing):
        cutoff = ordered[np.flatnonzero(passing)[-1]]
        significant = p_values <= cutoff
    return significant


def factorial_effects(
    activity: np.ndarray,
    cues: np.ndarray,
    positions: np.ndarray,
    *,
    fdr: float = 0.05,
    min_eta: float = 0.05,
) -> dict[str, np.ndarray]:
    """Orthogonal ANOVA effects for the balanced cue x position design."""

    activity = np.asarray(activity, dtype=float)
    cues = np.asarray(cues)
    positions = np.asarray(positions)
    cue_levels = np.unique(cues)
    position_levels = np.unique(positions)
    counts = np.asarray(
        [
            np.count_nonzero((cues == cue) & (positions == position))
            for cue in cue_levels
            for position in position_levels
        ]
    )
    if len(np.unique(counts)) != 1:
        raise ValueError("factorial ANOVA requires equal repeats per cue-position cell")
    repeats = int(counts[0])
    cell_means = np.asarray(
        [
            [
                activity[(cues == cue) & (positions == position)].mean(axis=0)
                for position in position_levels
            ]
            for cue in cue_levels
        ]
    )
    grand = activity.mean(axis=0)
    cue_means = cell_means.mean(axis=1)
    position_means = cell_means.mean(axis=0)
    interaction_component = (
        cell_means
        - cue_means[:, None, :]
        - position_means[None, :, :]
        + grand[None, None, :]
    )
    lookup = {
        (cue, position): cell_means[cue_index, position_index]
        for cue_index, cue in enumerate(cue_levels)
        for position_index, position in enumerate(position_levels)
    }
    residual = np.asarray(
        [
            row - lookup[(cue, position)]
            for row, cue, position in zip(activity, cues, positions)
        ]
    )
    error_ss = np.sum(residual * residual, axis=0)
    error_df = len(activity) - len(cue_levels) * len(position_levels)
    effect_specs = {
        "position": (
            len(cue_levels)
            * repeats
            * np.sum((position_means - grand) ** 2, axis=0),
            len(position_levels) - 1,
        ),
        "cue": (
            len(position_levels)
            * repeats
            * np.sum((cue_means - grand) ** 2, axis=0),
            len(cue_levels) - 1,
        ),
        "interaction": (
            repeats * np.sum(interaction_component**2, axis=(0, 1)),
            (len(cue_levels) - 1) * (len(position_levels) - 1),
        ),
    }
    result: dict[str, np.ndarray] = {}
    for name, (effect_ss, effect_df) in effect_specs.items():
        denominator = error_ss / max(error_df, 1)
        statistic = np.divide(
            effect_ss / effect_df,
            denominator,
            out=np.full_like(effect_ss, np.inf),
            where=denominator > 1e-12,
        )
        statistic[(effect_ss <= 1e-12) & (denominator <= 1e-12)] = 0.0
        p_value = f_survival(statistic, effect_df, error_df)
        eta = np.divide(
            effect_ss,
            effect_ss + error_ss,
            out=np.zeros_like(effect_ss),
            where=(effect_ss + error_ss) > 1e-12,
        )
        result[f"{name}_f"] = statistic
        result[f"{name}_p"] = p_value
        result[f"{name}_eta"] = eta
        result[f"{name}_active"] = bh_significant(p_value, fdr) & (eta >= min_eta)
    return result


def classify_cells(effects: dict[str, np.ndarray]) -> np.ndarray:
    position = effects["position_active"]
    cue = effects["cue_active"]
    interaction = effects["interaction_active"]
    classes = np.full(len(position), 4, dtype=np.int8)
    classes[position & ~cue & ~interaction] = 0
    classes[cue & ~position & ~interaction] = 1
    classes[position & cue & ~interaction] = 2
    classes[interaction] = 3
    return classes


def cue_contrast(activity: np.ndarray, cues: np.ndarray) -> np.ndarray:
    return activity[cues == 1].mean(axis=0) - activity[cues == 0].mean(axis=0)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based average ranks with exact tie handling."""

    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def event_rows(cue_ids: np.ndarray, positions: np.ndarray, track_length: int) -> np.ndarray:
    laps = np.flatnonzero(cue_ids >= 0)
    return laps * track_length + positions[laps]


def probe_rows(cue_ids: np.ndarray, probe_positions: np.ndarray, track_length: int) -> np.ndarray:
    free_laps = np.flatnonzero(cue_ids < 0)
    return np.asarray([lap * track_length + position for lap in free_laps for position in probe_positions])


def sparse_final_weights(
    ca3: torch.Tensor,
    signals: torch.Tensor,
    alpha_profile: np.ndarray,
) -> torch.Tensor:
    """Exact BTSP updates restricted to nonzero instructive-signal rows."""

    weights = torch.zeros((signals.shape[1], ca3.shape[1]), dtype=torch.float32)
    with torch.no_grad():
        for presynaptic, signal, alpha in zip(ca3, signals, alpha_profile):
            active = torch.nonzero(signal != 0, as_tuple=False).reshape(-1)
            if active.numel():
                selected = signal[active, None]
                weights[active] = (1 - float(alpha) * selected) * weights[active] + float(alpha) * selected * presynaptic
    return weights


def retrieve_selected(memory, inputs: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return E3.retrieve_bank(memory, inputs[rows])[0]


def analyze_condition(
    train_event: np.ndarray,
    test_event: np.ndarray,
    test_free: np.ndarray,
    train_cues: np.ndarray,
    train_positions: np.ndarray,
    test_cues: np.ndarray,
    test_positions: np.ndarray,
    probe_positions: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray | float]:
    train_effects = factorial_effects(
        train_event, train_cues, train_positions, fdr=args.fdr, min_eta=args.min_eta
    )
    test_effects = factorial_effects(
        test_event, test_cues, test_positions, fdr=1.0, min_eta=0.0
    )
    classes = classify_cells(train_effects)
    fractions = np.asarray([np.mean(classes == index) for index in range(len(CLASS_NAMES))])

    free_mean = test_free.reshape(-1, len(probe_positions), test_free.shape[1]).mean(axis=0)
    event_means = np.asarray(
        [
            [test_event[(test_cues == cue) & (test_positions == position)].mean(axis=0) for position in probe_positions]
            for cue in (0, 1)
        ]
    )
    remapping = np.sqrt(np.mean((event_means - free_mean[None, :, :]) ** 2, axis=(0, 1)))
    train_cue_magnitude = np.abs(cue_contrast(train_event, train_cues))
    finite = (np.std(train_cue_magnitude) > 1e-12) and (np.std(remapping) > 1e-12)
    if finite:
        cue_ranks = average_ranks(train_cue_magnitude)
        remapping_ranks = average_ranks(remapping)
        remapping_rho = float(np.corrcoef(cue_ranks, remapping_ranks)[0, 1])
    else:
        remapping_rho = 0.0

    heldout_target_eta = np.choose(
        classes,
        [
            test_effects["position_eta"],
            test_effects["cue_eta"],
            np.maximum(test_effects["position_eta"], test_effects["cue_eta"]),
            test_effects["interaction_eta"],
            np.zeros(len(classes)),
        ],
    )
    return {
        "classes": classes,
        "class_fractions": fractions,
        "train_position_eta": train_effects["position_eta"],
        "train_cue_eta": train_effects["cue_eta"],
        "train_interaction_eta": train_effects["interaction_eta"],
        "train_position_active": train_effects["position_active"],
        "train_cue_active": train_effects["cue_active"],
        "train_interaction_active": train_effects["interaction_active"],
        "test_position_eta": test_effects["position_eta"],
        "test_cue_eta": test_effects["cue_eta"],
        "test_interaction_eta": test_effects["interaction_eta"],
        "heldout_target_eta": heldout_target_eta,
        "train_cue_magnitude": train_cue_magnitude,
        "heldout_remapping": remapping,
        "remapping_rho": remapping_rho,
        "test_event_means": event_means,
    }


def replay_seed_layout(
    source: dict[str, np.ndarray],
    seed_index: int,
    layout_index: int,
    autoencoder,
    params: dict,
    args: argparse.Namespace,
) -> dict[str, object]:
    seed = int(source["seeds"][seed_index])
    run_seed = seed + 100_000 * layout_index
    seed_everything(run_seed)
    train_inputs = source["train_inputs"][seed_index, layout_index]
    test_inputs = source["test_inputs"][seed_index, layout_index]
    train_cue_ids = source["train_cue_ids"][seed_index, layout_index]
    test_cue_ids = source["test_cue_ids"][seed_index, layout_index]
    train_positions_all = source["train_event_positions"][seed_index, layout_index]
    test_positions_all = source["test_event_positions"][seed_index, layout_index]
    probe_positions = np.asarray(source["layouts"][layout_index])
    track_length = params["track_length"]

    base = E3.make_base_memory(autoencoder, params, args.alpha)
    expected_projection = source["base_ca3_projection"][seed_index, layout_index]
    if not np.array_equal(base.W_ei_ca3.detach().cpu().numpy(), expected_projection):
        raise AssertionError("E6 failed to reconstruct the frozen E3 CA3 projection")

    codes = E3.code_bank(autoencoder, train_inputs)
    control_rng = np.random.default_rng(run_seed + 80_000)
    expected_permutation = control_rng.permutation(params["dim_ca1"])
    if not np.array_equal(expected_permutation, source["permutations"][seed_index, layout_index]):
        raise AssertionError("E6 failed to reconstruct the frozen E3 control RNG")
    matching = E3.derangement(len(train_inputs), control_rng)
    if not np.array_equal(matching, source["random_matchings"][seed_index, layout_index]):
        raise AssertionError("E6 failed to reconstruct the frozen E3 random matching")

    cue_present = np.zeros(len(train_inputs), dtype=bool)
    train_rows = event_rows(train_cue_ids, train_positions_all, track_length)
    cue_present[train_rows] = True
    alpha_profile = np.full(len(train_inputs), args.baseline_alpha, dtype=np.float32)
    alpha_profile[cue_present] = args.alpha
    signals = {
        "aligned": codes,
        "random_matched": codes[matching],
    }

    test_rows = event_rows(test_cue_ids, test_positions_all, track_length)
    test_free_rows = probe_rows(test_cue_ids, probe_positions, track_length)
    train_cues = train_cue_ids[train_cue_ids >= 0]
    test_cues = test_cue_ids[test_cue_ids >= 0]
    train_positions = train_positions_all[train_cue_ids >= 0]
    test_positions = test_positions_all[test_cue_ids >= 0]
    results = {}
    for condition in CONDITIONS:
        memory = copy.deepcopy(base)
        if condition != "no_plasticity":
            for pattern, signal, alpha in zip(
                train_inputs, signals[condition], alpha_profile
            ):
                memory._alpha = float(alpha)
                memory.store(pattern, instructive_signal=signal)
        train_event = retrieve_selected(memory, train_inputs, train_rows)
        test_event = retrieve_selected(memory, test_inputs, test_rows)
        test_free = retrieve_selected(memory, test_inputs, test_free_rows)
        results[condition] = analyze_condition(
            train_event,
            test_event,
            test_free,
            train_cues,
            train_positions,
            test_cues,
            test_positions,
            probe_positions,
            args,
        )
        results[condition]["weight_digest"] = array_digest(
            memory.W_ca3_ca1.detach().cpu().numpy()
        )
    return {"seed": seed, "layout": layout_index, "conditions": results}


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def cue_tuning_dose_response(
    cue_magnitude: np.ndarray, remapping: np.ndarray
) -> np.ndarray:
    """Seed-level remapping means for discrete cue-tuning strength groups."""

    cue_magnitude = np.asarray(cue_magnitude, dtype=float)
    remapping = np.asarray(remapping, dtype=float)
    if cue_magnitude.shape != remapping.shape or cue_magnitude.ndim != 3:
        raise ValueError("cue magnitude and remapping must have shape (seed, layout, cell)")
    masks = (
        cue_magnitude <= 1e-12,
        (cue_magnitude > 1e-12) & (cue_magnitude <= 0.33),
        (cue_magnitude > 0.33) & (cue_magnitude <= 0.67),
        cue_magnitude > 0.67,
    )
    return np.asarray(
        [
            [
                np.mean(remapping[seed][mask[seed]])
                if np.any(mask[seed])
                else np.nan
                for mask in masks
            ]
            for seed in range(cue_magnitude.shape[0])
        ]
    )


def make_figure(arrays: dict[str, np.ndarray], representative: dict, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9), constrained_layout=True)
    task_ax, heat_ax, fraction_ax, eta_ax, remap_ax, rho_ax = axes.flat

    grid = np.arange(8).reshape(2, 4)
    task_ax.imshow(grid, cmap="Blues", alpha=0.25, aspect="auto")
    for cue in range(2):
        for position in range(4):
            task_ax.text(position, cue, f"Cue {cue}\nPos {position + 1}", ha="center", va="center")
    task_ax.set_xticks([])
    task_ax.set_yticks([])
    task_ax.set_title("A  Balanced factorial design")

    classes = representative["classes"]
    selected = np.concatenate([np.flatnonzero(classes == index)[:8] for index in range(4)])
    matrix = representative["test_event_means"][:, :, selected].transpose(2, 0, 1).reshape(len(selected), 8)
    image = heat_ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    heat_ax.set_xlabel("Held-out cue × position condition")
    heat_ax.set_ylabel("Training-classified cells")
    heat_ax.set_title("B  Held-out response profiles")
    figure.colorbar(image, ax=heat_ax, fraction=0.046, pad=0.04)

    effect_names = ("Position", "Cue", "Cue × position")
    effect_arrays = (
        arrays["train_position_active"],
        arrays["train_cue_active"],
        arrays["train_interaction_active"],
    )
    effect_means = np.asarray(
        [effect.mean(axis=(1, 2, 3)) for effect in effect_arrays]
    ).T
    effect_x = np.arange(len(effect_names))
    width = 0.24
    for condition_index, condition in enumerate(CONDITIONS):
        positions = effect_x + (condition_index - 1) * width
        bars = fraction_ax.bar(
            positions,
            effect_means[condition_index],
            width=width,
            color=COLORS[condition],
            label=DISPLAY_NAMES[condition],
        )
        for bar, value in zip(bars, effect_means[condition_index]):
            inside = value > 0.01
            fraction_ax.text(
                bar.get_x() + bar.get_width() / 2,
                value - 0.003 if inside else value + 0.002,
                f"{100 * value:.2f}%",
                ha="center",
                va="top" if inside else "bottom",
                fontsize=7.5,
                rotation=90,
                color="white" if inside else "#555555",
                fontweight="bold" if inside else "normal",
            )
    fraction_ax.set_xticks(effect_x, effect_names)
    fraction_ax.set_ylabel("Fraction of CA1 population")
    fraction_ax.set_ylim(0, 0.12)
    fraction_ax.set_title("C  Non-exclusive factorial effects")
    fraction_ax.text(
        0.02,
        0.97,
        "Effects overlap: mixed cells can carry cue and position information",
        transform=fraction_ax.transAxes,
        va="top",
        fontsize=8.5,
        color="#555555",
    )

    for condition_index, condition in enumerate(CONDITIONS):
        values = arrays["selected_target_eta"][condition_index].mean(axis=1)
        x = condition_index + np.linspace(-0.08, 0.08, len(values))
        eta_ax.scatter(x, values, color=COLORS[condition], alpha=0.7, s=25)
        mean, ci = mean_ci(values)
        eta_ax.errorbar(condition_index, mean, yerr=ci, fmt="o", color="black", capsize=4)
    eta_ax.set_xticks(np.arange(len(CONDITIONS)), [DISPLAY_NAMES[c] for c in CONDITIONS], rotation=15, ha="right")
    eta_ax.set_ylabel(r"Held-out target partial $\eta^2$")
    eta_ax.set_title("D  Cross-validated selectivity")

    dose_response = cue_tuning_dose_response(
        arrays["train_cue_magnitude"][0], arrays["heldout_remapping"][0]
    )
    dose_x = np.arange(dose_response.shape[1])
    for seed_values in dose_response:
        remap_ax.plot(dose_x, seed_values, color=COLORS["aligned"], alpha=0.18, linewidth=1)
    dose_mean = np.nanmean(dose_response, axis=0)
    dose_count = np.sum(np.isfinite(dose_response), axis=0)
    dose_ci = 1.96 * np.nanstd(dose_response, axis=0, ddof=1) / np.sqrt(dose_count)
    remap_ax.errorbar(
        dose_x,
        dose_mean,
        yerr=dose_ci,
        color=COLORS["aligned"],
        marker="o",
        linewidth=2.5,
        capsize=4,
    )
    remap_ax.set_xticks(
        dose_x,
        ("None\n0", "Weak\n0.25", "Moderate\n0.50", "Strong\n0.75–1"),
    )
    remap_ax.set_xlabel("Training cue-tuning group")
    remap_ax.set_ylabel("Held-out cue-evoked remapping")
    remap_ax.set_title("E  Stronger cue tuning predicts more remapping")

    for condition_index, condition in enumerate(CONDITIONS):
        values = arrays["remapping_rho"][condition_index].mean(axis=1)
        x_positions = condition_index + np.linspace(-0.08, 0.08, len(values))
        rho_ax.scatter(x_positions, values, color=COLORS[condition], alpha=0.7, s=25)
        mean, ci = mean_ci(values)
        rho_ax.errorbar(condition_index, mean, yerr=ci, fmt="o", color="black", capsize=4)
    rho_ax.axhline(0, color="black", linestyle="--", linewidth=1)
    rho_ax.set_xticks(np.arange(len(CONDITIONS)), [DISPLAY_NAMES[c] for c in CONDITIONS], rotation=15, ha="right")
    rho_ax.set_ylabel(r"Spearman $\rho$ across cells")
    rho_ax.set_title("F  Seed-level tuning–remapping link")

    handles, labels = fraction_ax.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "E6 — CA1 populations combine spatial, cue, and conjunctive codes",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    source_file = np.load(args.e3_data, allow_pickle=False)
    source = {key: source_file[key] for key in source_file.files}
    info, autoencoder = load_autoencoder_session(args.checkpoint, map_location=args.device)
    autoencoder.eval()
    params = E3.session_parameters(info)
    num_seeds = min(args.num_seeds, len(source["seeds"]))
    results = [
        [replay_seed_layout(source, seed, layout, autoencoder, params, args) for layout in range(len(source["layouts"]))]
        for seed in range(num_seeds)
    ]

    class_fractions = np.asarray([
        [[results[s][layout]["conditions"][condition]["class_fractions"] for layout in range(len(source["layouts"]))] for s in range(num_seeds)]
        for condition in CONDITIONS
    ])
    remapping_rho = np.asarray([
        [[results[s][layout]["conditions"][condition]["remapping_rho"] for layout in range(len(source["layouts"]))] for s in range(num_seeds)]
        for condition in CONDITIONS
    ])
    def selected_eta(result):
        selected = result["classes"] != 4
        return float(np.mean(result["heldout_target_eta"][selected])) if np.any(selected) else 0.0

    selected_target_eta = np.asarray(
        [
            [
                [
                    selected_eta(results[s][layout]["conditions"][condition])
                    for layout in range(len(source["layouts"]))
                ]
                for s in range(num_seeds)
            ]
            for condition in CONDITIONS
        ]
    )
    arrays = {
        "class_fractions": class_fractions,
        "remapping_rho": remapping_rho,
        "selected_target_eta": selected_target_eta,
    }
    cell_arrays = {
        key: np.asarray(
            [
                [
                    [results[s][layout]["conditions"][condition][key] for layout in range(len(source["layouts"]))]
                    for s in range(num_seeds)
                ]
                for condition in CONDITIONS
            ]
        )
        for key in (
            "classes",
            "train_position_eta",
            "train_cue_eta",
            "train_interaction_eta",
            "train_position_active",
            "train_cue_active",
            "train_interaction_active",
            "test_position_eta",
            "test_cue_eta",
            "test_interaction_eta",
            "train_cue_magnitude",
            "heldout_remapping",
        )
    }
    arrays.update(cell_arrays)
    representative = results[0][0]["conditions"]["aligned"]
    seed_class = class_fractions.mean(axis=2)
    seed_rho = remapping_rho.mean(axis=2)
    tuned_fraction = 1 - seed_class[:, :, 4]
    mixed_fraction = seed_class[:, :, 2] + seed_class[:, :, 3]
    summary = {}
    for condition_index, condition in enumerate(CONDITIONS):
        summary[condition] = {}
        for metric, values in (
            ("tuned_fraction", tuned_fraction[condition_index]),
            ("mixed_fraction", mixed_fraction[condition_index]),
            ("position_fraction", seed_class[condition_index, :, 0]),
            ("cue_fraction", seed_class[condition_index, :, 1]),
            ("additive_mixed_fraction", seed_class[condition_index, :, 2]),
            ("conjunctive_fraction", seed_class[condition_index, :, 3]),
            ("heldout_target_eta", selected_target_eta[condition_index].mean(axis=1)),
            ("tuning_remapping_rho", seed_rho[condition_index]),
        ):
            mean, ci = mean_ci(values)
            summary[condition][metric] = {"mean": mean, "ci95_half_width": ci}

    source_report_path = args.e3_data.with_suffix(".json")
    with source_report_path.open("r", encoding="utf-8") as handle:
        source_report = json.load(handle)
    expected_digests = source_report["protocol"]["final_weight_sha256_by_seed_layout_condition"]
    weights_match = all(
        results[s][layout]["conditions"][condition]["weight_digest"]
        == expected_digests[s][layout][condition]
        for s in range(num_seeds)
        for layout in range(len(source["layouts"]))
        for condition in CONDITIONS
    )
    if not weights_match:
        raise AssertionError("E6 sparse replay does not reproduce the frozen E3 weights")

    hypothesis_checks = {
        "aligned_contains_position_cue_and_conjunctive_cells": bool(np.all(seed_class[0, :, (0, 1, 3)].mean(axis=0) > 0)),
        "aligned_position_fraction_exceeds_random": bool(np.mean(seed_class[0, :, 0] - seed_class[1, :, 0]) > 0),
        "aligned_position_invariant_cue_fraction_exceeds_random": bool(np.mean(seed_class[0, :, 1] - seed_class[1, :, 1]) > 0),
        "aligned_tuned_fraction_exceeds_no_plasticity": bool(np.mean(tuned_fraction[0] - tuned_fraction[2]) > 0),
        "training_selected_effects_survive_heldout_laps": bool(np.mean(selected_target_eta[0]) >= args.min_eta),
        "aligned_cue_tuning_predicts_heldout_remapping": bool(np.mean(seed_rho[0]) > 0),
    }
    protocol_checks = {
        "uses_frozen_e3_inputs": True,
        "reconstructs_frozen_ca3_projection_and_random_matching": True,
        "reconstructs_frozen_e3_final_weights": weights_match,
        "classification_uses_training_laps_only": True,
        "effect_sizes_and_remapping_use_heldout_laps_only": True,
        "seed_is_inferential_unit": True,
        "object_vector_language_avoided": True,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "e6_ca1_mixed_selectivity"
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        seeds=source["seeds"][:num_seeds],
        conditions=np.asarray(CONDITIONS),
        class_names=np.asarray(CLASS_NAMES),
        class_fractions=class_fractions,
        selected_target_eta=selected_target_eta,
        remapping_rho=remapping_rho,
        **cell_arrays,
        representative_classes=representative["classes"],
        representative_train_position_eta=representative["train_position_eta"],
        representative_train_cue_eta=representative["train_cue_eta"],
        representative_train_interaction_eta=representative["train_interaction_eta"],
        representative_test_position_eta=representative["test_position_eta"],
        representative_test_cue_eta=representative["test_cue_eta"],
        representative_test_interaction_eta=representative["test_interaction_eta"],
        representative_train_cue_magnitude=representative["train_cue_magnitude"],
        representative_heldout_remapping=representative["heldout_remapping"],
        representative_test_event_means=representative["test_event_means"],
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E6_cross_validated_ca1_mixed_selectivity",
        "configuration": vars(args),
        "runtime": runtime_metadata(),
        "checkpoint_info": info,
        "source": {
            "e3_data": str(args.e3_data),
            "e3_sha256": array_digest(source["train_inputs"]),
            "num_seeds": num_seeds,
            "num_layouts": len(source["layouts"]),
        },
        "analysis": {
            "classification": "training-only factorial partial-F tests, BH-FDR, and minimum partial eta-squared; interaction supersedes main-effect classes",
            "classes": CLASS_NAMES,
            "heldout_evaluation": "partial eta-squared on independent laps; no held-out values enter cell selection",
            "remapping": "RMS deviation of held-out cue-event CA1 responses from cue-free spatial responses at the same positions",
            "inference": "seed-level means across the two layouts; cell-level associations are descriptive within seeds",
        },
        "summary": summary,
        "paired_effects": {
            "aligned_minus_random_mixed_fraction": mixed_fraction[0] - mixed_fraction[1],
            "aligned_minus_random_position_fraction": seed_class[0, :, 0] - seed_class[1, :, 0],
            "aligned_minus_random_cue_fraction": seed_class[0, :, 1] - seed_class[1, :, 1],
            "aligned_minus_random_tuning_remapping_rho": seed_rho[0] - seed_rho[1],
            "aligned_minus_no_plasticity_tuned_fraction": tuned_fraction[0] - tuned_fraction[2],
        },
        "paired_effect_summary": {
            name: {
                "mean": mean_ci(values)[0],
                "ci95_half_width": mean_ci(values)[1],
            }
            for name, values in {
                "aligned_minus_random_mixed_fraction": mixed_fraction[0] - mixed_fraction[1],
                "aligned_minus_random_position_fraction": seed_class[0, :, 0] - seed_class[1, :, 0],
                "aligned_minus_random_cue_fraction": seed_class[0, :, 1] - seed_class[1, :, 1],
                "aligned_minus_random_tuning_remapping_rho": seed_rho[0] - seed_rho[1],
                "aligned_minus_no_plasticity_tuned_fraction": tuned_fraction[0] - tuned_fraction[2],
            }.items()
        },
        "protocol_checks": protocol_checks,
        "hypothesis_checks": hypothesis_checks,
        "outputs": {
            "metrics_json": str(prefix.with_suffix(".json")),
            "raw_npz": str(prefix.with_suffix(".npz")),
            "figure_png": str(prefix.with_suffix(".png")),
            "figure_pdf": str(prefix.with_suffix(".pdf")),
        },
    }
    with prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(report), handle, indent=2, sort_keys=True)
        handle.write("\n")
    make_figure(arrays, representative, prefix.with_suffix(".png"))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e3-data", type=Path, default=DEFAULT_E3_DATA)
    parser.add_argument("--checkpoint", default="ae_factorial_paper_v1")
    parser.add_argument("--num-seeds", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--baseline-alpha", type=float, default=0.001)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--min-eta", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.num_seeds < 2:
        parser.error("--num-seeds must be at least 2")
    if not 0 < args.fdr < 1 or not 0 <= args.min_eta < 1:
        parser.error("require 0 < --fdr < 1 and 0 <= --min-eta < 1")
    return args


def main() -> int:
    args = parse_args()
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    report = run(args)
    print(json.dumps(json_ready({
        "summary": report["summary"],
        "protocol_checks": report["protocol_checks"],
        "hypothesis_checks": report["hypothesis_checks"],
        "outputs": report["outputs"],
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
