"""Compose the four KAMemory main figures from frozen experiment arrays.

Run from any working directory:

    .venv/kamvenv/bin/python article/figures/make_main_figures.py

The script never reruns a simulation and never imports a notebook panel.  It
loads the frozen ``src/experiments/plots/*.npz`` arrays, writes one tidy source
data CSV per figure, and renders PNG, PDF, and SVG versions with a shared visual
language.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kamemory-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "src" / "experiments" / "plots"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

COLORS = {
    "aligned": "#2166AC",
    "fixed_permutation": "#C44E52",
    "decoder_rescue": "#2A9D8F",
    "random_matched": "#7B61A8",
    "no_plasticity": "#6B6B6B",
    "empirical": "#E69F00",
    "ca3": "#D8E8F5",
    "ca1": "#DCEFE7",
    "ec": "#F1E6D2",
    "plastic": "#B73E3E",
}

DISPLAY = {
    "aligned": "Aligned",
    "fixed_permutation": "Fixed permutation",
    "decoder_rescue": "Matched decoder",
    "random_matched": "Random matched",
    "no_plasticity": "No plasticity",
}

LINESTYLES = {
    "aligned": "-",
    "fixed_permutation": "--",
    "decoder_rescue": ":",
    "random_matched": (0, (3, 1, 1, 1)),
    "no_plasticity": "-.",
}

MARKERS = {
    "aligned": "o",
    "fixed_permutation": "s",
    "decoder_rescue": "D",
    "random_matched": "^",
    "no_plasticity": "v",
}

CLASS_COLORS = {
    "position": "#4C78A8",
    "cue": "#F58518",
    "additive_mixed": "#B279A2",
    "conjunctive": "#54A24B",
    "unclassified": "#D8D8D8",
}

CLASS_HATCHES = {
    "position": "///",
    "cue": "\\\\\\",
    "additive_mixed": "xx",
    "conjunctive": "...",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_npz(name: str) -> dict[str, np.ndarray]:
    path = SOURCE_DIR / f"{name}.npz"
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_ci(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    count = np.sum(np.isfinite(values), axis=axis)
    mean = np.nanmean(values, axis=axis)
    sd = np.nanstd(values, axis=axis, ddof=1)
    ci = np.divide(
        1.96 * sd,
        np.sqrt(count),
        out=np.zeros_like(np.asarray(mean, dtype=float)),
        where=count > 1,
    )
    return mean, ci


def nanmean_without_warning(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """NaN-aware mean that leaves structurally empty cells as NaN."""

    values = np.asarray(values, dtype=float)
    count = np.sum(np.isfinite(values), axis=axis)
    total = np.nansum(values, axis=axis)
    return np.divide(
        total,
        count,
        out=np.full_like(np.asarray(total, dtype=float), np.nan),
        where=count > 0,
    )


def tidy_axis(ax: plt.Axes, *, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#D9D9D9", linewidth=0.55, alpha=0.65)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def node(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    *,
    edgecolor: str = "#555555",
    fontsize: float = 7.2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#555555",
    linewidth: float = 1.2,
    style: str = "-|>",
    linestyle: str = "-",
    mutation_scale: float = 10,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
        )
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.{suffix}" for suffix in ("png", "pdf", "svg")]
    figure.savefig(paths[0], dpi=400, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(paths[1], bbox_inches="tight", pad_inches=0.04)
    figure.savefig(paths[2], bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    return paths


SOURCE_FIELDS = ("figure", "panel", "series", "seed", "x", "value", "detail")


def write_source_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SOURCE_FIELDS})


def condition_points(
    ax: plt.Axes,
    values: np.ndarray,
    names: list[str],
    *,
    ylabel: str,
    chance: float | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    for index, name in enumerate(names):
        seed_values = np.asarray(values[index], dtype=float)
        jitter = np.linspace(-0.115, 0.115, len(seed_values))
        ax.scatter(
            index + jitter,
            seed_values,
            s=15,
            color=COLORS[name],
            alpha=0.60,
            linewidths=0,
            zorder=2,
        )
        mean, ci = mean_ci(seed_values)
        ax.errorbar(
            index,
            mean,
            yerr=ci,
            fmt="o",
            color="#111111",
            markerfacecolor="white",
            markeredgewidth=0.9,
            markersize=4.5,
            capsize=2.5,
            linewidth=1,
            zorder=4,
        )
        ax.text(index, float(mean + ci) + 0.025, f"{mean:.2f}", ha="center", fontsize=6.2)
    if chance is not None:
        ax.axhline(chance, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(names)), [DISPLAY[name] for name in names], rotation=24, ha="right")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    tidy_axis(ax)


def make_figure_1(output_dir: Path) -> dict[str, object]:
    e1 = load_npz("e1_alignment")
    conditions = e1["conditions"].tolist()
    aligned_index = conditions.index("aligned")
    seed_scores = e1["cosine"][aligned_index].mean(axis=1)
    median = float(np.median(seed_scores))
    seed_index = int(np.argmin(np.abs(seed_scores - median)))
    memory_scores = e1["cosine"][aligned_index, seed_index]
    memory_index = int(np.argmin(np.abs(memory_scores - np.median(memory_scores))))

    figure = plt.figure(figsize=(7.2, 5.35))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.02, 0.98), hspace=0.34, wspace=0.25)
    architecture_ax = figure.add_subplot(grid[0, 0])
    rule_ax = figure.add_subplot(grid[0, 1])
    alignment_ax = figure.add_subplot(grid[1, 0])
    example_ax = figure.add_subplot(grid[1, 1])

    # A — implemented architecture.
    architecture_ax.set_xlim(0, 1)
    architecture_ax.set_ylim(0, 1)
    architecture_ax.axis("off")
    panel_label(architecture_ax, "A", -0.05, 1.03)
    architecture_ax.set_title("Implemented EC–CA3–CA1 memory", pad=8)
    node(architecture_ax, (0.03, 0.57), 0.19, 0.18, "EC input\n$x$", COLORS["ec"])
    node(architecture_ax, (0.31, 0.57), 0.19, 0.18, "CA3 code\n$h(x)$", COLORS["ca3"])
    node(architecture_ax, (0.59, 0.57), 0.19, 0.18, "CA1 state\n$z$", COLORS["ca1"])
    node(architecture_ax, (0.81, 0.57), 0.16, 0.18, "EC output\n$\\hat{x}$", COLORS["ec"])
    arrow(architecture_ax, (0.22, 0.66), (0.31, 0.66))
    arrow(
        architecture_ax,
        (0.50, 0.66),
        (0.59, 0.66),
        color=COLORS["plastic"],
        linewidth=2.1,
    )
    arrow(architecture_ax, (0.78, 0.66), (0.81, 0.66))
    architecture_ax.text(0.265, 0.79, "fixed", ha="center", color="#555555", fontsize=6.2)
    architecture_ax.text(0.545, 0.79, "plastic $W$", ha="center", color=COLORS["plastic"], fontsize=6.2)
    architecture_ax.text(0.795, 0.79, "fixed $D$", ha="center", color="#555555", fontsize=6.2)
    node(architecture_ax, (0.25, 0.16), 0.28, 0.17, "pretrained encoder\n$c=E(x)$", "#EEF2F7")
    arrow(architecture_ax, (0.13, 0.57), (0.30, 0.33), color=COLORS["aligned"], linestyle="--")
    arrow(architecture_ax, (0.53, 0.245), (0.66, 0.57), color=COLORS["aligned"], linestyle="--")
    architecture_ax.text(0.60, 0.36, "instructive target", color=COLORS["aligned"], fontsize=6.3, ha="center")
    architecture_ax.text(0.50, 0.02, "Storage changes only CA3→CA1; retrieval is read-only", ha="center", fontsize=6.4, color="#444444")

    # B — exact implemented update.
    rule_ax.set_xlim(0, 1)
    rule_ax.set_ylim(0, 1)
    rule_ax.axis("off")
    panel_label(rule_ax, "B", -0.05, 1.03)
    rule_ax.set_title("Exact target-gated update", pad=8)
    rule_ax.text(
        0.5,
        0.79,
        r"$W_{t+1}=(1-\alpha c_t)\odot W_t+\alpha c_t h_t^{\mathsf{T}}$",
        ha="center",
        va="center",
        fontsize=12,
    )
    node(rule_ax, (0.04, 0.43), 0.22, 0.17, "CA3 activity\n$h_t$", COLORS["ca3"])
    node(rule_ax, (0.39, 0.43), 0.22, 0.17, "CA1 target\n$c_t$", COLORS["ca1"])
    node(rule_ax, (0.74, 0.43), 0.22, 0.17, "updated rows\nof $W$", "#F6DDDD")
    arrow(rule_ax, (0.26, 0.515), (0.39, 0.515))
    rule_ax.text(0.325, 0.65, "outer product", ha="center", fontsize=6.0)
    arrow(rule_ax, (0.61, 0.515), (0.74, 0.515), color=COLORS["plastic"], linewidth=1.7)
    rule_ax.text(0.5, 0.24, "Active target coordinates overwrite toward the current CA3 code", ha="center", fontsize=7.0)
    rule_ax.text(0.5, 0.10, "BTSP-inspired activity-level rule; no explicit eligibility or plateau timing trace", ha="center", fontsize=6.4, color="#555555")

    # C — alignment and matched rescue logic.
    alignment_ax.set_xlim(0, 1)
    alignment_ax.set_ylim(0, 1)
    alignment_ax.axis("off")
    panel_label(alignment_ax, "C", -0.05, 1.03)
    alignment_ax.set_title("Coordinate compatibility, not information loss", pad=8)
    y_positions = (0.75, 0.46, 0.17)
    labels = (
        ("Aligned", "$c$  — $D$ →  $x$", COLORS["aligned"], "readable"),
        ("Permuted", "$Pc$  — $D$ →  not $x$", COLORS["fixed_permutation"], "mismatch"),
        ("Rescued", "$Pc$  — $DP^{\\mathsf{T}}$ →  $x$", COLORS["decoder_rescue"], "readable"),
    )
    for y, (label, formula, color, outcome) in zip(y_positions, labels):
        alignment_ax.text(0.02, y, label, ha="left", va="center", color=color, fontweight="bold")
        node(alignment_ax, (0.23, y - 0.08), 0.53, 0.16, formula, "#F7F7F7", edgecolor=color, fontsize=9)
        alignment_ax.text(0.96, y, outcome, ha="right", va="center", color=color, fontsize=6.7)
    alignment_ax.text(0.50, 0.01, "Fixed and rescued conditions learn identical plastic weights", ha="center", fontsize=6.4, color="#444444")

    # D — representative stored content selected by a reproducible median rule.
    example_ax.axis("off")
    panel_label(example_ax, "D", -0.05, 1.03)
    seed = int(e1["seeds"][seed_index])
    example_ax.set_title(f"Representative stored memory (seed {seed})", pad=8)
    items = [
        ("Target", e1["patterns"][seed_index, memory_index], None),
        (
            "Aligned",
            e1["outputs"][conditions.index("aligned"), seed_index, memory_index],
            e1["cosine"][conditions.index("aligned"), seed_index, memory_index],
        ),
        (
            "Permuted",
            e1["outputs"][conditions.index("fixed_permutation"), seed_index, memory_index],
            e1["cosine"][conditions.index("fixed_permutation"), seed_index, memory_index],
        ),
        (
            "Rescued",
            e1["outputs"][conditions.index("decoder_rescue"), seed_index, memory_index],
            e1["cosine"][conditions.index("decoder_rescue"), seed_index, memory_index],
        ),
    ]
    for index, (label, values, score) in enumerate(items):
        inset = example_ax.inset_axes([0.02 + index * 0.245, 0.31, 0.215, 0.42])
        inset.imshow(values.reshape(5, 10), cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title(label, fontsize=6.8, pad=3)
        for spine in inset.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.6)
        if score is not None:
            example_ax.text(0.127 + index * 0.245, 0.22, f"cos = {score:.2f}", ha="center", fontsize=6.3)
    example_ax.text(0.5, 0.04, "Example rule: median seed, then memory nearest that seed's median", ha="center", fontsize=6.0, color="#555555")

    paths = save_figure(figure, output_dir, "figure_1_model_and_rule")
    rows: list[dict[str, object]] = []
    for label, values, score in items:
        for unit, value in enumerate(values):
            rows.append(
                {
                    "figure": 1,
                    "panel": "D",
                    "series": label,
                    "seed": seed,
                    "x": unit,
                    "value": float(value),
                    "detail": "" if score is None else f"cosine={float(score):.8f};memory_index={memory_index}",
                }
            )
    source_path = output_dir / "figure_1_source_data.csv"
    write_source_csv(source_path, rows)
    return {
        "files": paths,
        "source_data": source_path,
        "inputs": [SOURCE_DIR / "e1_alignment.npz"],
        "representative": {"seed": seed, "seed_index": seed_index, "memory_index": memory_index},
    }


def make_figure_2(output_dir: Path) -> dict[str, object]:
    e1 = load_npz("e1_alignment")
    e4 = load_npz("e4_sensitivity")
    conditions = e1["conditions"].tolist()
    order = ["aligned", "fixed_permutation", "decoder_rescue", "random_matched", "no_plasticity"]
    indices = [conditions.index(name) for name in order]
    corrected = e1["chance_corrected"][indices].mean(axis=2)
    top_k = e1["top_k_f1"][indices].mean(axis=2)

    figure = plt.figure(figsize=(7.2, 6.8))
    grid = figure.add_gridspec(3, 2, height_ratios=(0.42, 1.0, 1.0), hspace=0.88, wspace=0.34)
    design_ax = figure.add_subplot(grid[0, :])
    primary_ax = figure.add_subplot(grid[1, 0])
    f1_ax = figure.add_subplot(grid[1, 1])
    rescue_ax = figure.add_subplot(grid[2, 0])
    dose_ax = figure.add_subplot(grid[2, 1])

    design_ax.set_xlim(0, 1)
    design_ax.set_ylim(0, 1)
    design_ax.axis("off")
    panel_label(design_ax, "A", -0.025, 1.03)
    design_ax.set_title("Five paired controls isolate decoder compatibility", pad=7)
    descriptions = {
        "aligned": "$c$, fixed $D$",
        "fixed_permutation": "$Pc$, fixed $D$",
        "decoder_rescue": "$Pc$, matched $DP^T$",
        "random_matched": "$c_j$, fixed $D$",
        "no_plasticity": "no update, fixed $D$",
    }
    for index, name in enumerate(order):
        x = 0.012 + index * 0.198
        node(design_ax, (x, 0.24), 0.178, 0.48, f"{DISPLAY[name]}\n{descriptions[name]}", "#F8F8F8", edgecolor=COLORS[name], fontsize=6.7)

    panel_label(primary_ax, "B")
    primary_ax.set_title("Content decoding after 28 stores")
    condition_points(
        primary_ax,
        corrected,
        order,
        ylabel="Chance-corrected cosine",
        chance=0,
        ylim=(-0.09, 0.68),
    )

    panel_label(f1_ax, "C")
    f1_ax.set_title("Active content recovered")
    condition_points(f1_ax, top_k, order, ylabel="Top-K F1", chance=0.10, ylim=(0, 0.72))

    panel_label(rescue_ax, "D")
    rescue_ax.set_title("Matched readout rescues identical learned weights")
    fixed = corrected[order.index("fixed_permutation")]
    rescue = corrected[order.index("decoder_rescue")]
    for left, right in zip(fixed, rescue):
        rescue_ax.plot((0, 1), (left, right), color="#B8B8B8", linewidth=0.75, alpha=0.75, zorder=1)
    rescue_ax.scatter(np.zeros_like(fixed), fixed, color=COLORS["fixed_permutation"], s=16, alpha=0.75, zorder=2)
    rescue_ax.scatter(np.ones_like(rescue), rescue, color=COLORS["decoder_rescue"], s=16, alpha=0.75, zorder=2)
    for index, values in enumerate((fixed, rescue)):
        mean, ci = mean_ci(values)
        rescue_ax.errorbar(index, mean, yerr=ci, fmt="o", color="#111111", markerfacecolor="white", capsize=3, zorder=4)
    rescue_ax.set_xticks((0, 1), ("Fixed permutation", "Matched decoder"))
    rescue_ax.set_ylabel("Chance-corrected cosine")
    rescue_ax.set_ylim(-0.09, 0.68)
    rescue_ax.axhline(0, color="#333333", linestyle="--", linewidth=0.8)
    rescue_ax.text(0.5, 0.95, "$W_{fixed}=W_{rescue}$", transform=rescue_ax.transAxes, ha="center", va="top", fontsize=7.0)
    tidy_axis(rescue_ax)

    panel_label(dose_ax, "E")
    dose_ax.set_title("Decoding falls smoothly with coordinate mismatch")
    fraction = e4["misalignment_grid"]
    scores = e4["misalignment_scores"][:, :, 0]
    for seed_values in scores:
        dose_ax.plot(100 * fraction, seed_values, color=COLORS["aligned"], alpha=0.10, linewidth=0.7)
    mean, ci = mean_ci(scores, axis=0)
    dose_ax.errorbar(100 * fraction, mean, yerr=ci, color=COLORS["aligned"], marker="o", capsize=2.5, linewidth=1.6, label="Fixed decoder")
    rescue_values = e4["misalignment_rescue"]
    rescue_mean, rescue_ci = mean_ci(rescue_values)
    dose_ax.errorbar(103, rescue_mean, yerr=rescue_ci, color=COLORS["decoder_rescue"], marker="D", capsize=3, linewidth=0, label="Matched decoder at 100%")
    dose_ax.set_xlabel("Instructive coordinates permuted (%)")
    dose_ax.set_ylabel("Chance-corrected cosine")
    dose_ax.set_xlim(-3, 108)
    dose_ax.set_ylim(-0.05, 0.72)
    dose_ax.axhline(0, color="#333333", linestyle="--", linewidth=0.8)
    dose_ax.legend(frameon=False, loc="lower left")
    tidy_axis(dose_ax)

    paths = save_figure(figure, output_dir, "figure_2_alignment_causality")
    rows: list[dict[str, object]] = []
    seeds = e1["seeds"]
    for condition_index, name in enumerate(order):
        for seed_index, seed in enumerate(seeds):
            rows.extend(
                [
                    {"figure": 2, "panel": "B", "series": name, "seed": int(seed), "x": 28, "value": float(corrected[condition_index, seed_index]), "detail": "seed mean across memories"},
                    {"figure": 2, "panel": "C", "series": name, "seed": int(seed), "x": 28, "value": float(top_k[condition_index, seed_index]), "detail": "seed mean across memories"},
                ]
            )
    for seed_index, seed in enumerate(e4["seeds"]):
        for level_index, level in enumerate(fraction):
            rows.append({"figure": 2, "panel": "E", "series": "fixed_decoder", "seed": int(seed), "x": float(level), "value": float(scores[seed_index, level_index]), "detail": "fraction coordinates permuted"})
        rows.append({"figure": 2, "panel": "E", "series": "matched_decoder", "seed": int(seed), "x": 1.0, "value": float(rescue_values[seed_index]), "detail": "full permutation rescue"})
    source_path = output_dir / "figure_2_source_data.csv"
    write_source_csv(source_path, rows)
    return {
        "files": paths,
        "source_data": source_path,
        "inputs": [SOURCE_DIR / "e1_alignment.npz", SOURCE_DIR / "e4_sensitivity.npz"],
    }


def seed_age_curves(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=float)
    curves = np.full((matrices.shape[0], matrices.shape[1]), np.nan)
    for seed_index, matrix in enumerate(matrices):
        for age in range(matrix.shape[0]):
            loads = np.arange(age, matrix.shape[0])
            memories = loads - age
            curves[seed_index, age] = np.nanmean(matrix[loads, memories])
    return curves


def degraded_seed_means(values: np.ndarray) -> np.ndarray:
    # condition × seed × corruption × replicate × memory -> condition × seed × corruption
    return np.asarray(values, dtype=float).mean(axis=(3, 4))


def plot_degraded(
    ax: plt.Axes,
    levels: np.ndarray,
    values: np.ndarray,
    names: list[str],
    nearest: np.ndarray,
    title: str,
) -> None:
    for condition_index, name in enumerate(names):
        mean, ci = mean_ci(values[condition_index], axis=0)
        ax.plot(
            100 * levels,
            mean,
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            marker=MARKERS[name],
            markersize=3,
            linewidth=1.2,
            label=DISPLAY[name],
        )
        ax.fill_between(100 * levels, mean - ci, mean + ci, color=COLORS[name], alpha=0.12, linewidth=0)
    nearest_mean, _ = mean_ci(nearest, axis=0)
    ax.plot(100 * levels, nearest_mean, color="#111111", linestyle=":", marker="s", markersize=3, linewidth=1, label="Nearest neighbor")
    ax.axhline(1 / 8, color="#333333", linestyle="--", linewidth=0.7)
    ax.set_title(title, fontsize=7.2)
    ax.set_xlabel("Cue corruption (%)")
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    tidy_axis(ax)


def make_figure_3(output_dir: Path) -> dict[str, object]:
    e2a = load_npz("e2a_retention")
    e2b = load_npz("e2b_degraded_cues")
    e5 = load_npz("e5_interference_analysis")
    conditions = e2a["conditions"].tolist()
    order = ["aligned", "fixed_permutation", "decoder_rescue", "no_plasticity"]
    indices = [conditions.index(name) for name in order]
    threshold = float(e2a["capacity_threshold"])

    figure = plt.figure(figsize=(7.2, 8.7))
    grid = figure.add_gridspec(3, 2, hspace=0.68, wspace=0.48)
    heat_ax = figure.add_subplot(grid[0, 0])
    age_ax = figure.add_subplot(grid[0, 1])
    capacity_ax = figure.add_subplot(grid[1, 0])
    degraded_grid = grid[1, 1].subgridspec(1, 2, wspace=0.40)
    mask_ax = figure.add_subplot(degraded_grid[0, 0])
    flip_ax = figure.add_subplot(degraded_grid[0, 1])
    mechanism_ax = figure.add_subplot(grid[2, 0])
    half_life_ax = figure.add_subplot(grid[2, 1])

    panel_label(heat_ax, "A")
    heat_ax.set_title("Sequential recall across storage")
    aligned_matrix = nanmean_without_warning(
        e2a["chance_corrected"][conditions.index("aligned")], axis=0
    )
    image = heat_ax.imshow(aligned_matrix, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    heat_ax.set_xlabel("Memory storage order")
    heat_ax.set_ylabel("Total memories stored")
    heat_ax.set_xticks((0, 19, 39, 59), (1, 20, 40, 60))
    heat_ax.set_yticks((0, 19, 39, 59), (1, 20, 40, 60))
    colorbar = figure.colorbar(image, ax=heat_ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Chance-corrected recall", fontsize=6.5)

    panel_label(age_ax, "B")
    age_ax.set_title("Forgetting is gradual and age-dependent")
    all_age_curves: dict[str, np.ndarray] = {}
    ages = np.arange(e2a["chance_corrected"].shape[-1])
    for source_index, name in zip(indices, order):
        curves = seed_age_curves(e2a["chance_corrected"][source_index])
        all_age_curves[name] = curves
        mean, ci = mean_ci(curves, axis=0)
        age_ax.plot(
            ages,
            mean,
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            linewidth=1.5,
            label=DISPLAY[name],
        )
        age_ax.fill_between(ages, mean - ci, mean + ci, color=COLORS[name], alpha=0.12, linewidth=0)
    age_ax.axhline(threshold, color="#333333", linestyle="--", linewidth=0.8)
    age_ax.set_xlabel("Memory age (subsequent stores)")
    age_ax.set_ylabel("Chance-corrected recall")
    age_ax.set_ylim(-0.05, 1.02)
    age_ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=5.6)
    age_ax.text(58, threshold + 0.025, "threshold", ha="right", fontsize=5.8, color="#444444")
    tidy_axis(age_ax)

    panel_label(capacity_ax, "C")
    capacity_ax.set_title("Aligned storage has finite capacity")
    capacities = e2a["load_capacity"][indices]
    for index, name in enumerate(order):
        values = capacities[index].astype(float)
        capacity_ax.scatter(index + np.linspace(-0.11, 0.11, len(values)), values, s=14, color=COLORS[name], alpha=0.65)
        mean, ci = mean_ci(values)
        capacity_ax.errorbar(index, mean, yerr=ci, fmt="o", color="#111111", markerfacecolor="white", capsize=2.5, zorder=4)
        capacity_ax.text(index, mean + ci + 0.65, f"{mean:.1f}", ha="center", fontsize=6.2)
    capacity_ax.set_xticks(range(4), [DISPLAY[name] for name in order], rotation=24, ha="right")
    capacity_ax.set_ylabel("Contiguous load capacity")
    capacity_ax.set_ylim(-0.5, max(12, float(np.nanmax(capacities)) + 2))
    tidy_axis(capacity_ax)

    panel_label(mask_ax, "D", -0.28, 1.13)
    mask_ax.text(1.15, 1.24, "Retrieval from degraded cues", transform=mask_ax.transAxes, ha="center", fontsize=8.2)
    mask_ax.text(1.15, 1.12, "matched decoder overlays aligned; dotted = nearest neighbor", transform=mask_ax.transAxes, ha="center", fontsize=5.5, color="#555555")
    e2b_conditions = e2b["conditions"].tolist()
    e2b_indices = [e2b_conditions.index(name) for name in order]
    mask_values = degraded_seed_means(e2b["mask_identity_accuracy"][e2b_indices])
    flip_values = degraded_seed_means(e2b["flip_identity_accuracy"][e2b_indices])
    nearest_mask = e2b["nearest_mask_identity"].mean(axis=(2, 3))
    nearest_flip = e2b["nearest_flip_identity"].mean(axis=(2, 3))
    plot_degraded(mask_ax, e2b["mask_levels"], mask_values, order, nearest_mask, "Masked bits")
    plot_degraded(flip_ax, e2b["bit_flip_levels"], flip_values, order, nearest_flip, "Bit flips")
    mask_ax.set_ylabel("Correct memory identity")
    flip_ax.set_yticklabels([])
    if mask_ax.get_legend() is not None:
        mask_ax.get_legend().remove()

    panel_label(mechanism_ax, "E")
    mechanism_ax.set_title("Trace survival and crosstalk predict recall")
    correlations = np.vstack((e5["trace_recall_correlation"], e5["crosstalk_recall_correlation"]))
    correlation_names = ("Trace survival", "Crosstalk")
    correlation_colors = (COLORS["aligned"], COLORS["fixed_permutation"])
    for index, (values, color) in enumerate(zip(correlations, correlation_colors)):
        mechanism_ax.scatter(index + np.linspace(-0.10, 0.10, len(values)), values, color=color, s=15, alpha=0.65)
        mean, ci = mean_ci(values)
        mechanism_ax.errorbar(index, mean, yerr=ci, fmt="o", color="#111111", markerfacecolor="white", capsize=2.5, zorder=4)
        mechanism_ax.text(index, mean + (0.07 if mean >= 0 else -0.09), f"$r$={mean:.2f}", ha="center", fontsize=6.4)
    mechanism_ax.axhline(0, color="#333333", linestyle="--", linewidth=0.8)
    mechanism_ax.set_xticks((0, 1), correlation_names)
    mechanism_ax.set_ylabel("Within-seed correlation with recall")
    mechanism_ax.set_ylim(-0.86, 0.86)
    tidy_axis(mechanism_ax)

    panel_label(half_life_ax, "F")
    half_life_ax.set_title("Learning rate sets forgetting timescale")
    alpha = e5["alpha_values"]
    half_life = e5["alpha_recall_half_life"]
    for seed_values in half_life.T:
        half_life_ax.plot(alpha, seed_values, color=COLORS["aligned"], alpha=0.12, linewidth=0.7)
    mean, ci = mean_ci(half_life, axis=1)
    half_life_ax.errorbar(alpha, mean, yerr=ci, color=COLORS["aligned"], marker="o", capsize=2.5, linewidth=1.6)
    for x, value in zip(alpha, mean):
        half_life_ax.text(x, value + 1.6, f"{value:.1f}", ha="center", fontsize=6.2)
    half_life_ax.set_xlabel(r"Learning rate $\alpha$")
    half_life_ax.set_ylabel("Recall half-life (stores)")
    half_life_ax.set_ylim(0, 42)
    tidy_axis(half_life_ax)

    paths = save_figure(figure, output_dir, "figure_3_memory_and_interference")
    rows: list[dict[str, object]] = []
    for name in order:
        for seed_index, seed in enumerate(e2a["seeds"]):
            for age, value in enumerate(all_age_curves[name][seed_index]):
                rows.append({"figure": 3, "panel": "B", "series": name, "seed": int(seed), "x": age, "value": float(value), "detail": "seed mean across valid load-memory diagonal"})
            rows.append({"figure": 3, "panel": "C", "series": name, "seed": int(seed), "x": 60, "value": float(capacities[order.index(name), seed_index]), "detail": f"threshold={threshold:.8f}"})
    for condition_index, name in enumerate(order):
        for seed_index, seed in enumerate(e2b["seeds"]):
            for level_index, level in enumerate(e2b["mask_levels"]):
                rows.append({"figure": 3, "panel": "D-mask", "series": name, "seed": int(seed), "x": float(level), "value": float(mask_values[condition_index, seed_index, level_index]), "detail": "identity accuracy"})
            for level_index, level in enumerate(e2b["bit_flip_levels"]):
                rows.append({"figure": 3, "panel": "D-flip", "series": name, "seed": int(seed), "x": float(level), "value": float(flip_values[condition_index, seed_index, level_index]), "detail": "identity accuracy"})
    for seed_index, seed in enumerate(e5["seeds"]):
        rows.extend(
            [
                {"figure": 3, "panel": "E", "series": "trace_survival", "seed": int(seed), "x": "", "value": float(correlations[0, seed_index]), "detail": "within-seed correlation with recall"},
                {"figure": 3, "panel": "E", "series": "crosstalk", "seed": int(seed), "x": "", "value": float(correlations[1, seed_index]), "detail": "within-seed correlation with recall"},
            ]
        )
        for alpha_index, value in enumerate(alpha):
            rows.append({"figure": 3, "panel": "F", "series": "recall_half_life", "seed": int(seed), "x": float(value), "value": float(half_life[alpha_index, seed_index]), "detail": "subsequent stores"})
    source_path = output_dir / "figure_3_source_data.csv"
    write_source_csv(source_path, rows)
    return {
        "files": paths,
        "source_data": source_path,
        "inputs": [SOURCE_DIR / "e2a_retention.npz", SOURCE_DIR / "e2b_degraded_cues.npz", SOURCE_DIR / "e5_interference_analysis.npz"],
    }


def cue_tuning_dose_response(cue_magnitude: np.ndarray, remapping: np.ndarray) -> np.ndarray:
    cue_magnitude = np.asarray(cue_magnitude, dtype=float)
    remapping = np.asarray(remapping, dtype=float)
    masks = (
        cue_magnitude <= 1e-12,
        (cue_magnitude > 1e-12) & (cue_magnitude <= 0.33),
        (cue_magnitude > 0.33) & (cue_magnitude <= 0.67),
        cue_magnitude > 0.67,
    )
    response = np.full((cue_magnitude.shape[0], 4), np.nan)
    for seed_index in range(cue_magnitude.shape[0]):
        for group_index, mask in enumerate(masks):
            selected = mask[seed_index]
            if np.any(selected):
                response[seed_index, group_index] = np.mean(remapping[seed_index][selected])
    return response


def conditional_spatial(table: np.ndarray) -> np.ndarray:
    table = np.asarray(table, dtype=float)
    return np.divide(table[:, 0], table.sum(axis=1), out=np.full(2, np.nan), where=table.sum(axis=1) > 0)


def make_figure_4(output_dir: Path) -> dict[str, object]:
    e6 = load_npz("e6_ca1_mixed_selectivity")
    e7 = load_npz("e7_ca1_data_comparison")
    e9 = load_npz("e9_is_heterogeneity")

    figure = plt.figure(figsize=(7.2, 8.9))
    grid = figure.add_gridspec(3, 2, hspace=0.68, wspace=0.48)
    task_ax = figure.add_subplot(grid[0, 0])
    responses_ax = figure.add_subplot(grid[0, 1])
    classes_ax = figure.add_subplot(grid[1, 0])
    remap_ax = figure.add_subplot(grid[1, 1])
    data_ax = figure.add_subplot(grid[2, 0])
    boundary_ax = figure.add_subplot(grid[2, 1])

    panel_label(task_ax, "A")
    task_ax.set_title("Factorial cue × position task")
    task_ax.set_xlim(-0.8, 4.2)
    task_ax.set_ylim(-1.05, 2.05)
    task_ax.axis("off")
    cue_colors = ("#9ECAE1", "#FDD0A2")
    for cue in range(2):
        for position in range(4):
            task_ax.add_patch(Rectangle((position - 0.38, 1 - cue - 0.28), 0.76, 0.56, facecolor=cue_colors[cue], edgecolor="#666666", linewidth=0.6))
            task_ax.text(position, 1 - cue, f"C{cue + 1}\nP{position + 1}", ha="center", va="center", fontsize=6.5)
    task_ax.text(-0.58, 1, "Cue 1", ha="center", va="center", rotation=90, fontsize=6.5)
    task_ax.text(-0.58, 0, "Cue 2", ha="center", va="center", rotation=90, fontsize=6.5)
    for position in range(4):
        task_ax.text(position, 1.52, f"position {position + 1}", ha="center", fontsize=6.0)
    task_ax.add_patch(Rectangle((-0.38, -0.78), 3.76, 0.32, facecolor="#EEEEEE", edgecolor="#777777", linewidth=0.6))
    task_ax.text(1.5, -0.62, "cue-free laps probe spatial fields", ha="center", va="center", fontsize=6.2)
    arrow(task_ax, (3.48, 0.50), (4.05, 0.50), color=COLORS["aligned"], linewidth=1.3)
    task_ax.text(3.77, 0.66, "held-out\nlaps", ha="center", fontsize=6.0, color=COLORS["aligned"])

    panel_label(responses_ax, "B")
    responses_ax.set_title("Training-classified cells generalize")
    classes = e6["representative_classes"]
    event_means = e6["representative_test_event_means"]
    selections: list[tuple[str, np.ndarray]] = []
    selection_specs = (("Position only", 0, 7, e6["representative_test_position_eta"]), ("Cue only", 1, 2, e6["representative_test_cue_eta"]), ("Conjunctive", 3, 8, e6["representative_test_interaction_eta"]))
    selected_indices = []
    group_ticks = []
    row_start = 0
    for label, class_index, limit, ranking in selection_specs:
        candidates = np.flatnonzero(classes == class_index)
        chosen = candidates[np.argsort(ranking[candidates])[::-1]][:limit]
        selections.append((label, chosen))
        selected_indices.extend(chosen.tolist())
        group_ticks.append((row_start + (len(chosen) - 1) / 2, label))
        row_start += len(chosen)
    selected_indices_array = np.asarray(selected_indices, dtype=int)
    response_matrix = event_means[:, :, selected_indices_array].transpose(2, 0, 1).reshape(len(selected_indices), 8)
    image = responses_ax.imshow(response_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    responses_ax.set_xticks(range(8), ("C1P1", "C1P2", "C1P3", "C1P4", "C2P1", "C2P2", "C2P3", "C2P4"), rotation=45, ha="right")
    responses_ax.set_yticks([tick for tick, _ in group_ticks], [label for _, label in group_ticks])
    boundary = 0
    for _, chosen in selections[:-1]:
        boundary += len(chosen)
        responses_ax.axhline(boundary - 0.5, color="white", linewidth=1.2)
    responses_ax.set_xlabel("Held-out cue–position condition")
    colorbar = figure.colorbar(image, ax=responses_ax, fraction=0.046, pad=0.03)
    colorbar.set_label("CA1 activity", fontsize=6.5)

    panel_label(classes_ax, "C")
    classes_ax.set_title("Sparse population is mostly conjunctive")
    class_names = e6["class_names"].tolist()
    aligned_index = e6["conditions"].tolist().index("aligned")
    class_seed = e6["class_fractions"][aligned_index].mean(axis=1)
    class_mean = class_seed.mean(axis=0)
    tuned_fraction = 1 - class_mean[class_names.index("unclassified")]
    tuned_names = ["position", "cue", "additive_mixed", "conjunctive"]
    tuned_values = np.asarray([class_mean[class_names.index(name)] for name in tuned_names])
    tuned_composition = tuned_values / tuned_values.sum()
    classes_ax.set_xlim(0, 1)
    classes_ax.set_ylim(-0.5, 1.8)
    classes_ax.set_yticks((1.15, 0.25), ("All units", "Tuned cells"))
    classes_ax.set_xticks(np.linspace(0, 1, 6))
    classes_ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    classes_ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    classes_ax.set_axisbelow(True)
    classes_ax.barh(1.15, tuned_fraction, height=0.42, color=COLORS["aligned"], label="Tuned")
    classes_ax.barh(1.15, 1 - tuned_fraction, left=tuned_fraction, height=0.42, color=CLASS_COLORS["unclassified"], label="Unclassified")
    classes_ax.annotate(
        f"{100*tuned_fraction:.1f}% tuned",
        xy=(tuned_fraction / 2, 1.36),
        xytext=(0.20, 1.62),
        arrowprops=dict(arrowstyle="-", color=COLORS["aligned"], linewidth=0.7),
        fontsize=5.9,
        ha="center",
    )
    left = 0.0
    for name, value in zip(tuned_names, tuned_composition):
        classes_ax.barh(
            0.25,
            value,
            left=left,
            height=0.42,
            color=CLASS_COLORS[name],
            hatch=CLASS_HATCHES[name],
            edgecolor="white",
            linewidth=0.35,
        )
        if value >= 0.15:
            classes_ax.text(left + value / 2, 0.25, f"{100*value:.1f}%", ha="center", va="center", color="white", fontsize=5.9)
        left += value
    classes_ax.annotate(f"cue {100*tuned_composition[1]:.1f}%", xy=(tuned_composition[0] + tuned_composition[1] / 2, 0.48), xytext=(0.24, 0.78), arrowprops=dict(arrowstyle="-", color=CLASS_COLORS["cue"], linewidth=0.7), fontsize=5.8, ha="center")
    classes_ax.annotate(f"position {100*tuned_composition[0]:.1f}%", xy=(tuned_composition[0] / 2, 0.48), xytext=(0.06, 0.88), arrowprops=dict(arrowstyle="-", color=CLASS_COLORS["position"], linewidth=0.7), fontsize=5.8, ha="left")
    classes_ax.text(0.52, -0.20, "additive mixed 0.0%", fontsize=5.8, ha="center", color=CLASS_COLORS["additive_mixed"])
    legend_handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=CLASS_COLORS[name],
            hatch=CLASS_HATCHES[name],
            edgecolor="#666666",
            linewidth=0.35,
        )
        for name in tuned_names
    ]
    classes_ax.legend(legend_handles, ("Position", "Cue", "Additive", "Conjunctive"), frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    classes_ax.spines[["top", "right", "left"]].set_visible(False)

    panel_label(remap_ax, "D")
    remap_ax.set_title("Cue tuning predicts remapping")
    dose = cue_tuning_dose_response(e6["train_cue_magnitude"][aligned_index], e6["heldout_remapping"][aligned_index])
    seed_rho = e6["remapping_rho"][aligned_index].mean(axis=1)
    x = np.arange(4)
    for seed_values in dose:
        remap_ax.plot(x, seed_values, color=COLORS["aligned"], alpha=0.16, linewidth=0.75)
    mean, ci = mean_ci(dose, axis=0)
    remap_ax.errorbar(x, mean, yerr=ci, color=COLORS["aligned"], marker="o", capsize=2.5, linewidth=1.6)
    remap_ax.set_xticks(x, ("None", "Weak", "Moderate", "Strong"))
    remap_ax.set_xlabel("Training cue-tuning group")
    remap_ax.set_ylabel("Held-out cue-evoked remapping")
    remap_ax.text(
        0.04,
        0.94,
        rf"mean seed-level $\rho={seed_rho.mean():.3f}$",
        transform=remap_ax.transAxes,
        va="top",
        fontsize=6.5,
    )
    tidy_axis(remap_ax)

    panel_label(data_ax, "E")
    data_ax.set_title("Directional agreement, excessive coupling")
    empirical = conditional_spatial(e7["empirical_table"])
    aligned_tables = e7["aligned_tables"]
    model_conditional = np.asarray([conditional_spatial(table) for table in aligned_tables])
    # Plot cue-inactive first, cue-responsive second.
    empirical_plot = empirical[[1, 0]]
    model_plot = model_conditional[:, [1, 0]]
    x = np.arange(2)
    width = 0.34
    data_ax.bar(x - width / 2, empirical_plot, width, color=COLORS["empirical"], label="Published CA1")
    data_ax.bar(x + width / 2, model_plot.mean(axis=0), width, color=COLORS["aligned"], label="Model")
    for category in range(2):
        data_ax.scatter(x[category] + width / 2 + np.linspace(-0.055, 0.055, len(model_plot)), model_plot[:, category], facecolor="white", edgecolor=COLORS["aligned"], linewidth=0.65, s=11, zorder=3)
        data_ax.text(x[category] - width / 2, empirical_plot[category] + 0.035, f"{100*empirical_plot[category]:.1f}%", ha="center", fontsize=5.9)
        data_ax.text(x[category] + width / 2, model_plot[:, category].mean() + 0.035, f"{100*model_plot[:, category].mean():.1f}%", ha="center", fontsize=5.9)
    data_ax.set_xticks(x, ("Cue inactive", "Cue responsive"))
    data_ax.set_ylabel("Cells with spatial effect")
    data_ax.set_ylim(0, 1.10)
    data_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    data_ax.legend(frameon=False, loc="lower right")
    tidy_axis(data_ax)

    panel_label(boundary_ax, "F", x=-0.18, y=1.13)
    boundary_ax.set_title("Background plateaus do not repair mismatch")
    rates = e9["background_rates"]
    error = e9["primary_absolute_log_odds_ratio_error"]
    output_seed = e9["output_cosine"].mean(axis=2)
    output_mean, output_ci = mean_ci(output_seed, axis=1)
    scatter = boundary_ax.scatter(error, output_mean, c=100 * rates, cmap="plasma", s=34, edgecolor="#333333", linewidth=0.45, zorder=3)
    boundary_ax.errorbar(error, output_mean, yerr=output_ci, fmt="none", ecolor="#555555", linewidth=0.7, capsize=2, zorder=2)
    label_offsets = {
        0: (-12, 8),
        5: (4, 8),
        6: (4, -12),
        7: (4, -12),
    }
    for index, offset in label_offsets.items():
        rate = rates[index]
        boundary_ax.annotate(f"{100*rate:g}%", (error[index], output_mean[index]), xytext=offset, textcoords="offset points", fontsize=5.8)
    boundary_ax.set_xlabel("Absolute published log-OR error\n(lower is better)")
    boundary_ax.set_ylabel("Held-out EC-output cosine\n(higher is better)")
    boundary_ax.set_xlim(error.min() - 0.12, error.max() + 0.23)
    boundary_ax.set_ylim(output_mean.min() - 0.06, output_mean.max() + 0.06)
    colorbar = figure.colorbar(scatter, ax=boundary_ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Background event rate (%)", fontsize=6.2)
    tidy_axis(boundary_ax)

    paths = save_figure(figure, output_dir, "figure_4_ca1_and_biological_boundary")
    rows: list[dict[str, object]] = []
    for row_index, cell_index in enumerate(selected_indices):
        for condition_index, value in enumerate(response_matrix[row_index]):
            rows.append({"figure": 4, "panel": "B", "series": f"cell_{cell_index}", "seed": int(e6["seeds"][0]), "x": condition_index, "value": float(value), "detail": "held-out response; class selected on training laps"})
    for seed_index, seed in enumerate(e6["seeds"]):
        for class_index, name in enumerate(class_names):
            rows.append({"figure": 4, "panel": "C", "series": name, "seed": int(seed), "x": "", "value": float(class_seed[seed_index, class_index]), "detail": "mean across paired layouts"})
        for group_index, value in enumerate(dose[seed_index]):
            rows.append({"figure": 4, "panel": "D", "series": "aligned", "seed": int(seed), "x": group_index, "value": float(value), "detail": "cue-tuning group 0:none 1:weak 2:moderate 3:strong"})
    rows.extend(
        [
            {"figure": 4, "panel": "E", "series": "published_cue_inactive", "seed": "published", "x": 0, "value": float(empirical_plot[0]), "detail": "273/477"},
            {"figure": 4, "panel": "E", "series": "published_cue_responsive", "seed": "published", "x": 1, "value": float(empirical_plot[1]), "detail": "71/108"},
        ]
    )
    for seed_index, seed in enumerate(e7["seeds"]):
        rows.append({"figure": 4, "panel": "E", "series": "model_cue_inactive", "seed": int(seed), "x": 0, "value": float(model_plot[seed_index, 0]), "detail": "pooled layouts within seed"})
        rows.append({"figure": 4, "panel": "E", "series": "model_cue_responsive", "seed": int(seed), "x": 1, "value": float(model_plot[seed_index, 1]), "detail": "pooled layouts within seed"})
    for rate_index, rate in enumerate(rates):
        for seed_index, seed in enumerate(e9["seeds"]):
            rows.append({"figure": 4, "panel": "F", "series": "output_cosine", "seed": int(seed), "x": float(rate), "value": float(output_seed[rate_index, seed_index]), "detail": f"absolute_log_or_error={float(error[rate_index]):.8f}"})
    source_path = output_dir / "figure_4_source_data.csv"
    write_source_csv(source_path, rows)
    return {
        "files": paths,
        "source_data": source_path,
        "inputs": [SOURCE_DIR / "e6_ca1_mixed_selectivity.npz", SOURCE_DIR / "e7_ca1_data_comparison.npz", SOURCE_DIR / "e9_is_heterogeneity.npz"],
        "representative": {"seed": int(e6["seeds"][0]), "selected_cell_indices": selected_indices},
    }


def make_manifest(output_dir: Path, results: list[dict[str, object]]) -> Path:
    checkpoint_lineage = {
        "figures_1_to_3": {
            "checkpoint": "ae_8",
            "stimulus_distribution": "exactly K-hot 50-dimensional sparse patterns",
            "checkpoint_files": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                }
                for path in (
                    ROOT / "src/data/autoencoders/ae_8/autoencoder.pt",
                    ROOT / "src/data/autoencoders/ae_8/info.json",
                )
            ],
        },
        "figure_4": {
            "checkpoint": "ae_factorial_paper_v1",
            "stimulus_distribution": "E3/E6/E7/E9 factorial spatial+sensory track",
            "checkpoint_files": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                }
                for path in (
                    ROOT / "src/data/autoencoders/ae_factorial_paper_v1/autoencoder.pt",
                    ROOT / "src/data/autoencoders/ae_factorial_paper_v1/info.json",
                )
            ],
        },
    }
    manifest = {
        "schema_version": 1,
        "command": ".venv/kamvenv/bin/python article/figures/make_main_figures.py",
        "principles": [
            "All quantitative panels are composed from frozen NPZ arrays.",
            "Independent points and confidence intervals use the network/data seed.",
            "Intervals are mean +/- 1.96 SEM across independent seeds.",
            "No saved experiment raster panel is imported.",
            "Figure 1 schematics reflect the implemented backend and are explicitly conceptual.",
        ],
        "checkpoint_lineage": checkpoint_lineage,
        "figures": [],
    }
    for figure_index, result in enumerate(results, start=1):
        entry = {
            "figure": figure_index,
            "outputs": [
                {"path": str(Path(path).relative_to(ROOT)), "sha256": sha256(Path(path))}
                for path in result["files"]
            ],
            "source_data": {
                "path": str(Path(result["source_data"]).relative_to(ROOT)),
                "sha256": sha256(Path(result["source_data"])),
            },
            "inputs": [
                {"path": str(Path(path).relative_to(ROOT)), "sha256": sha256(Path(path))}
                for path in result["inputs"]
            ],
        }
        if "representative" in result:
            entry["representative"] = result["representative"]
        manifest["figures"].append(entry)
    path = output_dir / "main_figure_manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        make_figure_1(args.output_dir),
        make_figure_2(args.output_dir),
        make_figure_3(args.output_dir),
        make_figure_4(args.output_dir),
    ]
    manifest = make_manifest(args.output_dir, results)
    print(
        json.dumps(
            {
                "figures": [[str(path) for path in result["files"]] for result in results],
                "source_data": [str(result["source_data"]) for result in results],
                "manifest": str(manifest),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
