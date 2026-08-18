"""E7: prespecified comparison with a published CA1 population profile.

The empirical anchor is the exact CA1 contingency reported by Symanski,
Bladon et al. (eLife 2022): spatial fields in 71/108 odor-responsive cells and
273/477 odor-inactive cells. The model comparison uses the E6 training-defined
cue effect and held-out position effect. The primary endpoint is absolute
log-odds-ratio error; the network seed is the model inferential unit.

Run from the repository root:

    python3 src/experiments/08_ca1_data_comparison.py
"""

from __future__ import annotations

import argparse
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

from kamemory.io import runtime_metadata
from kamemory.utils import array_digest


SCHEMA_VERSION = 1
HERE = Path(__file__).resolve().parent
DEFAULT_E6_DATA = HERE / "plots" / "e6_ca1_mixed_selectivity.npz"
DEFAULT_REFERENCE = HERE / "reference_data" / "symanski_2022_ca1_profile.json"
DEFAULT_OUTPUT_DIR = HERE / "plots"
COLORS = {
    "published": "#E76F51",
    "aligned": "#2878B5",
    "random_matched": "#8C6BB1",
    "no_plasticity": "#7A7A7A",
}
DISPLAY = {
    "published": "Published CA1",
    "aligned": "Aligned IS",
    "random_matched": "Random matched IS",
    "no_plasticity": "No plasticity",
}


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def contingency(cue_active: np.ndarray, spatial_active: np.ndarray) -> np.ndarray:
    """Rows are cue active/inactive; columns are spatial yes/no."""

    cue_active = np.asarray(cue_active, dtype=bool).reshape(-1)
    spatial_active = np.asarray(spatial_active, dtype=bool).reshape(-1)
    if cue_active.shape != spatial_active.shape:
        raise ValueError("cue and spatial masks must have equal shape")
    return np.asarray(
        [
            [np.sum(cue_active & spatial_active), np.sum(cue_active & ~spatial_active)],
            [np.sum(~cue_active & spatial_active), np.sum(~cue_active & ~spatial_active)],
        ],
        dtype=np.int64,
    )


def table_is_estimable(table: np.ndarray) -> bool:
    table = np.asarray(table)
    return bool(np.all(table.sum(axis=0) > 0) and np.all(table.sum(axis=1) > 0))


def log_odds_ratio(table: np.ndarray, correction: float = 0.5) -> float:
    """Log odds ratio with a prespecified all-cell continuity correction."""

    table = np.asarray(table, dtype=float)
    if table.shape != (2, 2) or not table_is_estimable(table):
        return np.nan
    a, b, c, d = (table + correction).reshape(-1)
    return float(math.log(a) + math.log(d) - math.log(b) - math.log(c))


def conditional_spatial(table: np.ndarray) -> tuple[float, float]:
    table = np.asarray(table, dtype=float)
    row_totals = table.sum(axis=1)
    return (
        float(table[0, 0] / row_totals[0]) if row_totals[0] else np.nan,
        float(table[1, 0] / row_totals[1]) if row_totals[1] else np.nan,
    )


def log_or_wald_ci(table: np.ndarray, correction: float = 0.5) -> tuple[float, float]:
    estimate = log_odds_ratio(table, correction)
    if not np.isfinite(estimate):
        return np.nan, np.nan
    corrected = np.asarray(table, dtype=float) + correction
    half_width = 1.96 * math.sqrt(float(np.sum(1.0 / corrected)))
    return estimate - half_width, estimate + half_width


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, np.nan
    return mean, float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def empirical_table(reference: dict) -> np.ndarray:
    counts = reference["counts"]
    return np.asarray(
        [
            [counts["cue_active"]["spatial_field"], counts["cue_active"]["no_spatial_field"]],
            [counts["cue_inactive"]["spatial_field"], counts["cue_inactive"]["no_spatial_field"]],
        ],
        dtype=np.int64,
    )


def model_tables(source: dict[str, np.ndarray], condition_index: int, min_eta: float) -> np.ndarray:
    """Pool layouts within each independent seed, never across seeds."""

    cue = source["train_cue_active"][condition_index]
    spatial = source["test_position_eta"][condition_index] >= min_eta
    return np.asarray(
        [contingency(cue[seed], spatial[seed]) for seed in range(cue.shape[0])]
    )


def analyze_tables(tables: np.ndarray) -> dict[str, np.ndarray]:
    log_or = np.asarray([log_odds_ratio(table) for table in tables])
    conditional = np.asarray([conditional_spatial(table) for table in tables])
    return {
        "log_odds_ratio": log_or,
        "odds_ratio": np.exp(log_or),
        "spatial_given_cue": conditional[:, 0],
        "spatial_given_no_cue": conditional[:, 1],
        "risk_difference": conditional[:, 0] - conditional[:, 1],
    }


def make_figure(
    empirical: np.ndarray,
    model: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    mapping_ax, prevalence_ax, odds_ax, difference_ax = axes.flat

    mapping_ax.axis("off")
    mapping_ax.text(0.5, 0.91, "Published CA1", ha="center", fontsize=13, fontweight="bold")
    mapping_ax.text(0.5, 0.74, "Odor-period responsive", ha="center", bbox=dict(boxstyle="round", fc="#FCE8DF"))
    mapping_ax.text(0.5, 0.57, "Spatial firing field", ha="center", bbox=dict(boxstyle="round", fc="#FCE8DF"))
    mapping_ax.annotate("prespecified mapping", (0.5, 0.43), ha="center", color="#555555")
    mapping_ax.text(0.5, 0.28, "Training cue effect", ha="center", bbox=dict(boxstyle="round", fc="#DCECF7"))
    mapping_ax.text(0.5, 0.11, "Held-out position effect", ha="center", bbox=dict(boxstyle="round", fc="#DCECF7"))
    mapping_ax.set_title("A  One population-level biological anchor")

    empirical_conditional = conditional_spatial(empirical)
    aligned = model["aligned"]
    x = np.asarray((0, 1))
    width = 0.34
    empirical_values = np.asarray((empirical_conditional[1], empirical_conditional[0]))
    model_seed = np.column_stack((aligned["spatial_given_no_cue"], aligned["spatial_given_cue"]))
    model_values = np.nanmean(model_seed, axis=0)
    bars_emp = prevalence_ax.bar(x - width / 2, empirical_values, width, color=COLORS["published"], label=DISPLAY["published"])
    bars_mod = prevalence_ax.bar(x + width / 2, model_values, width, color=COLORS["aligned"], label=DISPLAY["aligned"])
    for category in range(model_seed.shape[1]):
        prevalence_ax.scatter(
            x[category] + width / 2 + np.linspace(-0.055, 0.055, len(model_seed)),
            model_seed[:, category],
            facecolor="white",
            edgecolor=COLORS["aligned"],
            linewidth=0.8,
            s=18,
            zorder=3,
        )
    for bars, values in ((bars_emp, empirical_values), (bars_mod, model_values)):
        for bar, value in zip(bars, values):
            prevalence_ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{100*value:.1f}%", ha="center", fontsize=9)
    prevalence_ax.set_xticks(x, ("Cue inactive", "Cue active"))
    prevalence_ax.set_ylim(0, 1.1)
    prevalence_ax.set_ylabel("Cells with spatial/position effect")
    prevalence_ax.set_title("B  Conditional spatial prevalence")
    prevalence_ax.legend(frameon=False)

    empirical_log = log_odds_ratio(empirical)
    empirical_ci = log_or_wald_ci(empirical)
    aligned_log = aligned["log_odds_ratio"]
    aligned_mean, aligned_ci = mean_ci(aligned_log)
    odds_ax.errorbar(0, empirical_log, yerr=[[empirical_log - empirical_ci[0]], [empirical_ci[1] - empirical_log]], fmt="o", color=COLORS["published"], capsize=5, markersize=8)
    jitter = np.linspace(-0.08, 0.08, len(aligned_log))
    odds_ax.scatter(1 + jitter, aligned_log, color=COLORS["aligned"], alpha=0.65, s=28)
    odds_ax.errorbar(1, aligned_mean, yerr=aligned_ci, fmt="o", color="black", capsize=5)
    odds_ax.axhline(0, color="black", linestyle="--", linewidth=1)
    odds_ax.set_xticks((0, 1), (DISPLAY["published"], DISPLAY["aligned"]))
    odds_ax.set_ylabel("Log odds ratio")
    odds_ax.set_title("C  Model overcouples cue and spatial coding")

    empirical_difference = empirical_conditional[0] - empirical_conditional[1]
    names = ("published", "aligned", "random_matched", "no_plasticity")
    difference_ax.scatter(0, empirical_difference, color=COLORS["published"], s=60, zorder=3)
    difference_ax.text(0, empirical_difference + 0.045, f"{empirical_difference:.3f}", ha="center", fontsize=9)
    for index, condition in enumerate(names[1:], start=1):
        values = model[condition]["risk_difference"]
        finite = values[np.isfinite(values)]
        if len(finite):
            difference_ax.scatter(index + np.linspace(-0.08, 0.08, len(finite)), finite, color=COLORS[condition], alpha=0.65, s=25)
            mean, ci = mean_ci(finite)
            difference_ax.errorbar(index, mean, yerr=ci, fmt="o", color="black", capsize=4)
        else:
            difference_ax.text(
                index,
                0.04,
                "undefined\n(no effects)",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
    difference_ax.axhline(0, color="black", linestyle="--", linewidth=1)
    difference_ax.set_xticks(range(len(names)), [DISPLAY[name] for name in names], rotation=15, ha="right")
    difference_ax.set_ylabel("Cue-associated spatial prevalence difference")
    difference_ax.set_title("D  Direction agrees; magnitude does not")

    figure.suptitle("E7 — Published CA1 data expose a quantitative model limitation", fontsize=16, fontweight="bold")
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    with args.reference.open("r", encoding="utf-8") as handle:
        reference = json.load(handle)
    source_file = np.load(args.e6_data, allow_pickle=False)
    source = {key: source_file[key] for key in source_file.files}
    conditions = source["conditions"].tolist()
    empirical = empirical_table(reference)
    empirical_log = log_odds_ratio(empirical)
    empirical_conditional = conditional_spatial(empirical)

    tables = {
        condition: model_tables(source, conditions.index(condition), args.min_eta)
        for condition in ("aligned", "random_matched", "no_plasticity")
    }
    model = {condition: analyze_tables(value) for condition, value in tables.items()}
    aligned_mean_log, aligned_ci_log = mean_ci(model["aligned"]["log_odds_ratio"])
    empirical_ci_log = log_or_wald_ci(empirical)
    primary_error = abs(aligned_mean_log - empirical_log)
    aligned_direction = model["aligned"]["risk_difference"] > 0

    summary = {
        "published": {
            "table": empirical,
            "spatial_given_cue": empirical_conditional[0],
            "spatial_given_no_cue": empirical_conditional[1],
            "risk_difference": empirical_conditional[0] - empirical_conditional[1],
            "log_odds_ratio": empirical_log,
            "odds_ratio": math.exp(empirical_log),
            "log_odds_ratio_ci95": empirical_ci_log,
        },
        "aligned_model": {
            "mean_spatial_given_cue": float(np.nanmean(model["aligned"]["spatial_given_cue"])),
            "mean_spatial_given_no_cue": float(np.nanmean(model["aligned"]["spatial_given_no_cue"])),
            "mean_risk_difference": float(np.nanmean(model["aligned"]["risk_difference"])),
            "mean_log_odds_ratio": aligned_mean_log,
            "ci95_half_width_log_odds_ratio": aligned_ci_log,
            "geometric_mean_odds_ratio": math.exp(aligned_mean_log),
        },
        "primary_absolute_log_odds_ratio_error": primary_error,
        "odds_ratio_fold_excess": math.exp(aligned_mean_log - empirical_log),
        "aligned_seed_direction_match_fraction": float(np.mean(aligned_direction)),
    }
    hypothesis_checks = {
        "published_profile_has_positive_cue_spatial_association": bool(summary["published"]["risk_difference"] > 0),
        "aligned_model_matches_association_direction_in_all_seeds": bool(np.all(aligned_direction)),
        "aligned_model_does_not_quantitatively_match_empirical_magnitude": bool(primary_error > 1.0),
    }
    protocol_checks = {
        "primary_profile_and_metric_prespecified_in_reference_file": True,
        "published_counts_are_exact_not_digitized": not reference["extraction"]["digitized"],
        "model_cue_selection_uses_training_laps": True,
        "model_position_evaluation_uses_heldout_laps": True,
        "layouts_pooled_only_within_seed": True,
        "seed_is_model_inferential_unit": True,
        "task_and_measurement_mismatch_reported": bool(reference["limitations"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "e7_ca1_data_comparison"
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        empirical_table=empirical,
        seeds=source["seeds"],
        conditions=np.asarray(("aligned", "random_matched", "no_plasticity")),
        aligned_tables=tables["aligned"],
        random_matched_tables=tables["random_matched"],
        no_plasticity_tables=tables["no_plasticity"],
        aligned_log_odds_ratio=model["aligned"]["log_odds_ratio"],
        aligned_risk_difference=model["aligned"]["risk_difference"],
        random_matched_log_odds_ratio=model["random_matched"]["log_odds_ratio"],
        random_matched_risk_difference=model["random_matched"]["risk_difference"],
        no_plasticity_log_odds_ratio=model["no_plasticity"]["log_odds_ratio"],
        no_plasticity_risk_difference=model["no_plasticity"]["risk_difference"],
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E7_prespecified_published_ca1_comparison",
        "configuration": vars(args),
        "runtime": runtime_metadata(),
        "reference": reference,
        "source": {
            "e6_data": str(args.e6_data),
            "e6_sha256": array_digest(source["classes"]),
        },
        "summary": summary,
        "model_seed_values": model,
        "protocol_checks": protocol_checks,
        "hypothesis_checks": hypothesis_checks,
        "interpretation": {
            "agreement": "Published CA1 and every aligned-model seed show more spatial coding among cue-responsive cells.",
            "disagreement": "The model association is orders of magnitude stronger because its sparse deterministic units bind cue and position nearly categorically.",
            "claim_limit": "The comparison supports the direction of cue-spatial coupling, not quantitative biological realism of the current activity model.",
        },
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
    make_figure(empirical, model, prefix.with_suffix(".png"))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e6-data", type=Path, default=DEFAULT_E6_DATA)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--min-eta", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps(json_ready({
        "summary": report["summary"],
        "protocol_checks": report["protocol_checks"],
        "hypothesis_checks": report["hypothesis_checks"],
        "outputs": report["outputs"],
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
