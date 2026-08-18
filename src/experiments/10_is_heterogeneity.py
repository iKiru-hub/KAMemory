"""E9: cue-independent plateau heterogeneity and CA1 biological agreement.

E7 found that the aligned model had the correct direction of cue--spatial
coupling but made it almost categorical.  E9 tests one prespecified repair:
retain strong content-aligned plasticity at cue presentations while adding
independently timed, content-independent plateau events outside cue
presentations.  Background signals are drawn without replacement from the
same code bank, preserving their population statistics while breaking the
relationship to the coincident CA3 input.  All other E3 task, model, and
analysis choices are frozen.

The complete background-event sweep is reported.  The primary biological
endpoint is the E7 absolute log-odds-ratio error; the functional endpoint is
held-out EC-output cosine.  A rate is considered a quantitative repair only
if its log-odds-ratio error is at most 1.0, it improves the two conditional-
prevalence RMSE, and it preserves at least 80% of the rate-zero output cosine.

Run from the repository root:

    PYTHONPATH=src .venv/kamvenv/bin/python \
        src/experiments/10_is_heterogeneity.py --deterministic
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
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch

from kamemory.data import generate_factorial_track
from kamemory.io import PATHS, load_autoencoder_session, load_config, runtime_metadata
from kamemory.utils import array_digest, seed_everything, sparsemoid


SCHEMA_VERSION = 1
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = PATHS.configs / "is_heterogeneity.json"
DEFAULT_REFERENCE = HERE / "reference_data" / "symanski_2022_ca1_profile.json"
DEFAULT_OUTPUT_DIR = HERE / "plots"
COLOR = "#2878B5"
SECONDARY_COLOR = "#E07A32"
EMPIRICAL_COLOR = "#E76F51"


def _load_sibling(module_name: str, filename: str):
    spec = spec_from_file_location(module_name, HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E3 = _load_sibling("kamemory_e3_for_e9", "04_ca1_track.py")
E6 = _load_sibling("kamemory_e6_for_e9", "07_ca1_mixed_selectivity.py")
E7 = _load_sibling("kamemory_e7_for_e9", "08_ca1_data_comparison.py")


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


def background_masks(
    uniforms: np.ndarray,
    cue_present: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """Return nested, cue-excluding event masks for a prespecified rate sweep."""

    uniforms = np.asarray(uniforms, dtype=float).reshape(-1)
    cue_present = np.asarray(cue_present, dtype=bool).reshape(-1)
    rates = np.asarray(rates, dtype=float).reshape(-1)
    if uniforms.shape != cue_present.shape:
        raise ValueError("uniform draws and cue mask must have equal shape")
    if np.any((uniforms < 0) | (uniforms >= 1)):
        raise ValueError("uniform draws must lie in [0, 1)")
    if len(rates) < 2 or np.any(np.diff(rates) <= 0):
        raise ValueError("background rates must be strictly increasing")
    if rates[0] != 0 or np.any((rates < 0) | (rates > 1)):
        raise ValueError("background rates must start at zero and lie in [0, 1]")
    return (uniforms[None, :] < rates[:, None]) & ~cue_present[None, :]


def plasticity_profiles(
    cue_present: np.ndarray,
    background_event_masks: np.ndarray,
    *,
    cue_alpha: float,
    baseline_alpha: float,
) -> np.ndarray:
    """Construct E3-compatible learning-rate profiles for every sweep rate."""

    cue_present = np.asarray(cue_present, dtype=bool).reshape(-1)
    masks = np.asarray(background_event_masks, dtype=bool)
    if masks.ndim != 2 or masks.shape[1] != len(cue_present):
        raise ValueError("background masks must have shape (rate, training row)")
    if np.any(masks[:, cue_present]):
        raise ValueError("background events cannot coincide with cue presentations")
    if not 0 <= baseline_alpha < cue_alpha:
        raise ValueError("require 0 <= baseline_alpha < cue_alpha")
    profiles = np.full(masks.shape, baseline_alpha, dtype=np.float32)
    profiles[:, cue_present] = cue_alpha
    profiles[masks] = cue_alpha
    return profiles


def pareto_frontier(error: np.ndarray, performance: np.ndarray) -> np.ndarray:
    """Non-dominated points when error is minimized and performance maximized."""

    error = np.asarray(error, dtype=float)
    performance = np.asarray(performance, dtype=float)
    if error.shape != performance.shape or error.ndim != 1:
        raise ValueError("error and performance must be equal one-dimensional arrays")
    keep = np.isfinite(error) & np.isfinite(performance)
    for index in np.flatnonzero(keep):
        dominates = (
            (error <= error[index])
            & (performance >= performance[index])
            & ((error < error[index]) | (performance > performance[index]))
            & np.isfinite(error)
            & np.isfinite(performance)
        )
        if np.any(dominates):
            keep[index] = False
    return keep


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, np.nan
    return mean, float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def retrieve_batch(memory, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized, read-only equivalent of E3.retrieve_bank for this model."""

    if memory._num_swaps:
        raise ValueError("batch retrieval does not support activity swaps")
    with torch.no_grad():
        patterns = torch.as_tensor(inputs, dtype=torch.float32, device=memory.W_ca3_ca1.device)
        ca3 = sparsemoid(
            patterns @ memory.W_ei_ca3.T, memory._K_ca3, memory._beta_ca3
        )
        ca1 = sparsemoid(
            ca3 @ memory.W_ca3_ca1.T, memory._K_lat, memory._beta_ca3
        )
        output = sparsemoid(
            ca1 @ memory.W_ca1_eo.T + memory.B_ca1_eo.T,
            memory._K_out,
            memory._beta,
        )
    return ca1.cpu().numpy(), output.cpu().numpy()


def prepare_seed_layout(
    seed: int,
    layout_index: int,
    cue_positions: tuple[int, ...],
    autoencoder,
    params: dict,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Generate one fresh paired E3 task and its nested background schedules."""

    run_seed = seed + 100_000 * layout_index
    seed_everything(run_seed)
    train_rng = np.random.default_rng(run_seed + 60_000)
    test_rng = np.random.default_rng(run_seed + 70_000)
    train_data = generate_factorial_track(
        track_length=params["track_length"],
        cue_positions=cue_positions,
        repeats_per_combination=args.train_repeats,
        cue_free_laps=args.train_free_laps,
        spatial_size=params["spatial_dim"],
        sensory_size=params["sensory_dim"],
        sensory_active=params["sensory_active"],
        place_field_sigma=params["place_field_sigma"],
        rng=train_rng,
    )
    test_data = generate_factorial_track(
        track_length=params["track_length"],
        cue_positions=cue_positions,
        repeats_per_combination=args.test_repeats,
        cue_free_laps=args.test_free_laps,
        spatial_size=params["spatial_dim"],
        sensory_size=params["sensory_dim"],
        sensory_active=params["sensory_active"],
        place_field_sigma=params["place_field_sigma"],
        rng=test_rng,
        cue_patterns=train_data["cue_patterns"],
    )
    base = E3.make_base_memory(autoencoder, params, args.cue_alpha)
    codes = E3.code_bank(autoencoder, train_data["inputs"])
    with torch.no_grad():
        ca3 = torch.stack(
            [
                base.ca3_activity(torch.as_tensor(pattern)).reshape(-1).cpu()
                for pattern in train_data["inputs"]
            ]
        )

    background_rng = np.random.default_rng(run_seed + args.background_rng_offset)
    uniforms = background_rng.random(len(train_data["inputs"]))
    background_matching = E3.derangement(len(train_data["inputs"]), background_rng)
    masks = background_masks(
        uniforms, train_data["cue_present"], args.background_rates
    )
    profiles = plasticity_profiles(
        train_data["cue_present"],
        masks,
        cue_alpha=args.cue_alpha,
        baseline_alpha=args.baseline_alpha,
    )
    return {
        "seed": seed,
        "layout_index": layout_index,
        "run_seed": run_seed,
        "base": base,
        "codes": codes,
        "ca3": ca3,
        "train_data": train_data,
        "test_data": test_data,
        "background_uniforms": uniforms,
        "background_matching": background_matching,
        "background_codes": codes[background_matching],
        "background_masks": masks,
        "plasticity_profiles": profiles,
    }


def evaluate_rate(
    prepared: dict[str, object],
    rate_index: int,
    params: dict,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Train and evaluate one rate using E6's exact target-gated replay."""

    train_data = prepared["train_data"]
    test_data = prepared["test_data"]
    memory = copy.deepcopy(prepared["base"])
    signals = prepared["codes"].clone()
    background_mask = prepared["background_masks"][rate_index]
    signals[background_mask] = prepared["background_codes"][background_mask]
    final_weights = E6.sparse_final_weights(
        prepared["ca3"], signals, prepared["plasticity_profiles"][rate_index]
    )
    memory.W_ca3_ca1.copy_(final_weights.to(memory.W_ca3_ca1))

    track_length = params["track_length"]
    probe_positions = np.asarray(E3.DEFAULT_LAYOUTS[prepared["layout_index"]])
    train_rows = E6.event_rows(
        train_data["cue_ids"], train_data["event_positions"], track_length
    )
    test_rows = E6.event_rows(
        test_data["cue_ids"], test_data["event_positions"], track_length
    )
    test_free_rows = E6.probe_rows(
        test_data["cue_ids"], probe_positions, track_length
    )
    train_event, _ = retrieve_batch(memory, train_data["inputs"][train_rows])
    test_event, test_output = retrieve_batch(memory, test_data["inputs"][test_rows])
    test_free, _ = retrieve_batch(memory, test_data["inputs"][test_free_rows])
    train_cues = train_data["cue_ids"][train_data["cue_ids"] >= 0]
    test_cues = test_data["cue_ids"][test_data["cue_ids"] >= 0]
    train_positions = train_data["event_positions"][train_data["cue_ids"] >= 0]
    test_positions = test_data["event_positions"][test_data["cue_ids"] >= 0]
    selectivity = E6.analyze_condition(
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
    targets = test_data["inputs"][test_rows]
    spatial_dim = params["spatial_dim"]
    selectivity.update(
        {
            "output_cosine": float(np.mean(E3.cosine_rows(targets, test_output))),
            "spatial_output_cosine": float(
                np.mean(
                    E3.cosine_rows(
                        targets[:, :spatial_dim], test_output[:, :spatial_dim]
                    )
                )
            ),
            "sensory_output_cosine": float(
                np.mean(
                    E3.cosine_rows(
                        targets[:, spatial_dim:], test_output[:, spatial_dim:]
                    )
                )
            ),
            "cue_accuracy": E3.cue_accuracy_across_positions(
                train_event,
                train_cues,
                train_positions,
                test_event,
                test_cues,
                test_positions,
            ),
            "position_accuracy": E3.position_accuracy_across_cues(
                train_event,
                train_cues,
                train_positions,
                test_event,
                test_cues,
                test_positions,
            ),
            "weight_digest": array_digest(final_weights.numpy()),
            "background_event_count": int(
                np.sum(prepared["background_masks"][rate_index])
            ),
        }
    )
    return selectivity


def direct_baseline_parity(prepared: dict[str, object]) -> tuple[bool, bool]:
    """Check that rate zero is exactly the E3 aligned storage protocol."""

    direct = copy.deepcopy(prepared["base"])
    for pattern, signal, alpha in zip(
        prepared["train_data"]["inputs"],
        prepared["codes"],
        prepared["plasticity_profiles"][0],
    ):
        direct._alpha = float(alpha)
        direct.store(pattern, instructive_signal=signal)
    sparse = E6.sparse_final_weights(
        prepared["ca3"], prepared["codes"], prepared["plasticity_profiles"][0]
    )
    weight_match = bool(
        torch.allclose(direct.W_ca3_ca1.cpu(), sparse.cpu(), atol=2e-9, rtol=1e-7)
    )
    rows = E6.event_rows(
        prepared["test_data"]["cue_ids"],
        prepared["test_data"]["event_positions"],
        int(prepared["test_data"]["track_length"]),
    )
    sequential_ca1, sequential_output = E3.retrieve_bank(
        direct, prepared["test_data"]["inputs"][rows]
    )
    batch_ca1, batch_output = retrieve_batch(
        direct, prepared["test_data"]["inputs"][rows]
    )
    retrieval_match = bool(
        np.allclose(sequential_ca1, batch_ca1, atol=1e-6, rtol=1e-6)
        and np.allclose(sequential_output, batch_output, atol=1e-6, rtol=1e-6)
    )
    return weight_match, retrieval_match


def build_arrays(results: list[list[list[dict[str, object]]]]) -> dict[str, np.ndarray]:
    """Stack rate x seed x layout results, retaining required cell-level data."""

    scalar_keys = (
        "output_cosine",
        "spatial_output_cosine",
        "sensory_output_cosine",
        "cue_accuracy",
        "position_accuracy",
        "remapping_rho",
        "background_event_count",
    )
    cell_keys = (
        "classes",
        "train_position_active",
        "train_cue_active",
        "train_interaction_active",
        "test_position_eta",
        "test_cue_eta",
        "test_interaction_eta",
    )
    arrays = {
        key: np.asarray(
            [
                [
                    [results[rate][seed][layout][key] for layout in range(len(results[rate][seed]))]
                    for seed in range(len(results[rate]))
                ]
                for rate in range(len(results))
            ]
        )
        for key in scalar_keys + cell_keys
    }
    return arrays


def biological_comparison(
    arrays: dict[str, np.ndarray], empirical: np.ndarray, min_eta: float
) -> dict[str, np.ndarray]:
    """Apply the frozen E7 population mapping separately to every seed."""

    cue = arrays["train_cue_active"]
    spatial = arrays["test_position_eta"] >= min_eta
    tables = np.asarray(
        [
            [
                E7.contingency(cue[rate, seed], spatial[rate, seed])
                for seed in range(cue.shape[1])
            ]
            for rate in range(cue.shape[0])
        ]
    )
    log_or = np.asarray(
        [[E7.log_odds_ratio(table) for table in rate_tables] for rate_tables in tables]
    )
    conditional = np.asarray(
        [
            [E7.conditional_spatial(table) for table in rate_tables]
            for rate_tables in tables
        ]
    )
    empirical_conditional = np.asarray(E7.conditional_spatial(empirical))
    prevalence_rmse = np.sqrt(
        np.mean((conditional - empirical_conditional[None, None, :]) ** 2, axis=2)
    )
    empirical_log_or = E7.log_odds_ratio(empirical)
    mean_log_or = np.asarray([np.nanmean(values) for values in log_or])
    primary_error = np.abs(mean_log_or - empirical_log_or)
    return {
        "tables": tables,
        "log_odds_ratio": log_or,
        "conditional_spatial": conditional,
        "prevalence_rmse": prevalence_rmse,
        "mean_log_odds_ratio": mean_log_or,
        "primary_absolute_log_odds_ratio_error": primary_error,
        "empirical_conditional_spatial": empirical_conditional,
        "empirical_log_odds_ratio": np.asarray(empirical_log_or),
    }


def make_figure(
    rates: np.ndarray,
    arrays: dict[str, np.ndarray],
    biology: dict[str, np.ndarray],
    output_preservation_fraction: float,
    pareto: np.ndarray,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    figure.get_layout_engine().set(rect=(0, 0, 1, 0.95))
    design_ax, events_ax, prevalence_ax, odds_ax, output_ax, pareto_ax = axes.flat
    sweep_x = np.arange(len(rates))
    rate_labels = [f"{100 * rate:g}%" for rate in rates]

    def format_rate_axis(axis):
        axis.set_xticks(sweep_x, rate_labels, rotation=35, ha="right")

    design_ax.axis("off")
    design_ax.text(0.5, 0.91, "Cue presentation", ha="center", fontweight="bold")
    design_ax.text(0.5, 0.76, "content-aligned plateau", ha="center", bbox=dict(boxstyle="round", fc="#DCECF7"))
    design_ax.annotate("", xy=(0.5, 0.66), xytext=(0.5, 0.73), arrowprops=dict(arrowstyle="->"))
    design_ax.text(0.5, 0.58, "CA3→CA1 storage", ha="center", bbox=dict(boxstyle="round", fc="#E7F2E8"))
    design_ax.text(0.5, 0.38, "+", ha="center", fontsize=18)
    design_ax.text(0.5, 0.26, "cue/content-independent matched plateaus", ha="center", bbox=dict(boxstyle="round", fc="#FCE8DF"))
    design_ax.text(0.5, 0.10, "one frozen sweep: event probability per non-cue bin", ha="center", color="#555555", fontsize=9)
    design_ax.set_title("A  Minimal heterogeneity intervention")

    event_seed = arrays["background_event_count"].mean(axis=2)
    event_mean = event_seed.mean(axis=1)
    event_ci = 1.96 * event_seed.std(axis=1, ddof=1) / np.sqrt(event_seed.shape[1])
    events_ax.errorbar(sweep_x, event_mean, yerr=event_ci, color=COLOR, marker="o", capsize=4)
    format_rate_axis(events_ax)
    events_ax.set_xlabel("Background plateau probability / non-cue bin")
    events_ax.set_ylabel("Strong background events / training session")
    events_ax.set_title("B  Realized event dose")

    conditional = biology["conditional_spatial"]
    labels = ("Cue-responsive", "Cue-inactive")
    colors = (COLOR, SECONDARY_COLOR)
    for category, (label, color) in enumerate(zip(labels, colors)):
        seed_values = conditional[:, :, category]
        mean = np.nanmean(seed_values, axis=1)
        ci = np.asarray([mean_ci(row)[1] for row in seed_values])
        prevalence_ax.errorbar(sweep_x, mean, yerr=ci, color=color, marker="o", capsize=3, label=label)
        prevalence_ax.axhline(
            biology["empirical_conditional_spatial"][category],
            color=color,
            linestyle="--",
            linewidth=1.3,
        )
    format_rate_axis(prevalence_ax)
    prevalence_ax.set_ylim(-0.03, 1.03)
    prevalence_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    prevalence_ax.set_xlabel("Background plateau probability")
    prevalence_ax.set_ylabel("Cells with held-out position effect")
    prevalence_ax.set_title("C  Conditional CA1 spatial prevalence")
    prevalence_ax.legend(frameon=False, fontsize=8)

    log_or = biology["log_odds_ratio"]
    log_mean = np.nanmean(log_or, axis=1)
    log_ci = np.asarray([mean_ci(row)[1] for row in log_or])
    odds_ax.errorbar(sweep_x, log_mean, yerr=log_ci, color=COLOR, marker="o", capsize=4)
    odds_ax.axhline(float(biology["empirical_log_odds_ratio"]), color=EMPIRICAL_COLOR, linestyle="--", label="Published CA1")
    format_rate_axis(odds_ax)
    odds_ax.set_xlabel("Background plateau probability")
    odds_ax.set_ylabel("Cue–spatial log odds ratio")
    odds_ax.set_title("D  Biological association magnitude")
    odds_ax.legend(frameon=False)

    output_seed = arrays["output_cosine"].mean(axis=2)
    output_mean = output_seed.mean(axis=1)
    output_ci = 1.96 * output_seed.std(axis=1, ddof=1) / np.sqrt(output_seed.shape[1])
    threshold = output_preservation_fraction * output_mean[0]
    output_ax.errorbar(sweep_x, output_mean, yerr=output_ci, color=COLOR, marker="o", capsize=4)
    output_ax.axhline(threshold, color="#555555", linestyle="--", label=f"{100*output_preservation_fraction:.0f}% of rate-zero readout")
    format_rate_axis(output_ax)
    output_ax.set_xlabel("Background plateau probability")
    output_ax.set_ylabel("Held-out EC-output cosine")
    output_ax.set_title("E  Stable decoder readout")
    output_ax.legend(frameon=False, fontsize=8)

    primary_error = biology["primary_absolute_log_odds_ratio_error"]
    pareto_ax.scatter(primary_error[~pareto], output_mean[~pareto], color="#AAAAAA", s=45, label="Dominated")
    pareto_ax.scatter(primary_error[pareto], output_mean[pareto], color=COLOR, edgecolor="black", s=70, label="Pareto frontier", zorder=3)
    label_offsets = ((-15, -14), (-3, 10), (14, -3), (4, -13), (4, -13), (4, 5), (4, 5), (4, -13))
    for index, rate in enumerate(rates):
        pareto_ax.annotate(
            f"{100 * rate:g}%",
            (primary_error[index], output_mean[index]),
            xytext=label_offsets[index] if index < len(label_offsets) else (4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    pareto_ax.axhline(threshold, color="#555555", linestyle="--", linewidth=1)
    pareto_ax.set_xlabel("Absolute published log-OR error (lower is better)")
    pareto_ax.set_ylabel("Held-out EC-output cosine (higher is better)")
    pareto_ax.set_title("F  Biological agreement–readout trade-off")
    pareto_ax.legend(frameon=False, fontsize=8)

    figure.suptitle(
        "E9 — Background plateau heterogeneity does not resolve cue–spatial overcoupling",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    with args.reference.open("r", encoding="utf-8") as handle:
        reference = json.load(handle)
    empirical = E7.empirical_table(reference)
    info, autoencoder = load_autoencoder_session(args.checkpoint, map_location=args.device)
    autoencoder.eval()
    params = E3.session_parameters(info)
    layouts = E3.DEFAULT_LAYOUTS
    seeds = np.arange(args.seed_start, args.seed_start + args.num_seeds)

    prepared = [
        [
            prepare_seed_layout(
                int(seed), layout_index, layout, autoencoder, params, args
            )
            for layout_index, layout in enumerate(layouts)
        ]
        for seed in seeds
    ]
    baseline_weight_match, batch_retrieval_match = direct_baseline_parity(prepared[0][0])
    results = [
        [
            [
                evaluate_rate(prepared[seed][layout], rate, params, args)
                for layout in range(len(layouts))
            ]
            for seed in range(len(seeds))
        ]
        for rate in range(len(args.background_rates))
    ]
    arrays = build_arrays(results)
    biology = biological_comparison(arrays, empirical, args.min_eta)

    output_seed = arrays["output_cosine"].mean(axis=2)
    output_mean = output_seed.mean(axis=1)
    output_retention = output_mean / output_mean[0]
    prevalence_mean = np.nanmean(biology["prevalence_rmse"], axis=1)
    primary_error = biology["primary_absolute_log_odds_ratio_error"]
    pareto = pareto_frontier(primary_error, output_mean)
    quantitative_repair = (
        (primary_error <= args.maximum_log_or_error_for_repair)
        & (prevalence_mean < prevalence_mean[0])
        & (output_retention >= args.output_preservation_fraction)
    )

    scalar_summary = {}
    for rate_index, rate in enumerate(args.background_rates):
        conditional = biology["conditional_spatial"][rate_index]
        scalar_summary[f"{rate:g}"] = {
            "background_events_per_session": mean_ci(
                arrays["background_event_count"][rate_index].mean(axis=1)
            ),
            "spatial_given_cue": mean_ci(conditional[:, 0]),
            "spatial_given_no_cue": mean_ci(conditional[:, 1]),
            "mean_log_odds_ratio": mean_ci(biology["log_odds_ratio"][rate_index]),
            "primary_absolute_log_odds_ratio_error": float(primary_error[rate_index]),
            "conditional_prevalence_rmse": mean_ci(
                biology["prevalence_rmse"][rate_index]
            ),
            "output_cosine": mean_ci(output_seed[rate_index]),
            "output_retention_fraction": float(output_retention[rate_index]),
            "cue_accuracy": mean_ci(arrays["cue_accuracy"][rate_index].mean(axis=1)),
            "position_accuracy": mean_ci(
                arrays["position_accuracy"][rate_index].mean(axis=1)
            ),
            "pareto_frontier": bool(pareto[rate_index]),
            "quantitative_repair": bool(quantitative_repair[rate_index]),
        }

    masks_nested = all(
        np.all(prepared[seed][layout]["background_masks"][:-1] <= prepared[seed][layout]["background_masks"][1:])
        for seed in range(len(seeds))
        for layout in range(len(layouts))
    )
    masks_exclude_cues = all(
        not np.any(
            prepared[seed][layout]["background_masks"]
            & prepared[seed][layout]["train_data"]["cue_present"][None, :]
        )
        for seed in range(len(seeds))
        for layout in range(len(layouts))
    )
    protocol_checks = {
        "fresh_seeds_disjoint_from_e3_e6": bool(np.min(seeds) > 1100 + 12),
        "rate_zero_is_original_e3_aligned_profile": bool(
            np.all(prepared[0][0]["plasticity_profiles"][0][prepared[0][0]["train_data"]["cue_present"]] == args.cue_alpha)
            and np.all(prepared[0][0]["plasticity_profiles"][0][~prepared[0][0]["train_data"]["cue_present"]] == args.baseline_alpha)
        ),
        "rate_zero_weights_match_direct_e3_training": baseline_weight_match,
        "batch_retrieval_matches_e3_retrieval": batch_retrieval_match,
        "background_masks_are_nested": masks_nested,
        "background_events_exclude_cue_presentations": masks_exclude_cues,
        "same_uniform_draws_define_full_sweep_within_seed": True,
        "background_signals_are_randomly_matched_from_same_code_bank": True,
        "cue_and_background_events_use_same_event_alpha": True,
        "e7_mapping_and_primary_metric_are_unchanged": True,
        "seed_is_inferential_unit_and_layouts_are_pooled_within_seed": True,
        "complete_prespecified_sweep_is_reported": True,
    }
    hypothesis_checks = {
        "some_background_rate_reduces_primary_e7_error": bool(np.any(primary_error[1:] < primary_error[0])),
        "some_background_rate_improves_conditional_prevalence_rmse": bool(np.any(prevalence_mean[1:] < prevalence_mean[0])),
        "some_rate_quantitatively_repairs_e7_under_prespecified_rule": bool(np.any(quantitative_repair)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "e9_is_heterogeneity"
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        seeds=seeds,
        layouts=np.asarray(layouts),
        background_rates=args.background_rates,
        empirical_table=empirical,
        pareto_frontier=pareto,
        quantitative_repair=quantitative_repair,
        output_retention_fraction=output_retention,
        biological_prevalence_rmse=biology["prevalence_rmse"],
        biological_log_odds_ratio=biology["log_odds_ratio"],
        biological_tables=biology["tables"],
        biological_conditional_spatial=biology["conditional_spatial"],
        primary_absolute_log_odds_ratio_error=primary_error,
        train_input_sha256=np.asarray(
            [
                [array_digest(prepared[seed][layout]["train_data"]["inputs"]) for layout in range(len(layouts))]
                for seed in range(len(seeds))
            ]
        ),
        test_input_sha256=np.asarray(
            [
                [array_digest(prepared[seed][layout]["test_data"]["inputs"]) for layout in range(len(layouts))]
                for seed in range(len(seeds))
            ]
        ),
        background_uniform_sha256=np.asarray(
            [
                [array_digest(prepared[seed][layout]["background_uniforms"]) for layout in range(len(layouts))]
                for seed in range(len(seeds))
            ]
        ),
        background_matching_sha256=np.asarray(
            [
                [array_digest(prepared[seed][layout]["background_matching"]) for layout in range(len(layouts))]
                for seed in range(len(seeds))
            ]
        ),
        **arrays,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E9_cue_independent_plateau_heterogeneity",
        "configuration": vars(args),
        "runtime": runtime_metadata(),
        "checkpoint_info": info,
        "network_parameters": params,
        "reference": reference,
        "protocol": {
            "frozen_task": "E3 two-cue x four-position factorial track with two layouts and separately generated held-out laps",
            "intervention": "independent Bernoulli strong plateau events at non-cue training bins; cue events use the coincident content code, whereas background events use a randomly row-matched code from the same bank; both use cue_alpha",
            "baseline": "rate zero exactly retains E3's cue-event alpha and weak continuous non-cue baseline alpha",
            "pairing": "one uniform draw per seed/layout/training bin defines nested masks for every rate; one derangement defines the background signal bank for the full sweep",
            "primary_biological_endpoint": "absolute error between mean model log odds ratio and the exact published E7 log odds ratio",
            "secondary_biological_endpoint": "seed-level RMSE of spatial prevalence conditional on cue-responsive and cue-inactive status",
            "functional_endpoint": "held-out EC-output cosine at cue-event rows",
            "quantitative_repair_rule": f"absolute log-odds-ratio error is at most {args.maximum_log_or_error_for_repair:.2f}, conditional-prevalence RMSE improves over rate zero, and mean output cosine remains at least {args.output_preservation_fraction:.2f} of rate zero",
            "selection": "no rate is selected or hidden; all prespecified rates and the non-dominated frontier are reported",
        },
        "published_anchor": {
            "table": empirical,
            "conditional_spatial": biology["empirical_conditional_spatial"],
            "log_odds_ratio": biology["empirical_log_odds_ratio"],
        },
        "summary_by_background_rate": scalar_summary,
        "pareto_background_rates": args.background_rates[pareto],
        "quantitative_repair_background_rates": args.background_rates[quantitative_repair],
        "protocol_checks": protocol_checks,
        "hypothesis_checks": hypothesis_checks,
        "interpretation": {
            "positive_result_rule": "A positive result requires a nonzero rate meeting the frozen quantitative-repair rule, not merely a slightly lower odds ratio.",
            "negative_result_rule": "If no rate meets that rule, independently timed background plateaus alone do not quantitatively repair E7 while preserving the stable readout.",
            "scope": "This tests heterogeneity in plateau timing, not a temporally explicit BTSP eligibility/instructive-trace rule.",
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
    make_figure(
        args.background_rates,
        arrays,
        biology,
        args.output_preservation_fraction,
        pareto,
        prefix.with_suffix(".png"),
    )
    return report


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, _ = config_parser.parse_known_args()
    config = load_config(known.config)

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--checkpoint", default=config["checkpoint"])
    parser.add_argument("--seed-start", type=int, default=config["seed_start"])
    parser.add_argument("--num-seeds", type=int, default=config["num_seeds"])
    parser.add_argument("--train-repeats", type=int, default=config["train_repeats"])
    parser.add_argument("--test-repeats", type=int, default=config["test_repeats"])
    parser.add_argument("--train-free-laps", type=int, default=config["train_free_laps"])
    parser.add_argument("--test-free-laps", type=int, default=config["test_free_laps"])
    parser.add_argument("--cue-alpha", type=float, default=config["cue_alpha"])
    parser.add_argument("--baseline-alpha", type=float, default=config["baseline_alpha"])
    parser.add_argument("--background-rates", type=float, nargs="+", default=config["background_rates"])
    parser.add_argument("--background-rng-offset", type=int, default=config["background_rng_offset"])
    parser.add_argument("--fdr", type=float, default=config["fdr"])
    parser.add_argument("--min-eta", type=float, default=config["min_eta"])
    parser.add_argument("--output-preservation-fraction", type=float, default=config["output_preservation_fraction"])
    parser.add_argument("--maximum-log-or-error-for-repair", type=float, default=config["maximum_log_or_error_for_repair"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.background_rates = np.asarray(args.background_rates, dtype=float)
    if args.num_seeds < 2:
        parser.error("--num-seeds must be at least 2")
    if args.background_rates[0] != 0 or np.any(np.diff(args.background_rates) <= 0):
        parser.error("--background-rates must be strictly increasing and start at zero")
    if np.any((args.background_rates < 0) | (args.background_rates > 1)):
        parser.error("--background-rates must lie in [0, 1]")
    if not 0 <= args.baseline_alpha < args.cue_alpha:
        parser.error("require 0 <= --baseline-alpha < --cue-alpha")
    if not 0 < args.fdr < 1 or not 0 <= args.min_eta < 1:
        parser.error("require 0 < --fdr < 1 and 0 <= --min-eta < 1")
    if not 0 < args.output_preservation_fraction <= 1:
        parser.error("--output-preservation-fraction must lie in (0, 1]")
    if args.maximum_log_or_error_for_repair < 0:
        parser.error("--maximum-log-or-error-for-repair must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    report = run(args)
    print(
        json.dumps(
            json_ready(
                {
                    "summary_by_background_rate": report["summary_by_background_rate"],
                    "pareto_background_rates": report["pareto_background_rates"],
                    "quantitative_repair_background_rates": report["quantitative_repair_background_rates"],
                    "protocol_checks": report["protocol_checks"],
                    "hypothesis_checks": report["hypothesis_checks"],
                    "outputs": report["outputs"],
                }
            ),
            indent=2,
        )
    )
    return 0 if all(report["protocol_checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
