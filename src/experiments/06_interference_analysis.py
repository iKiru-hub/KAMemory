"""E5: exact synaptic-trace account of recency and forgetting.

The aligned E2 update can be unrolled exactly. For memory m stored with CA1
target s_m and CA3 code c_m, its contribution at later load T is

    [alpha s_m prod_{u=m+1..T}(1 - alpha s_u)] c_m^T.

This experiment reconstructs that contribution from the frozen E2 source data,
decomposes recall into surviving self-trace and later crosstalk, and tests—using
the network seed as the independent unit—whether these quantities explain
forgetting beyond memory age alone.

Run from the repository root:

    python3 src/experiments/06_interference_analysis.py --deterministic
"""

from __future__ import annotations

import argparse
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
import numpy as np
import torch

from kamemory.io import load_autoencoder_session, runtime_metadata
from kamemory.utils import array_digest, sparsemoid


SCHEMA_VERSION = 1
DEFAULT_E2_DATA = Path(__file__).resolve().parent / "plots" / "e2a_retention.npz"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
DEFAULT_ALPHAS = (0.12, 0.2087942893137534, 0.35, 0.55)
AGE_BIN_LABELS = ("Immediate", "Recent", "Middle", "Remote")
AGE_BIN_BOUNDS = ((0, 1), (1, 5), (5, 15), (15, 30))


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


def checkpoint_parameters(info: dict) -> dict[str, int | float]:
    if "network_params" in info:
        params = info["network_params"]
        return {
            "input_dim": int(params["dim_ei"]),
            "input_active": int(params["K_ei"]),
            "output_active": int(params["K_eo"]),
            "dim_ca1": int(params["dim_ca1"]),
            "dim_ca3": int(params["dim_ca3"]),
            "ca1_active": int(params["K_ca1"]),
            "ca3_active": int(params["K_ca3"]),
            "beta": float(params["beta_ca1"]),
        }
    return {
        "input_dim": int(info["dim_ei"]),
        "input_active": int(info["K"]),
        "output_active": int(info["K"]),
        "dim_ca1": int(info["dim_ca1"]),
        "dim_ca3": int(info["dim_ca3"]),
        "ca1_active": int(info["K_lat"]),
        "ca3_active": min(22, int(info["dim_ca3"])),
        "beta": float(info["beta"]),
    }


def cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum(left * right, dim=1)
    denominator = torch.linalg.vector_norm(left, dim=1) * torch.linalg.vector_norm(
        right, dim=1
    )
    return numerator / torch.clamp(denominator, min=1e-12)


def age_curve(matrix: np.ndarray) -> np.ndarray:
    """Mean along each lower-triangular memory-age diagonal."""

    matrix = np.asarray(matrix, dtype=float)
    num_loads = matrix.shape[0]
    return np.asarray(
        [
            np.nanmean(matrix[np.arange(age, num_loads), np.arange(num_loads - age)])
            for age in range(num_loads)
        ]
    )


def first_half_age(curve: np.ndarray) -> float:
    curve = np.asarray(curve, dtype=float)
    below = np.flatnonzero(curve < 0.5 * curve[0])
    return float(below[0]) if len(below) else np.nan


def r_squared(y: np.ndarray, predictors: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    predictors = np.asarray(predictors, dtype=float)
    finite = np.isfinite(y) & np.all(np.isfinite(predictors), axis=1)
    y = y[finite]
    predictors = predictors[finite]
    if len(y) <= predictors.shape[1] + 1 or np.var(y) <= 1e-12:
        return np.nan
    mean = predictors.mean(axis=0)
    scale = predictors.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack((np.ones(len(y)), (predictors - mean) / scale))
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    residual = np.sum((y - fitted) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    return float(1 - residual / total)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3:
        return np.nan
    x, y = x[finite], y[finite]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def mean_ci(values: np.ndarray, axis=0) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=axis)
    count = np.sum(np.isfinite(values), axis=axis)
    sd = np.nanstd(values, axis=axis, ddof=1)
    return mean, 1.96 * sd / np.sqrt(np.maximum(count, 1))


def contribution_step(
    coefficients: torch.Tensor,
    weights: torch.Tensor,
    signal: torch.Tensor,
    presynaptic: torch.Tensor,
    load: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one update to both W and its exact per-memory decomposition."""

    if load:
        coefficients[:load] *= 1 - alpha * signal
    coefficients[load] = alpha * signal
    weights = (1 - alpha * signal[:, None]) * weights + alpha * torch.outer(
        signal, presynaptic
    )
    return coefficients, weights


def reconstruct_seed(
    patterns: np.ndarray,
    projection: np.ndarray,
    autoencoder,
    params: dict,
    alpha: float,
) -> dict[str, np.ndarray]:
    """Reconstruct aligned storage and its exact contribution decomposition."""

    patterns_t = torch.as_tensor(patterns, dtype=torch.float32)
    projection_t = torch.as_tensor(projection, dtype=torch.float32)
    with torch.no_grad():
        signals = autoencoder.encode(patterns_t).detach().cpu()
        # E2 processes one column at a time. With beta_ca3=5400, tiny BLAS
        # differences from a batched matmul can cross the sparsemoid threshold.
        ca3 = torch.stack(
            [
                sparsemoid(
                    (projection_t @ pattern.reshape(-1, 1)).T,
                    params["ca3_active"],
                    100.0 * params["beta"],
                ).reshape(-1)
                for pattern in patterns_t
            ]
        ).cpu()

    num_patterns = len(patterns)
    corrected = np.full((num_patterns, num_patterns), np.nan, dtype=np.float32)
    ca1_similarity = np.full_like(corrected, np.nan)
    trace_survival = np.full_like(corrected, np.nan)
    crosstalk_ratio = np.full_like(corrected, np.nan)
    coefficients = torch.zeros((num_patterns, params["dim_ca1"]), dtype=torch.float32)
    weights = torch.zeros((params["dim_ca1"], params["dim_ca3"]), dtype=torch.float32)
    decoder = autoencoder.decoder[0]
    chance = params["input_active"] / params["input_dim"]

    with torch.no_grad():
        for load in range(num_patterns):
            signal = signals[load]
            presynaptic = ca3[load]
            coefficients, weights = contribution_step(
                coefficients, weights, signal, presynaptic, load, alpha
            )

            queries = ca3[: load + 1]
            ca1_rows = []
            output_rows = []
            # Preserve E2's column-wise retrieval path for numerical parity.
            for query in queries:
                ca1_row = sparsemoid(
                    (weights @ query.reshape(-1, 1)).T,
                    params["ca1_active"],
                    100.0 * params["beta"],
                ).reshape(-1)
                output_row = sparsemoid(
                    decoder(ca1_row).reshape(1, -1),
                    params["output_active"],
                    params["beta"],
                ).reshape(-1)
                ca1_rows.append(ca1_row)
                output_rows.append(output_row)
            ca1 = torch.stack(ca1_rows)
            output = torch.stack(output_rows)
            raw = cosine_rows(patterns_t[: load + 1], output)
            corrected[load, : load + 1] = ((raw - chance) / (1 - chance)).numpy()
            ca1_similarity[load, : load + 1] = cosine_rows(
                signals[: load + 1], ca1
            ).numpy()

            initial_norm = torch.linalg.vector_norm(alpha * signals[: load + 1], dim=1)
            current_norm = torch.linalg.vector_norm(coefficients[: load + 1], dim=1)
            trace_survival[load, : load + 1] = (
                current_norm / torch.clamp(initial_norm, min=1e-12)
            ).numpy()

            gram = queries @ queries.T
            decomposed = gram @ coefficients[: load + 1]
            self_component = torch.diag(gram)[:, None] * coefficients[: load + 1]
            cross_component = decomposed - self_component
            crosstalk_ratio[load, : load + 1] = (
                torch.linalg.vector_norm(cross_component, dim=1)
                / torch.clamp(
                    torch.linalg.vector_norm(self_component, dim=1), min=1e-12
                )
            ).numpy()

    return {
        "signals": signals.numpy(),
        "ca3": ca3.numpy(),
        "chance_corrected": corrected,
        "ca1_similarity": ca1_similarity,
        "trace_survival": trace_survival,
        "crosstalk_ratio": crosstalk_ratio,
        "final_weights": weights.numpy(),
    }


def seed_statistics(
    recall: np.ndarray,
    trace_survival: np.ndarray,
    crosstalk_ratio: np.ndarray,
) -> dict[str, float]:
    load, memory = np.tril_indices(len(recall), k=-1)
    age = (load - memory).astype(float)
    outcome = recall[load, memory]
    log_survival = np.log(np.maximum(trace_survival[load, memory], 1e-12))
    log_crosstalk = np.log1p(crosstalk_ratio[load, memory])
    age_predictors = np.column_stack((age, age**2))
    full_predictors = np.column_stack(
        (age, age**2, log_survival, log_crosstalk)
    )
    age_r2 = r_squared(outcome, age_predictors)
    full_r2 = r_squared(outcome, full_predictors)
    return {
        "trace_recall_correlation": pearson(log_survival, outcome),
        "crosstalk_recall_correlation": pearson(log_crosstalk, outcome),
        "age_only_r2": age_r2,
        "full_r2": full_r2,
        "incremental_r2": full_r2 - age_r2,
    }


def age_bin_seed_means(matrix: np.ndarray) -> np.ndarray:
    rows = []
    for seed_matrix in matrix:
        curve = age_curve(seed_matrix)
        rows.append(
            [np.nanmean(curve[start:stop]) for start, stop in AGE_BIN_BOUNDS]
        )
    return np.asarray(rows)


def make_figure(arrays: dict, threshold: float, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), constrained_layout=True)
    heatmap_ax, curve_ax, survival_ax, scatter_ax, model_ax, alpha_ax = axes.flat

    measured = arrays["measured_recall"]
    trace = arrays["trace_survival"]
    finite_count = np.sum(np.isfinite(measured), axis=0)
    mean_matrix = np.divide(
        np.nansum(measured, axis=0),
        finite_count,
        out=np.full(finite_count.shape, np.nan, dtype=float),
        where=finite_count > 0,
    )
    image = heatmap_ax.imshow(mean_matrix, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
    heatmap_ax.set(
        title="A  Measured retention triangle",
        xlabel="Memory index",
        ylabel="Total memory load",
    )
    figure.colorbar(image, ax=heatmap_ax, label="Chance-corrected recall", fraction=0.046)

    recall_curves = np.asarray([age_curve(matrix) for matrix in measured])
    trace_curves = np.asarray([age_curve(matrix) for matrix in trace])
    ages = np.arange(recall_curves.shape[1])
    for curves, color, label in (
        (recall_curves, "#2878B5", "Measured recall"),
        (trace_curves, "#E6862A", "Exact surviving trace"),
    ):
        mean, ci = mean_ci(curves)
        curve_ax.plot(ages, mean, color=color, lw=2, label=label)
        curve_ax.fill_between(ages, mean - ci, mean + ci, color=color, alpha=0.16)
    curve_ax.set(
        title="B  Recency follows synaptic trace survival",
        xlabel="Memory age (subsequent stores)",
        ylabel="Normalized value",
        ylim=(-0.03, 1.03),
    )
    curve_ax.legend(frameon=False)

    survival_probability = np.asarray(
        [
            [
                np.mean(
                    matrix[np.arange(age, len(matrix)), np.arange(len(matrix) - age)]
                    >= threshold
                )
                for age in ages
            ]
            for matrix in measured
        ]
    )
    for row in survival_probability:
        survival_ax.plot(ages, row, color="#2878B5", alpha=0.12, lw=0.8)
    survival_mean, survival_ci = mean_ci(survival_probability)
    survival_ax.plot(ages, survival_mean, color="#2878B5", lw=2.5)
    survival_ax.fill_between(
        ages, survival_mean - survival_ci, survival_mean + survival_ci,
        color="#2878B5", alpha=0.18,
    )
    survival_ax.set(
        title="C  Distribution of memory survival across seeds",
        xlabel="Memory age (subsequent stores)",
        ylabel=f"Fraction above threshold ({threshold:.2f})",
        ylim=(-0.03, 1.03),
    )

    load, memory = np.tril_indices(measured.shape[-1], k=-1)
    x = np.concatenate([matrix[load, memory] for matrix in trace])
    y = np.concatenate([matrix[load, memory] for matrix in measured])
    scatter = scatter_ax.hexbin(x, y, gridsize=42, mincnt=1, cmap="Blues", bins="log")
    scatter_ax.set(
        title="D  Memory-level mechanism",
        xlabel="Exact surviving trace",
        ylabel="Chance-corrected recall",
        xlim=(-0.02, 1.02),
    )
    figure.colorbar(scatter, ax=scatter_ax, label="log count", fraction=0.046)

    age_r2 = arrays["age_only_r2"]
    full_r2 = arrays["full_r2"]
    for seed in range(len(age_r2)):
        model_ax.plot((0, 1), (age_r2[seed], full_r2[seed]), color="0.75", lw=1)
    model_ax.scatter(np.zeros(len(age_r2)), age_r2, color="#7A7A7A", s=28, zorder=3)
    model_ax.scatter(np.ones(len(full_r2)), full_r2, color="#2878B5", s=28, zorder=3)
    model_ax.set_xticks((0, 1), ("Age only", "Age + trace\n+ crosstalk"))
    model_ax.set(title="E  Mechanism adds modest variance beyond age", ylabel=r"Within-seed $R^2$", xlim=(-0.4, 1.4))

    alpha_values = arrays["alpha_values"]
    recall_half = arrays["alpha_recall_half_life"]
    trace_half = arrays["alpha_trace_half_life"]
    for values, color, marker, label in (
        (recall_half, "#2878B5", "o", "Measured recall half-life"),
        (trace_half, "#E6862A", "s", "Trace half-life"),
    ):
        mean, ci = mean_ci(values, axis=1)
        alpha_ax.errorbar(
            alpha_values, mean, yerr=ci, color=color, marker=marker,
            capsize=3, lw=2, label=label,
        )
    alpha_ax.axvline(0.35, color="black", ls=":", lw=1, label="E2 alpha")
    alpha_ax.set(
        title="F  Learning rate controls forgetting time",
        xlabel=r"Learning rate $\alpha$",
        ylabel="Half-life (subsequent stores)",
    )
    alpha_ax.legend(frameon=False, fontsize=8)

    figure.suptitle(
        "E5 — Row-specific synaptic overwriting explains gradual forgetting",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    if not args.e2_data.is_file():
        raise FileNotFoundError(
            f"missing frozen E2 source data: {args.e2_data}; run 02_retention.py first"
        )
    source = np.load(args.e2_data)
    conditions = source["conditions"].astype(str)
    aligned_index = int(np.flatnonzero(conditions == "aligned")[0])
    patterns = source["patterns"]
    projections = source["base_ca3_projection"]
    measured_recall = source["chance_corrected"][aligned_index]
    measured_ca1 = source["ca1_similarity"][aligned_index]
    saved_weights = source["final_weights"][aligned_index]
    seeds = source["seeds"]
    threshold = float(source["capacity_threshold"])

    info, autoencoder = load_autoencoder_session(args.checkpoint, map_location="cpu")
    autoencoder.eval()
    params = checkpoint_parameters(info)
    if patterns.shape[2] != params["input_dim"] or projections.shape[1:] != (
        params["dim_ca3"], params["input_dim"]
    ):
        raise ValueError("E2 source arrays do not match the selected checkpoint")

    e2_reconstructions = []
    for seed_index, seed in enumerate(seeds):
        print(f"E5 exact reconstruction seed {int(seed)} ({seed_index + 1}/{len(seeds)})", flush=True)
        e2_reconstructions.append(
            reconstruct_seed(
                patterns[seed_index], projections[seed_index], autoencoder,
                params, args.e2_alpha,
            )
        )
    reconstructed_recall = np.stack(
        [item["chance_corrected"] for item in e2_reconstructions]
    )
    reconstructed_ca1 = np.stack(
        [item["ca1_similarity"] for item in e2_reconstructions]
    )
    trace_survival = np.stack(
        [item["trace_survival"] for item in e2_reconstructions]
    )
    crosstalk_ratio = np.stack(
        [item["crosstalk_ratio"] for item in e2_reconstructions]
    )
    reconstructed_weights = np.stack(
        [item["final_weights"] for item in e2_reconstructions]
    )
    signals = np.stack([item["signals"] for item in e2_reconstructions])
    ca3 = np.stack([item["ca3"] for item in e2_reconstructions])

    parity = {
        "recall_max_abs_error": float(np.nanmax(np.abs(reconstructed_recall - measured_recall))),
        "ca1_similarity_max_abs_error": float(np.nanmax(np.abs(reconstructed_ca1 - measured_ca1))),
        "final_weight_max_abs_error": float(np.max(np.abs(reconstructed_weights - saved_weights))),
    }
    parity["passed"] = bool(
        parity["recall_max_abs_error"] < 1e-6
        and parity["ca1_similarity_max_abs_error"] < 1e-6
        and parity["final_weight_max_abs_error"] < 1e-6
    )
    if not parity["passed"]:
        raise AssertionError(f"exact E2 reconstruction failed: {parity}")

    statistics = [
        seed_statistics(measured_recall[index], trace_survival[index], crosstalk_ratio[index])
        for index in range(len(seeds))
    ]
    statistic_arrays = {
        key: np.asarray([item[key] for item in statistics]) for key in statistics[0]
    }

    alpha_values = np.asarray(args.alphas, dtype=float)
    alpha_recall_curves = np.empty(
        (len(alpha_values), len(seeds), patterns.shape[1]), dtype=np.float32
    )
    alpha_trace_curves = np.empty_like(alpha_recall_curves)
    for alpha_index, alpha in enumerate(alpha_values):
        print(f"E5 learning-rate replay alpha={alpha:.6g}", flush=True)
        for seed_index in range(len(seeds)):
            if np.isclose(alpha, args.e2_alpha):
                recall_matrix = measured_recall[seed_index]
                trace_matrix = trace_survival[seed_index]
            else:
                replay = reconstruct_seed(
                    patterns[seed_index], projections[seed_index], autoencoder,
                    params, float(alpha),
                )
                recall_matrix = replay["chance_corrected"]
                trace_matrix = replay["trace_survival"]
            alpha_recall_curves[alpha_index, seed_index] = age_curve(recall_matrix)
            alpha_trace_curves[alpha_index, seed_index] = age_curve(trace_matrix)
    alpha_recall_half_life = np.asarray(
        [[first_half_age(curve) for curve in curves] for curves in alpha_recall_curves]
    )
    alpha_trace_half_life = np.asarray(
        [[first_half_age(curve) for curve in curves] for curves in alpha_trace_curves]
    )

    survival_probability = np.asarray(
        [
            [
                np.mean(
                    matrix[np.arange(age, len(matrix)), np.arange(len(matrix) - age)]
                    >= threshold
                )
                for age in range(len(matrix))
            ]
            for matrix in measured_recall
        ]
    )
    arrays = {
        "seeds": seeds,
        "measured_recall": measured_recall,
        "reconstructed_recall": reconstructed_recall,
        "measured_ca1_similarity": measured_ca1,
        "reconstructed_ca1_similarity": reconstructed_ca1,
        "trace_survival": trace_survival,
        "crosstalk_ratio": crosstalk_ratio,
        "signals": signals,
        "ca3": ca3,
        "survival_probability": survival_probability,
        "age_bin_seed_means": age_bin_seed_means(measured_recall),
        "alpha_values": alpha_values,
        "alpha_recall_curves": alpha_recall_curves,
        "alpha_trace_curves": alpha_trace_curves,
        "alpha_recall_half_life": alpha_recall_half_life,
        "alpha_trace_half_life": alpha_trace_half_life,
        **statistic_arrays,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "e5_interference_analysis"
    np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
    make_figure(arrays, threshold, prefix.with_suffix(".png"))

    incremental = statistic_arrays["incremental_r2"]
    trace_correlation = statistic_arrays["trace_recall_correlation"]
    crosstalk_correlation = statistic_arrays["crosstalk_recall_correlation"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E5_synaptic_trace_survival_and_forgetting",
        "configuration": vars(args),
        "checkpoint_info": info,
        "network_parameters": params,
        "derivation": {
            "update": "W_t = (1 - alpha s_t) elementwise W_(t-1) + alpha s_t c_t^T",
            "memory_contribution_at_T": "[alpha s_m product_(u=m+1..T)(1-alpha s_u)] c_m^T",
            "trace_survival": "Frobenius norm of surviving memory contribution divided by its norm immediately after storage",
            "crosstalk_ratio": "norm of all non-self contribution to the queried CA3 code divided by norm of its surviving self contribution",
        },
        "protocol": {
            "source_e2_data": str(args.e2_data),
            "source_e2_sha256": array_digest(measured_recall),
            "seeds": seeds,
            "independent_unit": "network/data seed",
            "memory_level_points": "descriptive and nested within seed",
            "regression": "within-seed OLS; age+age^2 compared with age+age^2+log(trace survival)+log1p(crosstalk ratio)",
            "capacity_threshold": threshold,
            "learning_rate_values": alpha_values,
        },
        "exact_reconstruction": parity,
        "seed_level_statistics": statistic_arrays,
        "summary": {
            "trace_recall_correlation_mean": float(np.nanmean(trace_correlation)),
            "trace_recall_correlation_ci95": float(
                1.96 * np.nanstd(trace_correlation, ddof=1) / np.sqrt(len(trace_correlation))
            ),
            "crosstalk_recall_correlation_mean": float(np.nanmean(crosstalk_correlation)),
            "crosstalk_recall_correlation_ci95": float(
                1.96 * np.nanstd(crosstalk_correlation, ddof=1) / np.sqrt(len(crosstalk_correlation))
            ),
            "incremental_r2_mean": float(np.nanmean(incremental)),
            "incremental_r2_ci95": float(
                1.96 * np.nanstd(incremental, ddof=1) / np.sqrt(len(incremental))
            ),
            "e2_recall_half_life_mean": float(
                np.nanmean(alpha_recall_half_life[np.argmin(np.abs(alpha_values - args.e2_alpha))])
            ),
            "half_life_by_alpha": {
                str(alpha): {
                    "recall_mean": float(np.nanmean(alpha_recall_half_life[index])),
                    "trace_mean": float(np.nanmean(alpha_trace_half_life[index])),
                    "recall_censored_fraction": float(np.mean(~np.isfinite(alpha_recall_half_life[index]))),
                    "trace_censored_fraction": float(np.mean(~np.isfinite(alpha_trace_half_life[index]))),
                }
                for index, alpha in enumerate(alpha_values)
            },
        },
        "hypothesis_checks": {
            "exactly_reconstructs_e2": parity["passed"],
            "trace_survival_correlates_with_recall": bool(np.nanmean(trace_correlation) > 0),
            "crosstalk_correlates_negatively_with_recall": bool(np.nanmean(crosstalk_correlation) < 0),
            "trace_and_crosstalk_explain_variance_beyond_age": bool(np.nanmean(incremental) > 0),
            "higher_alpha_shortens_trace_half_life": bool(
                np.nanmean(alpha_trace_half_life[0]) > np.nanmean(alpha_trace_half_life[-1])
            ),
        },
        "runtime": runtime_metadata(),
        "outputs": {
            "raw_npz": str(prefix.with_suffix(".npz")),
            "summary_json": str(prefix.with_suffix(".json")),
            "figure_png": str(prefix.with_suffix(".png")),
            "figure_pdf": str(prefix.with_suffix(".pdf")),
        },
    }
    with prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(report), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2-data", type=Path, default=DEFAULT_E2_DATA)
    parser.add_argument("--checkpoint", default="ae_8")
    parser.add_argument("--e2-alpha", type=float, default=0.35)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if not 0 < args.e2_alpha <= 1 or any(not 0 < alpha <= 1 for alpha in args.alphas):
        parser.error("all learning rates must lie in (0, 1]")
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    report = run(args)
    print(json.dumps(json_ready({
        "exact_reconstruction": report["exact_reconstruction"],
        "summary": report["summary"],
        "hypothesis_checks": report["hypothesis_checks"],
        "figure": report["outputs"]["figure_png"],
    }), indent=2))


if __name__ == "__main__":
    main()
