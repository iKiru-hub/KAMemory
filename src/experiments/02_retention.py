"""E2a: sequential-memory retention, interference, and capacity.

The experiment preserves the validated notebook protocol used by E0/E1:
the EC-to-CA3 projection is initialized before row-wise sparse memories are
drawn from NumPy's seeded global RNG.  After each storage event, every memory
seen so far is retrieved without learning.

Run from the repository root:

    python3 src/experiments/02_retention.py --deterministic
"""

from __future__ import annotations

import argparse
import copy
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

from kamemory.data import generate_legacy_sparse_patterns
from kamemory.io import load_autoencoder_session, runtime_metadata
from kamemory.models import Autoencoder, BTSPMemory
from kamemory.utils import array_digest, cosine_similarity, seed_everything


SCHEMA_VERSION = 1
CONDITIONS = (
    "aligned",
    "fixed_permutation",
    "decoder_rescue",
    "no_plasticity",
)
DISPLAY_NAMES = {
    "aligned": "Aligned IS",
    "fixed_permutation": "Fixed permutation",
    "decoder_rescue": "Permutation + decoder rescue",
    "no_plasticity": "No plasticity",
}
COLORS = {
    "aligned": "#2878B5",
    "fixed_permutation": "#D95F59",
    "decoder_rescue": "#2A9D8F",
    "no_plasticity": "#7A7A7A",
}
LINESTYLES = {"decoder_rescue": "--"}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


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


def session_parameters(info: dict) -> dict[str, int | float]:
    if "network_params" in info:
        params = info["network_params"]
        return {
            "input_dim": int(params["dim_ei"]),
            "dim_ca1": int(params["dim_ca1"]),
            "dim_ca3": int(params["dim_ca3"]),
            "input_active": int(params["K_ei"]),
            "output_active": int(params["K_eo"]),
            "ca1_active": int(params["K_ca1"]),
            "ca3_active": int(params["K_ca3"]),
            "beta": float(params["beta_ca1"]),
        }
    return {
        "input_dim": int(info["dim_ei"]),
        "dim_ca1": int(info["dim_ca1"]),
        "dim_ca3": int(info["dim_ca3"]),
        "input_active": int(info["K"]),
        "output_active": int(info["K"]),
        "ca1_active": int(info["K_lat"]),
        "ca3_active": min(22, int(info["dim_ca3"])),
        "beta": float(info["beta"]),
    }


def make_base_memory(autoencoder: Autoencoder, params: dict, alpha: float) -> BTSPMemory:
    return BTSPMemory.from_autoencoder(
        autoencoder,
        K_lat=params["ca1_active"],
        K_out=params["output_active"],
        dim_ca3=params["dim_ca3"],
        K_ca3=params["ca3_active"],
        beta=params["beta"],
        alpha=alpha,
    )


def configure_condition(
    base: BTSPMemory, condition: str, permutation: torch.Tensor
) -> BTSPMemory:
    memory = copy.deepcopy(base)
    if condition == "decoder_rescue":
        memory.W_ca1_eo.copy_(base.W_ca1_eo[:, permutation])
        probe = torch.randn(base._dim_ca1, 1)
        if not torch.allclose(
            base.W_ca1_eo @ probe,
            memory.W_ca1_eo @ probe[permutation],
            atol=1e-5,
            rtol=1e-5,
        ):
            raise AssertionError("matched decoder has the wrong permutation orientation")
    if condition == "no_plasticity":
        memory.learning_enabled = False
    return memory


def top_k_f1(target: np.ndarray, output: np.ndarray, k: int) -> float:
    target_indices = set(np.flatnonzero(target > 0.5).tolist())
    predicted_indices = set(np.argpartition(output, -k)[-k:].tolist())
    overlap = len(target_indices & predicted_indices)
    precision = overlap / max(len(predicted_indices), 1)
    recall = overlap / max(len(target_indices), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def contiguous_capacity(score_by_load: np.ndarray, threshold: float) -> int:
    """Largest one-indexed load before the first sub-threshold mean score."""

    valid = np.asarray(score_by_load, dtype=float)
    below = np.flatnonzero((~np.isfinite(valid)) | (valid < threshold))
    return int(below[0]) if len(below) else int(len(valid))


def retention_lifetimes(score_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Consecutive successful recalls from storage, measured in later stores."""

    loads, memories = score_matrix.shape
    lifetimes = np.zeros(memories, dtype=np.int32)
    for memory_index in range(memories):
        trace = score_matrix[memory_index:, memory_index]
        finite = trace[np.isfinite(trace)]
        failures = np.flatnonzero(finite < threshold)
        successful_points = int(failures[0]) if len(failures) else len(finite)
        lifetimes[memory_index] = max(successful_points - 1, 0)
    return lifetimes


def run_seed(
    seed: int,
    autoencoder: Autoencoder,
    params: dict,
    *,
    num_patterns: int,
    alpha: float,
    capacity_threshold: float,
) -> dict[str, object]:
    seed_everything(seed)
    base = make_base_memory(autoencoder, params, alpha)
    patterns = generate_legacy_sparse_patterns(
        num_patterns, params["input_active"], params["input_dim"]
    )
    with torch.no_grad():
        codes = autoencoder.encode(torch.as_tensor(patterns)).detach().cpu()
    control_rng = np.random.default_rng(seed + 30_000)
    permutation = torch.as_tensor(
        control_rng.permutation(params["dim_ca1"]), dtype=torch.long
    )
    signals = {
        "aligned": codes,
        "fixed_permutation": codes[:, permutation],
        "decoder_rescue": codes[:, permutation],
        "no_plasticity": codes,
    }
    chance = params["input_active"] / params["input_dim"]
    results: dict[str, object] = {}

    for condition in CONDITIONS:
        memory = configure_condition(base, condition, permutation)
        cosine = np.full((num_patterns, num_patterns), np.nan, dtype=np.float32)
        corrected = np.full_like(cosine, np.nan)
        f1 = np.full_like(cosine, np.nan)
        mse = np.full_like(cosine, np.nan)
        ca1_similarity = np.full_like(cosine, np.nan)

        for load_index in range(num_patterns):
            memory.store(
                patterns[load_index],
                instructive_signal=signals[condition][load_index],
            )
            with torch.no_grad():
                for memory_index in range(load_index + 1):
                    target = torch.as_tensor(patterns[memory_index]).reshape(-1, 1)
                    output, ca1 = memory.retrieve(target, return_ca1=True)
                    raw_cosine = float(cosine_similarity(target, output).item())
                    output_np = output.detach().cpu().numpy().reshape(-1)
                    cosine[load_index, memory_index] = raw_cosine
                    corrected[load_index, memory_index] = (
                        raw_cosine - chance
                    ) / (1 - chance)
                    f1[load_index, memory_index] = top_k_f1(
                        patterns[memory_index], output_np, params["input_active"]
                    )
                    mse[load_index, memory_index] = float(
                        torch.mean((output - target) ** 2).item()
                    )
                    ca1_similarity[load_index, memory_index] = float(
                        cosine_similarity(
                            ca1.detach().cpu(), codes[memory_index].reshape(-1, 1)
                        ).item()
                    )

        mean_by_load = np.asarray(
            [np.nanmean(corrected[index, : index + 1]) for index in range(num_patterns)]
        )
        retained_count = np.asarray(
            [
                np.count_nonzero(corrected[index, : index + 1] >= capacity_threshold)
                for index in range(num_patterns)
            ],
            dtype=np.int32,
        )
        results[condition] = {
            "cosine": cosine,
            "chance_corrected": corrected,
            "top_k_f1": f1,
            "mse": mse,
            "ca1_similarity": ca1_similarity,
            "mean_by_load": mean_by_load,
            "retained_count": retained_count,
            "load_capacity": contiguous_capacity(mean_by_load, capacity_threshold),
            "retention_lifetimes": retention_lifetimes(
                corrected, capacity_threshold
            ),
            "final_weights": memory.W_ca3_ca1.detach().cpu().numpy().copy(),
        }

    results["metadata"] = {
        "seed": seed,
        "patterns": patterns,
        "permutation": permutation.numpy(),
        "base_ca3_projection": base.W_ei_ca3.detach().cpu().numpy().copy(),
    }
    return results


def values_by_age(matrix: np.ndarray) -> list[np.ndarray]:
    num_seeds, num_loads, _ = matrix.shape
    values = []
    for age in range(num_loads):
        age_values = []
        for seed in range(num_seeds):
            load_indices = np.arange(age, num_loads)
            memory_indices = load_indices - age
            age_values.extend(matrix[seed, load_indices, memory_indices].tolist())
        values.append(np.asarray(age_values))
    return values


def mean_and_ci(values: np.ndarray, axis=0) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=axis)
    count = np.sum(np.isfinite(values), axis=axis)
    sd = np.nanstd(values, axis=axis, ddof=1)
    ci = 1.96 * sd / np.sqrt(np.maximum(count, 1))
    return mean, ci


def make_figure(
    arrays: dict[str, np.ndarray],
    threshold: float,
    params: dict,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    heatmap_ax, age_ax, load_ax, count_ax, capacity_ax, remote_ax = axes.flat

    aligned_values = arrays["chance_corrected"][0]
    aligned_count = np.sum(np.isfinite(aligned_values), axis=0)
    aligned_mean = np.divide(
        np.nansum(aligned_values, axis=0),
        aligned_count,
        out=np.full(aligned_count.shape, np.nan, dtype=float),
        where=aligned_count > 0,
    )
    image = heatmap_ax.imshow(
        aligned_mean,
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    heatmap_ax.set_xlabel("Memory storage order")
    heatmap_ax.set_ylabel("Total memories stored")
    heatmap_ax.set_title("A  Aligned retention matrix")
    figure.colorbar(image, ax=heatmap_ax, label="Chance-corrected cosine")

    for condition_index, condition in enumerate(CONDITIONS):
        matrix = arrays["chance_corrected"][condition_index]
        age_series = values_by_age(matrix)
        mean = np.asarray([np.nanmean(values) for values in age_series])
        ci = np.asarray(
            [
                1.96 * np.nanstd(values, ddof=1) / np.sqrt(max(len(values), 1))
                if len(values) > 1
                else 0
                for values in age_series
            ]
        )
        ages = np.arange(len(mean))
        age_ax.plot(
            ages,
            mean,
            label=DISPLAY_NAMES[condition],
            color=COLORS[condition],
            linestyle=LINESTYLES.get(condition, "-"),
        )
        age_ax.fill_between(ages, mean - ci, mean + ci, color=COLORS[condition], alpha=0.15)
    age_ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
    age_ax.set_xlabel("Memory age (subsequent storage events)")
    age_ax.set_ylabel("Chance-corrected cosine")
    age_ax.set_title("B  Retention decays with memory age")
    age_ax.legend(frameon=False, fontsize=8)

    loads = np.arange(1, arrays["mean_by_load"].shape[-1] + 1)
    for condition_index, condition in enumerate(CONDITIONS):
        mean, ci = mean_and_ci(arrays["mean_by_load"][condition_index], axis=0)
        load_ax.plot(
            loads,
            mean,
            color=COLORS[condition],
            linestyle=LINESTYLES.get(condition, "-"),
            label=DISPLAY_NAMES[condition],
        )
        load_ax.fill_between(loads, mean - ci, mean + ci, color=COLORS[condition], alpha=0.15)
    load_ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
    load_ax.set_xlabel("Total memories stored")
    load_ax.set_ylabel("Mean chance-corrected cosine")
    load_ax.set_title("C  Population performance under load")

    for condition_index, condition in enumerate(CONDITIONS):
        mean, ci = mean_and_ci(arrays["retained_count"][condition_index], axis=0)
        count_ax.plot(
            loads,
            mean,
            color=COLORS[condition],
            linestyle=LINESTYLES.get(condition, "-"),
            label=DISPLAY_NAMES[condition],
        )
        count_ax.fill_between(loads, mean - ci, mean + ci, color=COLORS[condition], alpha=0.15)
    count_ax.plot(loads, loads, color="black", linestyle=":", linewidth=1, label="all stored")
    count_ax.set_xlabel("Total memories stored")
    count_ax.set_ylabel("Memories above threshold")
    count_ax.set_title("D  Effective retained-memory count")

    capacities = arrays["load_capacity"]
    for condition_index, condition in enumerate(CONDITIONS):
        values = capacities[condition_index]
        jitter = np.linspace(-0.10, 0.10, len(values))
        capacity_ax.scatter(
            condition_index + jitter,
            values,
            color=COLORS[condition],
            s=25,
            alpha=0.7,
        )
        mean = np.mean(values)
        ci = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
        capacity_ax.errorbar(condition_index, mean, yerr=ci, color="black", fmt="o", capsize=4)
    capacity_ax.set_xticks(
        range(len(CONDITIONS)),
        [DISPLAY_NAMES[name] for name in CONDITIONS],
        rotation=20,
        ha="right",
    )
    capacity_ax.set_ylabel("Contiguous load capacity")
    capacity_ax.set_title("E  Capacity across paired seeds")

    age_bins = (
        ("Immediate\n(age 0)", 0, 1),
        ("Recent\n(ages 1–4)", 1, 5),
        ("Middle\n(ages 5–14)", 5, 15),
        ("Remote\n(age ≥15)", 15, arrays["chance_corrected"].shape[-1]),
    )
    for condition_index, condition in enumerate(CONDITIONS):
        age_series = values_by_age(arrays["chance_corrected"][condition_index])
        bin_means = []
        for _, start, stop in age_bins:
            selected = [values for values in age_series[start:stop] if len(values)]
            bin_means.append(float(np.nanmean(np.concatenate(selected))) if selected else np.nan)
        remote_ax.plot(
            np.arange(len(age_bins)),
            bin_means,
            marker="o",
            color=COLORS[condition],
            linestyle=LINESTYLES.get(condition, "-"),
            label=DISPLAY_NAMES[condition],
        )
    remote_ax.set_xticks(
        range(len(age_bins)), [label for label, _, _ in age_bins]
    )
    remote_ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
    remote_ax.set_ylabel("Chance-corrected cosine")
    remote_ax.set_title("F  Immediate, recent, middle, remote recall")

    figure.suptitle(
        "E2a — Coordinate alignment protects sequential memories from interference",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    info, autoencoder = load_autoencoder_session(args.checkpoint, map_location=args.device)
    params = session_parameters(info)
    if args.ca3_active is not None:
        params["ca3_active"] = args.ca3_active
    autoencoder.eval()
    seeds = [args.seed_start + index for index in range(args.num_seeds)]
    seed_results = [
        run_seed(
            seed,
            autoencoder,
            params,
            num_patterns=args.num_patterns,
            alpha=args.alpha,
            capacity_threshold=args.capacity_threshold,
        )
        for seed in seeds
    ]

    metric_names = (
        "cosine",
        "chance_corrected",
        "top_k_f1",
        "mse",
        "ca1_similarity",
        "mean_by_load",
        "retained_count",
        "load_capacity",
        "retention_lifetimes",
        "final_weights",
    )
    arrays = {
        metric: np.stack(
            [
                np.stack([result[condition][metric] for result in seed_results])
                for condition in CONDITIONS
            ]
        )
        for metric in metric_names
    }
    aligned_capacity = arrays["load_capacity"][0]
    fixed_capacity = arrays["load_capacity"][1]
    rescued_capacity = arrays["load_capacity"][2]
    no_plasticity_capacity = arrays["load_capacity"][3]
    fixed_rescue_weights_identical = bool(
        np.array_equal(arrays["final_weights"][1], arrays["final_weights"][2])
    )
    checks = {
        "all_previous_memories_tested_after_each_store": bool(
            np.all(
                np.isfinite(arrays["chance_corrected"][:, :, np.tril_indices(args.num_patterns)[0], np.tril_indices(args.num_patterns)[1]])
            )
        ),
        "future_memories_are_not_tested": bool(
            np.all(
                np.isnan(arrays["chance_corrected"][:, :, np.triu_indices(args.num_patterns, 1)[0], np.triu_indices(args.num_patterns, 1)[1]])
            )
        ),
        "aligned_capacity_exceeds_fixed": bool(np.mean(aligned_capacity - fixed_capacity) > 0),
        "aligned_capacity_exceeds_no_plasticity": bool(
            np.mean(aligned_capacity - no_plasticity_capacity) > 0
        ),
        "decoder_rescues_fixed_capacity": bool(np.mean(rescued_capacity - fixed_capacity) > 0),
        "rescue_approaches_aligned_capacity": bool(
            abs(np.mean(rescued_capacity) - np.mean(aligned_capacity)) <= 2
        ),
        "fixed_and_rescue_learning_identical": fixed_rescue_weights_identical,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "e2a_retention"
    metadata = [result["metadata"] for result in seed_results]
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        seeds=np.asarray(seeds),
        conditions=np.asarray(CONDITIONS),
        patterns=np.stack([item["patterns"] for item in metadata]),
        permutations=np.stack([item["permutation"] for item in metadata]),
        base_ca3_projection=np.stack(
            [item["base_ca3_projection"] for item in metadata]
        ),
        capacity_threshold=np.asarray(args.capacity_threshold),
        **arrays,
    )
    capacity_summary = {}
    age_bin_summary = {}
    forgetting_profile = {}
    age_bin_ranges = {
        "immediate": (0, 1),
        "recent": (1, min(5, args.num_patterns)),
        "middle": (min(5, args.num_patterns), min(15, args.num_patterns)),
        "remote": (min(15, args.num_patterns), args.num_patterns),
    }
    for condition_index, condition in enumerate(CONDITIONS):
        values = arrays["load_capacity"][condition_index].astype(float)
        capacity_summary[condition] = {
            "mean_load_capacity": float(np.mean(values)),
            "sd_load_capacity": float(np.std(values, ddof=1)),
            "ci95_low": float(
                np.mean(values) - 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
            ),
            "ci95_high": float(
                np.mean(values) + 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
            ),
            "capacity_per_ca1_unit": float(np.mean(values) / params["dim_ca1"]),
            "capacity_per_ca3_unit": float(np.mean(values) / params["dim_ca3"]),
            "final_retained_count": float(
                np.mean(arrays["retained_count"][condition_index, :, -1])
            ),
        }
        age_series = values_by_age(arrays["chance_corrected"][condition_index])
        age_curve = np.asarray(
            [float(np.nanmean(values)) for values in age_series], dtype=float
        )
        adjacent_drops = age_curve[:-1] - age_curve[1:]
        half_level = 0.5 * age_curve[0]
        half_indices = np.flatnonzero(age_curve < half_level)
        largest_drop = float(np.nanmax(adjacent_drops)) if len(adjacent_drops) else 0.0
        forgetting_profile[condition] = {
            "mean_by_memory_age": age_curve,
            "largest_single_age_drop": largest_drop,
            "largest_drop_fraction_of_immediate": float(
                largest_drop / max(abs(age_curve[0]), 1e-12)
            ),
            "half_immediate_recall_age": (
                int(half_indices[0]) if len(half_indices) else None
            ),
            "classification": (
                "catastrophic"
                if largest_drop >= 0.5 * abs(age_curve[0])
                else "gradual"
            ),
        }
        age_bin_summary[condition] = {}
        for label, (start, stop) in age_bin_ranges.items():
            selected = [values for values in age_series[start:stop] if len(values)]
            age_bin_summary[condition][label] = (
                float(np.nanmean(np.concatenate(selected))) if selected else None
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E2a_retention_capacity",
        "configuration": vars(args),
        "runtime": runtime_metadata(),
        "checkpoint_info": info,
        "network_parameters": params,
        "protocol": {
            "dataset": "validated_notebook_rowwise_numpy_choice",
            "operation_order": "seed -> initialize EC-CA3 -> draw patterns -> sequential store/recall",
            "capacity_definition": (
                "largest contiguous total load for which the mean chance-corrected "
                "cosine across all memories stored so far remains at or above threshold"
            ),
            "threshold_origin": (
                "pre-existing notebook raw-cosine threshold 0.8, converted to "
                "chance-corrected coordinates for input sparsity 5/50"
            ),
            "pattern_sha256_by_seed": [
                array_digest(item["patterns"]) for item in metadata
            ],
        },
        "capacity_summary": capacity_summary,
        "retention_by_age_bin": age_bin_summary,
        "forgetting_profile": forgetting_profile,
        "paired_effects": {
            "aligned_minus_fixed_capacity": aligned_capacity - fixed_capacity,
            "aligned_minus_no_plasticity_capacity": aligned_capacity
            - no_plasticity_capacity,
            "rescue_minus_fixed_capacity": rescued_capacity - fixed_capacity,
            "rescue_minus_aligned_capacity": rescued_capacity - aligned_capacity,
        },
        "checks": checks,
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
    make_figure(arrays, args.capacity_threshold, params, prefix.with_suffix(".png"))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="ae_8")
    parser.add_argument("--seed-start", type=int, default=700)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--num-patterns", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--ca3-active", type=int, default=22)
    # Historical threshold: raw cosine 0.8 at chance 5/50 -> 7/9 corrected.
    parser.add_argument("--capacity-threshold", type=float, default=7 / 9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.num_seeds < 2:
        parser.error("--num-seeds must be at least 2")
    if args.num_patterns < 2:
        parser.error("--num-patterns must be at least 2")
    if not 0 <= args.capacity_threshold <= 1:
        parser.error("--capacity-threshold must lie in [0, 1]")
    return args


def main() -> int:
    args = parse_args()
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    report = run(args)
    concise = {
        "capacity_summary": report["capacity_summary"],
        "checks": report["checks"],
        "outputs": report["outputs"],
    }
    print(json.dumps(json_ready(concise), indent=2))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
