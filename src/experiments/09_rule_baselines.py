"""E8: rule-specific baselines for alignment, retention, and interference.

The experiment separates two questions:

1. Does decoder-coordinate alignment matter across local learning rules?
2. Does the validated target-gated overwrite have a distinctive sequential
   memory profile relative to bounded Hebbian and local delta-rule storage?

Alternative learning rates are selected on disjoint development seeds.  They
must first match the validated rule's early immediate recall (loads 1--5)
within a prespecified tolerance; retention-matrix mean breaks ties. Final comparisons use new
paired seeds, identical memories, CA3 projections, storage order, targets,
and initial weights.

Run from the repository root:

    .venv/kamvenv/bin/python src/experiments/09_rule_baselines.py --deterministic
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
from kamemory.io import PATHS, load_autoencoder_session, load_config, runtime_metadata
from kamemory.models import Autoencoder, BTSPMemory
from kamemory.utils import array_digest, cosine_similarity, seed_everything


SCHEMA_VERSION = 1
RULES = ("target_gated", "hebbian", "delta")
ALIGNMENT_CONDITIONS = ("aligned", "fixed_permutation", "decoder_rescue")
RULE_NAMES = {
    "target_gated": "Target-gated overwrite",
    "hebbian": "Potentiation-only Hebbian",
    "delta": "Local delta rule",
}
SHORT_RULE_NAMES = {
    "target_gated": "Target-gated",
    "hebbian": "Hebbian",
    "delta": "Delta",
}
CONDITION_NAMES = {
    "aligned": "Aligned",
    "fixed_permutation": "Fixed permutation",
    "decoder_rescue": "Matched decoder",
}
RULE_COLORS = {
    "target_gated": "#2878B5",
    "hebbian": "#E07A32",
    "delta": "#2A9D8F",
}
CONDITION_MARKERS = {
    "aligned": "o",
    "fixed_permutation": "s",
    "decoder_rescue": "D",
}
DEFAULT_CONFIG = PATHS.configs / "rule_baselines.json"
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


def make_base_memory(
    autoencoder: Autoencoder, params: dict, *, alpha: float = 0.35
) -> BTSPMemory:
    return BTSPMemory.from_autoencoder(
        autoencoder,
        K_lat=params["ca1_active"],
        K_out=params["output_active"],
        dim_ca3=params["dim_ca3"],
        K_ca3=params["ca3_active"],
        beta=params["beta"],
        alpha=alpha,
    )


def prepare_seed(
    seed: int,
    autoencoder: Autoencoder,
    params: dict,
    *,
    num_patterns: int,
) -> dict[str, object]:
    """Prepare paired assets in the validated notebook RNG order."""

    seed_everything(seed)
    base = make_base_memory(autoencoder, params)
    patterns = generate_legacy_sparse_patterns(
        num_patterns, params["input_active"], params["input_dim"]
    )
    with torch.no_grad():
        codes = autoencoder.encode(torch.as_tensor(patterns)).detach().cpu()
    rng = np.random.default_rng(seed + 80_000)
    permutation = torch.as_tensor(
        rng.permutation(params["dim_ca1"]), dtype=torch.long
    )
    return {
        "seed": seed,
        "base": base,
        "patterns": patterns,
        "codes": codes,
        "permutation": permutation,
    }


def configured_memory(
    base: BTSPMemory,
    *,
    rule: str,
    learning_rate: float,
    decoder_permutation: torch.Tensor | None = None,
) -> BTSPMemory:
    memory = copy.deepcopy(base)
    memory.reset_memory()
    memory.set_plasticity_rule(rule)
    memory.set_alpha(learning_rate)
    if decoder_permutation is not None:
        memory.W_ca1_eo.copy_(base.W_ca1_eo[:, decoder_permutation])
    return memory


def chance_correct(raw_cosine: float, params: dict) -> float:
    chance = params["input_active"] / params["input_dim"]
    return (raw_cosine - chance) / (1 - chance)


def contiguous_capacity(score_by_load: np.ndarray, threshold: float) -> int:
    below = np.flatnonzero(
        (~np.isfinite(score_by_load)) | (score_by_load < threshold)
    )
    return int(below[0]) if len(below) else int(len(score_by_load))


def age_curve(score_matrix: np.ndarray) -> np.ndarray:
    num_loads = score_matrix.shape[0]
    values = np.full(num_loads, np.nan, dtype=np.float64)
    for age in range(num_loads):
        load_indices = np.arange(age, num_loads)
        memory_indices = load_indices - age
        values[age] = np.nanmean(score_matrix[load_indices, memory_indices])
    return values


def recall_half_life(curve: np.ndarray) -> int:
    threshold = 0.5 * curve[0]
    below = np.flatnonzero(curve < threshold)
    return int(below[0]) if len(below) else int(len(curve))


def sequential_memory(
    base: BTSPMemory,
    patterns: np.ndarray,
    codes: torch.Tensor,
    params: dict,
    *,
    rule: str,
    learning_rate: float,
    capacity_threshold: float,
) -> dict[str, np.ndarray | float | int]:
    memory = configured_memory(
        base, rule=rule, learning_rate=learning_rate
    )
    num_patterns = len(patterns)
    corrected = np.full((num_patterns, num_patterns), np.nan, dtype=np.float32)
    raw_cosine = np.full_like(corrected, np.nan)
    high_weight_fraction = np.zeros(num_patterns, dtype=np.float32)
    weight_norm = np.zeros(num_patterns, dtype=np.float32)

    for load_index in range(num_patterns):
        memory.store(
            patterns[load_index], instructive_signal=codes[load_index]
        )
        weights = memory.W_ca3_ca1.detach()
        high_weight_fraction[load_index] = float((weights >= 0.95).float().mean())
        weight_norm[load_index] = float(torch.linalg.vector_norm(weights))
        with torch.no_grad():
            for memory_index in range(load_index + 1):
                target = torch.as_tensor(patterns[memory_index]).reshape(-1, 1)
                output = memory.retrieve(target)
                raw = float(cosine_similarity(target, output).item())
                raw_cosine[load_index, memory_index] = raw
                corrected[load_index, memory_index] = chance_correct(raw, params)

    mean_by_load = np.asarray(
        [np.nanmean(corrected[index, : index + 1]) for index in range(num_patterns)]
    )
    curve = age_curve(corrected)
    diagonal = np.diag(corrected)
    early_stop = min(5, len(diagonal))
    return {
        "raw_cosine": raw_cosine,
        "chance_corrected": corrected,
        "mean_by_load": mean_by_load,
        "age_curve": curve,
        "integrated_recall": float(np.nanmean(corrected)),
        "immediate_recall": float(np.nanmean(diagonal)),
        "early_immediate_recall": float(np.nanmean(diagonal[:early_stop])),
        "load_capacity": contiguous_capacity(mean_by_load, capacity_threshold),
        "half_life": recall_half_life(curve),
        "high_weight_fraction": high_weight_fraction,
        "weight_norm": weight_norm,
        "final_weights": memory.W_ca3_ca1.detach().cpu().numpy().copy(),
    }


def alignment_endpoint(
    base: BTSPMemory,
    patterns: np.ndarray,
    codes: torch.Tensor,
    permutation: torch.Tensor,
    params: dict,
    *,
    rule: str,
    learning_rate: float,
    condition: str,
) -> dict[str, object]:
    if condition == "aligned":
        signals = codes
        decoder_permutation = None
    elif condition in {"fixed_permutation", "decoder_rescue"}:
        signals = codes[:, permutation]
        decoder_permutation = permutation if condition == "decoder_rescue" else None
    else:
        raise ValueError(f"unknown alignment condition: {condition}")

    memory = configured_memory(
        base,
        rule=rule,
        learning_rate=learning_rate,
        decoder_permutation=decoder_permutation,
    )
    for pattern, signal in zip(patterns, signals):
        memory.store(pattern, instructive_signal=signal)

    raw_values = []
    corrected_values = []
    with torch.no_grad():
        for pattern in patterns:
            target = torch.as_tensor(pattern).reshape(-1, 1)
            output = memory.retrieve(target)
            raw = float(cosine_similarity(target, output).item())
            raw_values.append(raw)
            corrected_values.append(chance_correct(raw, params))
    return {
        "raw_cosine": np.asarray(raw_values),
        "chance_corrected": np.asarray(corrected_values),
        "mean_raw_cosine": float(np.mean(raw_values)),
        "mean_chance_corrected": float(np.mean(corrected_values)),
        "final_weights": memory.W_ca3_ca1.detach().cpu().numpy().copy(),
    }


def development_sweep(
    autoencoder: Autoencoder,
    params: dict,
    *,
    seeds: list[int],
    num_patterns: int,
    learning_rates: np.ndarray,
    capacity_threshold: float,
) -> dict[str, np.ndarray]:
    early_immediate = np.zeros((len(RULES), len(learning_rates), len(seeds)))
    integrated = np.zeros_like(early_immediate)
    capacity = np.zeros_like(early_immediate)
    for seed_index, seed in enumerate(seeds):
        assets = prepare_seed(
            seed, autoencoder, params, num_patterns=num_patterns
        )
        for rule_index, rule in enumerate(RULES):
            for rate_index, learning_rate in enumerate(learning_rates):
                result = sequential_memory(
                    assets["base"],
                    assets["patterns"],
                    assets["codes"],
                    params,
                    rule=rule,
                    learning_rate=float(learning_rate),
                    capacity_threshold=capacity_threshold,
                )
                early_immediate[rule_index, rate_index, seed_index] = result[
                    "early_immediate_recall"
                ]
                integrated[rule_index, rate_index, seed_index] = result[
                    "integrated_recall"
                ]
                capacity[rule_index, rate_index, seed_index] = result[
                    "load_capacity"
                ]
    return {
        "early_immediate_recall": early_immediate,
        "integrated_recall": integrated,
        "load_capacity": capacity,
    }


def select_learning_rates(
    sweep: dict[str, np.ndarray],
    learning_rates: np.ndarray,
    *,
    reference_rate: float,
    tolerance: float,
) -> tuple[dict[str, float], dict[str, object]]:
    reference_indices = np.flatnonzero(np.isclose(learning_rates, reference_rate))
    if not len(reference_indices):
        raise ValueError("reference learning rate must appear in the tuning grid")
    reference_index = int(reference_indices[0])
    reference_rule_index = RULES.index("target_gated")
    reference_immediate = float(
        np.mean(
            sweep["early_immediate_recall"][reference_rule_index, reference_index]
        )
    )

    selected = {"target_gated": float(reference_rate)}
    details: dict[str, object] = {
        "reference_early_immediate_recall": reference_immediate,
        "criterion": (
            "match target-gated early immediate recall over loads 1-5 within "
            "tolerance, then maximize "
            "development retention-matrix mean"
        ),
        "rules": {},
    }
    for rule_index, rule in enumerate(RULES):
        immediate_mean = np.mean(
            sweep["early_immediate_recall"][rule_index], axis=1
        )
        integrated_mean = np.mean(sweep["integrated_recall"][rule_index], axis=1)
        immediate_difference = np.abs(immediate_mean - reference_immediate)
        if rule == "target_gated":
            chosen = reference_index
            eligible = np.asarray([reference_index])
            reason = "prespecified validated E2 learning rate"
        else:
            eligible = np.flatnonzero(immediate_difference <= tolerance)
            if len(eligible):
                chosen = int(eligible[np.argmax(integrated_mean[eligible])])
                reason = "best retention among immediate-recall-matched rates"
            else:
                nearest_difference = np.min(immediate_difference)
                nearest = np.flatnonzero(
                    np.isclose(immediate_difference, nearest_difference)
                )
                chosen = int(nearest[np.argmax(integrated_mean[nearest])])
                reason = "closest immediate recall; no rate met tolerance"
            selected[rule] = float(learning_rates[chosen])
        details["rules"][rule] = {
            "selected_learning_rate": float(learning_rates[chosen]),
            "selected_early_immediate_recall": float(immediate_mean[chosen]),
            "selected_integrated_recall": float(integrated_mean[chosen]),
            "absolute_immediate_difference": float(immediate_difference[chosen]),
            "eligible_rate_indices": eligible,
            "selection_reason": reason,
        }
    return selected, details


def evaluate(
    autoencoder: Autoencoder,
    params: dict,
    *,
    seeds: list[int],
    num_patterns: int,
    endpoint_patterns: int,
    selected_rates: dict[str, float],
    capacity_threshold: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    sequential_results: list[list[dict[str, object]]] = [
        [] for _ in RULES
    ]
    endpoint_results: list[list[list[dict[str, object]]]] = [
        [[] for _ in ALIGNMENT_CONDITIONS] for _ in RULES
    ]
    metadata = []

    for seed in seeds:
        assets = prepare_seed(seed, autoencoder, params, num_patterns=num_patterns)
        for rule_index, rule in enumerate(RULES):
            sequential_results[rule_index].append(
                sequential_memory(
                    assets["base"],
                    assets["patterns"],
                    assets["codes"],
                    params,
                    rule=rule,
                    learning_rate=selected_rates[rule],
                    capacity_threshold=capacity_threshold,
                )
            )
            for condition_index, condition in enumerate(ALIGNMENT_CONDITIONS):
                endpoint_results[rule_index][condition_index].append(
                    alignment_endpoint(
                        assets["base"],
                        assets["patterns"][:endpoint_patterns],
                        assets["codes"][:endpoint_patterns],
                        assets["permutation"],
                        params,
                        rule=rule,
                        learning_rate=selected_rates[rule],
                        condition=condition,
                    )
                )
        metadata.append(
            {
                "seed": seed,
                "patterns": assets["patterns"],
                "permutation": assets["permutation"].numpy(),
                "base_ca3_projection": assets["base"].W_ei_ca3.detach().cpu().numpy().copy(),
            }
        )

    sequential_metrics = (
        "raw_cosine",
        "chance_corrected",
        "mean_by_load",
        "age_curve",
        "integrated_recall",
        "immediate_recall",
        "early_immediate_recall",
        "load_capacity",
        "half_life",
        "high_weight_fraction",
        "weight_norm",
        "final_weights",
    )
    endpoint_metrics = (
        "raw_cosine",
        "chance_corrected",
        "mean_raw_cosine",
        "mean_chance_corrected",
        "final_weights",
    )
    arrays = {
        metric: np.stack(
            [
                np.stack([result[metric] for result in rule_results])
                for rule_results in sequential_results
            ]
        )
        for metric in sequential_metrics
    }
    for metric in endpoint_metrics:
        arrays[f"alignment_{metric}"] = np.stack(
            [
                np.stack(
                    [
                        np.stack([result[metric] for result in condition_results])
                        for condition_results in rule_results
                    ]
                )
                for rule_results in endpoint_results
            ]
        )
    return arrays, metadata


def scalar_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    ci = 1.96 * sd / np.sqrt(len(values))
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
    }


def make_figure(
    tuning: dict[str, np.ndarray],
    learning_rates: np.ndarray,
    selected_rates: dict[str, float],
    arrays: dict[str, np.ndarray],
    threshold: float,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    tuning_ax, alignment_ax, age_ax, integrated_ax, immediate_ax, weights_ax = axes.flat

    for rule_index, rule in enumerate(RULES):
        x = np.mean(tuning["early_immediate_recall"][rule_index], axis=1)
        y = np.mean(tuning["integrated_recall"][rule_index], axis=1)
        tuning_ax.plot(x, y, color=RULE_COLORS[rule], alpha=0.65)
        tuning_ax.scatter(x, y, color=RULE_COLORS[rule], s=30, label=SHORT_RULE_NAMES[rule])
        chosen = int(np.flatnonzero(np.isclose(learning_rates, selected_rates[rule]))[0])
        tuning_ax.scatter(
            x[chosen], y[chosen], color=RULE_COLORS[rule], marker="*", s=180,
            edgecolor="black", linewidth=0.7, zorder=5
        )
        tuning_ax.annotate(
            f"η={selected_rates[rule]:g}", (x[chosen], y[chosen]),
            xytext=(5, 5), textcoords="offset points", fontsize=8
        )
    tuning_ax.set_xlabel("Development early immediate recall (loads 1–5)")
    tuning_ax.set_ylabel("Development retention-matrix mean")
    tuning_ax.set_title("A  Prespecified rate selection")
    tuning_ax.legend(frameon=False, fontsize=8)

    offsets = np.linspace(-0.20, 0.20, len(ALIGNMENT_CONDITIONS))
    for rule_index, rule in enumerate(RULES):
        for condition_index, condition in enumerate(ALIGNMENT_CONDITIONS):
            values = arrays["alignment_mean_chance_corrected"][rule_index, condition_index]
            x_center = rule_index + offsets[condition_index]
            jitter = np.linspace(-0.035, 0.035, len(values))
            alignment_ax.scatter(
                x_center + jitter,
                values,
                s=16,
                alpha=0.35,
                color=RULE_COLORS[rule],
                marker=CONDITION_MARKERS[condition],
            )
            ci = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
            alignment_ax.errorbar(
                x_center,
                np.mean(values),
                yerr=ci,
                color="black",
                marker=CONDITION_MARKERS[condition],
                markersize=6,
                capsize=3,
                linestyle="none",
            )
    alignment_ax.axhline(0, color="black", linestyle="--", linewidth=1)
    alignment_ax.set_xticks(
        range(len(RULES)), ["Target-\ngated", "Hebbian", "Delta"]
    )
    alignment_ax.set_ylabel("Chance-corrected endpoint recall")
    alignment_ax.set_title("B  Alignment constrains every rule")
    handles = [
        plt.Line2D(
            [], [], color="black", marker=CONDITION_MARKERS[name], linestyle="none",
            label=CONDITION_NAMES[name]
        )
        for name in ALIGNMENT_CONDITIONS
    ]
    alignment_ax.legend(handles=handles, frameon=False, fontsize=8)

    ages = np.arange(arrays["age_curve"].shape[-1])
    for rule_index, rule in enumerate(RULES):
        curves = arrays["age_curve"][rule_index]
        mean = np.mean(curves, axis=0)
        ci = 1.96 * np.std(curves, axis=0, ddof=1) / np.sqrt(curves.shape[0])
        age_ax.plot(ages, mean, color=RULE_COLORS[rule], label=SHORT_RULE_NAMES[rule])
        age_ax.fill_between(ages, mean - ci, mean + ci, color=RULE_COLORS[rule], alpha=0.15)
    age_ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
    age_ax.set_xlabel("Memory age (subsequent stores)")
    age_ax.set_ylabel("Chance-corrected recall")
    age_ax.set_title("C  Held-out retention by memory age")
    age_ax.legend(frameon=False, fontsize=8)

    for rule_index, rule in enumerate(RULES):
        values = arrays["integrated_recall"][rule_index]
        jitter = np.linspace(-0.10, 0.10, len(values))
        integrated_ax.scatter(
            rule_index + jitter, values, color=RULE_COLORS[rule], alpha=0.7, s=24
        )
        ci = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
        integrated_ax.errorbar(
            rule_index, np.mean(values), yerr=ci, color="black", fmt="o", capsize=4
        )
    integrated_ax.set_xticks(
        range(len(RULES)), ["Target-\ngated", "Hebbian", "Delta"]
    )
    integrated_ax.set_ylabel("Retention-matrix mean")
    integrated_ax.set_title("D  Integrated sequential recall")

    for rule_index, rule in enumerate(RULES):
        values = arrays["immediate_recall"][rule_index]
        jitter = np.linspace(-0.10, 0.10, len(values))
        immediate_ax.scatter(
            rule_index + jitter, values, color=RULE_COLORS[rule], alpha=0.7, s=24
        )
        ci = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
        immediate_ax.errorbar(
            rule_index, np.mean(values), yerr=ci, color="black", fmt="o", capsize=4
        )
    immediate_ax.set_xticks(
        range(len(RULES)), ["Target-\ngated", "Hebbian", "Delta"]
    )
    immediate_ax.set_ylabel("Age-0 recall across all loads")
    immediate_ax.set_title("E  Ability to encode the next memory")

    loads = np.arange(1, arrays["high_weight_fraction"].shape[-1] + 1)
    for rule_index, rule in enumerate(RULES):
        values = arrays["high_weight_fraction"][rule_index]
        mean = np.mean(values, axis=0)
        ci = 1.96 * np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
        weights_ax.plot(loads, mean, color=RULE_COLORS[rule], label=SHORT_RULE_NAMES[rule])
        weights_ax.fill_between(loads, mean - ci, mean + ci, color=RULE_COLORS[rule], alpha=0.15)
    weights_ax.set_xlabel("Memories stored")
    weights_ax.set_ylabel("Fraction of weights ≥ 0.95")
    weights_ax.set_title("F  Hebbian accumulation saturates weights")
    weights_ax.legend(frameon=False, fontsize=8)

    figure.suptitle(
        "E8 — Alignment generalizes across rules; memory trade-offs do not",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.07, right=0.985, bottom=0.08, top=0.91, wspace=0.30, hspace=0.34
    )
    figure.savefig(output_path, dpi=220)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    info, autoencoder = load_autoencoder_session(
        args.checkpoint, map_location=args.device
    )
    params = session_parameters(info)
    params["ca3_active"] = args.ca3_active
    autoencoder.eval()

    learning_rates = np.asarray(args.learning_rates, dtype=float)
    development_seeds = [
        args.dev_seed_start + index for index in range(args.dev_num_seeds)
    ]
    evaluation_seeds = [
        args.eval_seed_start + index for index in range(args.num_seeds)
    ]
    tuning = development_sweep(
        autoencoder,
        params,
        seeds=development_seeds,
        num_patterns=args.tuning_num_patterns,
        learning_rates=learning_rates,
        capacity_threshold=args.capacity_threshold,
    )
    selected_rates, selection = select_learning_rates(
        tuning,
        learning_rates,
        reference_rate=args.reference_learning_rate,
        tolerance=args.immediate_recall_tolerance,
    )
    arrays, metadata = evaluate(
        autoencoder,
        params,
        seeds=evaluation_seeds,
        num_patterns=args.num_patterns,
        endpoint_patterns=args.endpoint_patterns,
        selected_rates=selected_rates,
        capacity_threshold=args.capacity_threshold,
    )

    alignment_summary = {}
    rule_summary = {}
    for rule_index, rule in enumerate(RULES):
        alignment_summary[rule] = {
            condition: scalar_summary(
                arrays["alignment_mean_chance_corrected"][rule_index, condition_index]
            )
            for condition_index, condition in enumerate(ALIGNMENT_CONDITIONS)
        }
        rule_summary[rule] = {
            "selected_learning_rate": selected_rates[rule],
            "immediate_recall": scalar_summary(arrays["immediate_recall"][rule_index]),
            "early_immediate_recall": scalar_summary(
                arrays["early_immediate_recall"][rule_index]
            ),
            "integrated_recall": scalar_summary(arrays["integrated_recall"][rule_index]),
            "load_capacity": scalar_summary(arrays["load_capacity"][rule_index]),
            "half_life": scalar_summary(arrays["half_life"][rule_index]),
            "final_high_weight_fraction": scalar_summary(
                arrays["high_weight_fraction"][rule_index, :, -1]
            ),
            "final_weight_norm": scalar_summary(
                arrays["weight_norm"][rule_index, :, -1]
            ),
        }

    fixed_index = ALIGNMENT_CONDITIONS.index("fixed_permutation")
    rescue_index = ALIGNMENT_CONDITIONS.index("decoder_rescue")
    aligned_index = ALIGNMENT_CONDITIONS.index("aligned")
    fixed_rescue_identical = bool(
        np.array_equal(
            arrays["alignment_final_weights"][:, fixed_index],
            arrays["alignment_final_weights"][:, rescue_index],
        )
    )
    alignment_checks = {
        rule: {
            "aligned_exceeds_fixed": bool(
                np.mean(
                    arrays["alignment_mean_chance_corrected"][rule_index, aligned_index]
                    - arrays["alignment_mean_chance_corrected"][rule_index, fixed_index]
                )
                > 0
            ),
            "decoder_rescues_fixed": bool(
                np.mean(
                    arrays["alignment_mean_chance_corrected"][rule_index, rescue_index]
                    - arrays["alignment_mean_chance_corrected"][rule_index, fixed_index]
                )
                > 0
            ),
            "rescue_matches_aligned": bool(
                np.max(
                    np.abs(
                        arrays["alignment_mean_chance_corrected"][rule_index, rescue_index]
                        - arrays["alignment_mean_chance_corrected"][rule_index, aligned_index]
                    )
                )
                < 1e-5
            ),
        }
        for rule_index, rule in enumerate(RULES)
    }
    protocol_checks = {
        "development_and_evaluation_seeds_disjoint": bool(
            set(development_seeds).isdisjoint(evaluation_seeds)
        ),
        "fixed_and_rescue_weights_identical_for_all_rules": fixed_rescue_identical,
        "all_weights_finite": bool(np.all(np.isfinite(arrays["final_weights"]))),
        "all_weights_bounded": bool(
            np.min(arrays["final_weights"]) >= -1e-7
            and np.max(arrays["final_weights"]) <= 1 + 1e-7
        ),
        "all_alignment_checks_pass": bool(
            all(all(checks.values()) for checks in alignment_checks.values())
        ),
    }

    target_index = RULES.index("target_gated")
    paired_effects = {}
    paired_effect_summary = {}
    for rule_index, rule in enumerate(RULES):
        if rule == "target_gated":
            continue
        paired_effects[f"target_gated_minus_{rule}"] = {
            "integrated_recall": arrays["integrated_recall"][target_index]
            - arrays["integrated_recall"][rule_index],
            "immediate_recall": arrays["immediate_recall"][target_index]
            - arrays["immediate_recall"][rule_index],
            "early_immediate_recall": arrays["early_immediate_recall"][target_index]
            - arrays["early_immediate_recall"][rule_index],
            "load_capacity": arrays["load_capacity"][target_index]
            - arrays["load_capacity"][rule_index],
            "half_life": arrays["half_life"][target_index]
            - arrays["half_life"][rule_index],
        }
        paired_effect_summary[f"target_gated_minus_{rule}"] = {
            metric: scalar_summary(values)
            for metric, values in paired_effects[
                f"target_gated_minus_{rule}"
            ].items()
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "e8_rule_baselines"
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        rules=np.asarray(RULES),
        alignment_conditions=np.asarray(ALIGNMENT_CONDITIONS),
        learning_rates=learning_rates,
        selected_learning_rates=np.asarray([selected_rates[rule] for rule in RULES]),
        development_seeds=np.asarray(development_seeds),
        evaluation_seeds=np.asarray(evaluation_seeds),
        tuning_early_immediate_recall=tuning["early_immediate_recall"],
        tuning_integrated_recall=tuning["integrated_recall"],
        tuning_load_capacity=tuning["load_capacity"],
        patterns=np.stack([item["patterns"] for item in metadata]),
        permutations=np.stack([item["permutation"] for item in metadata]),
        base_ca3_projection=np.stack(
            [item["base_ca3_projection"] for item in metadata]
        ),
        **arrays,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E8_rule_specific_baselines",
        "configuration": vars(args),
        "runtime": runtime_metadata(),
        "checkpoint_info": info,
        "network_parameters": params,
        "rules": {
            "target_gated": {
                "equation": "W <- (1 - eta*c) * W + eta*c*k^T",
                "information": "presynaptic CA3 activity and content target/IS",
                "depression": "target-row-specific",
            },
            "hebbian": {
                "equation": "W <- clip(W + eta*c*k^T, 0, 1)",
                "information": "presynaptic CA3 activity and content target/IS",
                "depression": "none",
            },
            "delta": {
                "equation": "W <- clip(W + eta*(c-Wk)*k^T/||k||^2, 0, 1)",
                "information": "presynaptic CA3 activity, content target, and prediction error",
                "depression": "error-dependent on active presynaptic synapses",
            },
        },
        "protocol": {
            "primary_endpoint": "mean chance-corrected recall over the full triangular retention matrix",
            "rate_selection": selection,
            "paired_controls": (
                "same seeds, memories, CA3 projections, storage order, content targets, "
                "initial weights, encoder, and decoder"
            ),
            "pattern_sha256_by_evaluation_seed": [
                array_digest(item["patterns"]) for item in metadata
            ],
        },
        "alignment_summary": alignment_summary,
        "rule_summary": rule_summary,
        "paired_effects": paired_effects,
        "paired_effect_summary": paired_effect_summary,
        "alignment_checks": alignment_checks,
        "protocol_checks": protocol_checks,
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
        tuning,
        learning_rates,
        selected_rates,
        arrays,
        args.capacity_threshold,
        prefix.with_suffix(".png"),
    )
    return report


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, _ = config_parser.parse_known_args()
    config = load_config(known.config)
    development = config["development"]
    evaluation = config["evaluation"]

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--checkpoint", default=config["checkpoint"])
    parser.add_argument("--ca3-active", type=int, default=config["ca3_active"])
    parser.add_argument("--capacity-threshold", type=float, default=config["capacity_threshold"])
    parser.add_argument("--dev-seed-start", type=int, default=development["seed_start"])
    parser.add_argument("--dev-num-seeds", type=int, default=development["num_seeds"])
    parser.add_argument("--tuning-num-patterns", type=int, default=development["num_patterns"])
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=development["learning_rates"]
    )
    parser.add_argument(
        "--reference-learning-rate",
        type=float,
        default=development["reference_learning_rate"],
    )
    parser.add_argument(
        "--immediate-recall-tolerance",
        type=float,
        default=development["immediate_recall_tolerance"],
    )
    parser.add_argument("--eval-seed-start", type=int, default=evaluation["seed_start"])
    parser.add_argument("--num-seeds", type=int, default=evaluation["num_seeds"])
    parser.add_argument("--num-patterns", type=int, default=evaluation["num_patterns"])
    parser.add_argument(
        "--endpoint-patterns",
        type=int,
        default=evaluation["alignment_endpoint_patterns"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.dev_num_seeds < 2 or args.num_seeds < 2:
        parser.error("development and evaluation require at least two seeds")
    if args.tuning_num_patterns < 2 or args.num_patterns < 2:
        parser.error("pattern counts must be at least two")
    if not 2 <= args.endpoint_patterns <= args.num_patterns:
        parser.error("--endpoint-patterns must lie between 2 and --num-patterns")
    if not 0 <= args.capacity_threshold <= 1:
        parser.error("--capacity-threshold must lie in [0, 1]")
    if not 0 <= args.immediate_recall_tolerance <= 1:
        parser.error("--immediate-recall-tolerance must lie in [0, 1]")
    if any(rate < 0 or rate > 1 for rate in args.learning_rates):
        parser.error("all learning rates must lie in [0, 1]")
    return args


def main() -> int:
    args = parse_args()
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    report = run(args)
    concise = {
        "selected_learning_rates": {
            rule: report["rule_summary"][rule]["selected_learning_rate"]
            for rule in RULES
        },
        "rule_summary": report["rule_summary"],
        "alignment_summary": report["alignment_summary"],
        "protocol_checks": report["protocol_checks"],
        "outputs": report["outputs"],
    }
    print(json.dumps(json_ready(concise), indent=2))
    return 0 if all(report["protocol_checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
