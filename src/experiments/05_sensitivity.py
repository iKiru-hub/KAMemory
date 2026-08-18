"""E4: focused robustness around the saved optimized memory configuration.

The reference is frozen in ``src/configs/optimized_memory.json`` from the
repository's explicit ``src/optim_wb/best_params.yaml`` artifact. This is a
one-factor-at-a-time robustness analysis, not a new parameter optimization.

Run from the repository root:

    python3 src/experiments/05_sensitivity.py --deterministic
"""

from __future__ import annotations

import argparse
import copy
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
import numpy as np
import torch

from kamemory.data import generate_legacy_sparse_patterns
from kamemory.io import PATHS, load_autoencoder_session, runtime_metadata
from kamemory.models import BTSPMemory
from kamemory.utils import array_digest, seed_everything


SCHEMA_VERSION = 1
CONDITIONS = ("aligned", "fixed_permutation", "decoder_rescue", "random_matched")
DISPLAY_NAMES = {
    "aligned": "Aligned IS",
    "fixed_permutation": "Fixed permutation",
    "decoder_rescue": "Permutation + rescue",
    "random_matched": "Random matched IS",
}
COLORS = {
    "aligned": "#2878B5",
    "fixed_permutation": "#D95F59",
    "decoder_rescue": "#2A9D8F",
    "random_matched": "#8C6BB1",
}
LINESTYLES = {"decoder_rescue": "--"}
DEFAULT_CONFIG = PATHS.configs / "optimized_memory.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
OPTIMIZER_ARTIFACT = PATHS.source / "optim_wb" / "best_params.yaml"

SWEEPS = {
    "alpha": np.asarray((0.08, 0.12, 0.2087942893137534, 0.35, 0.55)),
    "dim_ca3": np.asarray((25, 50, 100, 200), dtype=int),
    "ca3_active": np.asarray((6, 14, 22, 30, 38), dtype=int),
    "ca1_active": np.asarray((8, 13, 18, 23, 28), dtype=int),
}
LOADS = np.asarray((8, 16, 28, 40, 60), dtype=int)
ALIGNMENT_FRACTIONS = np.asarray((0.0, 0.1, 0.25, 0.5, 0.75, 1.0))


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


def load_reference(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        reference = json.load(handle)
    required = {"alpha", "beta", "ca1_active", "ca3_active", "dim_ca3"}
    if not required.issubset(reference["parameters"]):
        raise ValueError(f"optimized configuration is missing {required - reference['parameters'].keys()}")
    return reference


def checkpoint_parameters(info: dict) -> dict[str, int | float]:
    if "network_params" in info:
        params = info["network_params"]
        return {
            "input_dim": int(params["dim_ei"]),
            "input_active": int(params["K_ei"]),
            "output_active": int(params["K_eo"]),
            "dim_ca1": int(params["dim_ca1"]),
            "ca1_active": int(params["K_ca1"]),
            "beta": float(params["beta_ca1"]),
        }
    return {
        "input_dim": int(info["dim_ei"]),
        "input_active": int(info["K"]),
        "output_active": int(info["K"]),
        "dim_ca1": int(info["dim_ca1"]),
        "ca1_active": int(info["K_lat"]),
        "beta": float(info["beta"]),
    }


def validate_reference(reference: dict, checkpoint: dict) -> None:
    contract = reference["checkpoint_contract"]
    for key in ("input_dim", "input_active", "output_active", "dim_ca1"):
        if checkpoint[key] != contract[key]:
            raise ValueError(f"checkpoint violates optimized contract for {key}")
    if checkpoint["ca1_active"] != contract["checkpoint_ca1_active"]:
        raise ValueError("checkpoint CA1 sparsity does not match optimized artifact")
    if checkpoint["beta"] != contract["checkpoint_beta"]:
        raise ValueError("checkpoint beta does not match optimized artifact")


def derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    identity = np.arange(size)
    for _ in range(1_000):
        candidate = rng.permutation(size)
        if np.all(candidate != identity):
            return candidate
    return np.roll(identity, 1)


def partial_permutation(
    size: int, fraction: float, rng: np.random.Generator
) -> np.ndarray:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must lie in [0, 1]")
    permutation = np.arange(size)
    count = int(round(fraction * size))
    if count < 2:
        return permutation
    selected = rng.choice(size, size=count, replace=False)
    permutation[selected] = np.roll(selected, 1)
    return permutation


def top_k_f1(target: np.ndarray, output: np.ndarray, k: int) -> float:
    truth = set(np.flatnonzero(target > 0.5).tolist())
    prediction = set(np.argpartition(output, -k)[-k:].tolist())
    overlap = len(truth & prediction)
    precision = overlap / max(len(prediction), 1)
    recall = overlap / max(len(truth), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / max(denominator, 1e-12))


def make_memory(autoencoder, checkpoint: dict, parameters: dict) -> BTSPMemory:
    return BTSPMemory.from_autoencoder(
        autoencoder,
        K_lat=int(parameters["ca1_active"]),
        K_out=checkpoint["output_active"],
        dim_ca3=int(parameters["dim_ca3"]),
        K_ca3=int(parameters["ca3_active"]),
        beta=float(parameters["beta"]),
        alpha=float(parameters["alpha"]),
    )


def signal_bank(memory: BTSPMemory, patterns: np.ndarray) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack(
            [memory.instructive_signal(pattern).reshape(-1) for pattern in patterns]
        ).cpu()


def configure_conditions(
    base: BTSPMemory,
    signals: torch.Tensor,
    permutation: torch.Tensor,
    random_matching: np.ndarray,
) -> tuple[dict[str, BTSPMemory], dict[str, torch.Tensor]]:
    memories = {condition: copy.deepcopy(base) for condition in CONDITIONS}
    memories["decoder_rescue"].W_ca1_eo.copy_(base.W_ca1_eo[:, permutation])
    probe = torch.randn(base._dim_ca1, 1)
    if not torch.allclose(
        base.W_ca1_eo @ probe,
        memories["decoder_rescue"].W_ca1_eo @ probe[permutation],
        atol=1e-5,
        rtol=1e-5,
    ):
        raise AssertionError("rescue decoder has the wrong orientation")
    condition_signals = {
        "aligned": signals,
        "fixed_permutation": signals[:, permutation],
        "decoder_rescue": signals[:, permutation],
        "random_matched": signals[random_matching],
    }
    return memories, condition_signals


def evaluate_memory(
    memory: BTSPMemory, patterns: np.ndarray, checkpoint: dict
) -> tuple[float, float, float]:
    raw_cosine = []
    chance_corrected = []
    f1 = []
    chance = checkpoint["input_active"] / checkpoint["input_dim"]
    with torch.no_grad():
        for pattern in patterns:
            output = memory.retrieve(pattern).detach().cpu().numpy().reshape(-1)
            similarity = cosine(pattern, output)
            raw_cosine.append(similarity)
            chance_corrected.append((similarity - chance) / (1 - chance))
            f1.append(top_k_f1(pattern, output, checkpoint["output_active"]))
    return float(np.mean(chance_corrected)), float(np.mean(raw_cosine)), float(np.mean(f1))


def seed_material(seed: int, num_patterns: int, checkpoint: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed_everything(seed)
    patterns = generate_legacy_sparse_patterns(
        num_patterns, checkpoint["input_active"], checkpoint["input_dim"]
    )
    rng = np.random.default_rng(seed + 70_000)
    permutation = rng.permutation(checkpoint["dim_ca1"])
    matching = derangement(num_patterns, rng)
    return patterns, permutation, matching


def run_setting(
    seed: int,
    autoencoder,
    checkpoint: dict,
    parameters: dict,
    patterns: np.ndarray,
    permutation: np.ndarray,
    random_matching: np.ndarray,
) -> np.ndarray:
    if parameters["ca3_active"] > parameters["dim_ca3"]:
        raise ValueError("ca3_active cannot exceed dim_ca3")
    # Separate model RNG from data RNG. Equal CA3 sizes receive identical
    # projections across all one-factor sweeps for a paired seed.
    np.random.seed(seed + 80_000 + int(parameters["dim_ca3"]))
    torch.manual_seed(seed + 90_000)
    base = make_memory(autoencoder, checkpoint, parameters)
    signals = signal_bank(base, patterns)
    memories, condition_signals = configure_conditions(
        base,
        signals,
        torch.as_tensor(permutation, dtype=torch.long),
        random_matching,
    )
    for condition in CONDITIONS:
        with torch.no_grad():
            for pattern, signal in zip(patterns, condition_signals[condition]):
                memories[condition].store(pattern, instructive_signal=signal)
    return np.asarray(
        [evaluate_memory(memories[name], patterns, checkpoint) for name in CONDITIONS],
        dtype=np.float32,
    )


def run_load_curve(
    seed: int,
    autoencoder,
    checkpoint: dict,
    parameters: dict,
    patterns: np.ndarray,
    permutation: np.ndarray,
    random_matching: np.ndarray,
    loads: np.ndarray,
) -> np.ndarray:
    np.random.seed(seed + 80_000 + int(parameters["dim_ca3"]))
    torch.manual_seed(seed + 90_000)
    base = make_memory(autoencoder, checkpoint, parameters)
    signals = signal_bank(base, patterns)
    memories, condition_signals = configure_conditions(
        base, signals, torch.as_tensor(permutation, dtype=torch.long), random_matching
    )
    scores = np.full((len(loads), len(CONDITIONS), 3), np.nan, dtype=np.float32)
    load_to_index = {int(load): index for index, load in enumerate(loads)}
    for pattern_index, pattern in enumerate(patterns):
        for condition in CONDITIONS:
            memories[condition].store(
                pattern, instructive_signal=condition_signals[condition][pattern_index]
            )
        load = pattern_index + 1
        if load in load_to_index:
            scores[load_to_index[load]] = np.asarray(
                [
                    evaluate_memory(memories[name], patterns[:load], checkpoint)
                    for name in CONDITIONS
                ]
            )
    return scores


def run_alignment_curve(
    seed: int,
    autoencoder,
    checkpoint: dict,
    parameters: dict,
    patterns: np.ndarray,
    fractions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(seed + 80_000 + int(parameters["dim_ca3"]))
    torch.manual_seed(seed + 90_000)
    base = make_memory(autoencoder, checkpoint, parameters)
    signals = signal_bank(base, patterns)
    scores = np.empty((len(fractions), 3), dtype=np.float32)
    alignments = np.empty(len(fractions), dtype=np.float32)
    final_permutation = None
    for index, fraction in enumerate(fractions):
        rng = np.random.default_rng(seed + 100_000 + index)
        permutation = partial_permutation(checkpoint["dim_ca1"], float(fraction), rng)
        final_permutation = permutation
        permuted = signals[:, torch.as_tensor(permutation, dtype=torch.long)]
        memory = copy.deepcopy(base)
        for pattern, signal in zip(patterns, permuted):
            memory.store(pattern, instructive_signal=signal)
        scores[index] = evaluate_memory(memory, patterns, checkpoint)
        alignments[index] = np.mean(
            [cosine(left.numpy(), right.numpy()) for left, right in zip(signals, permuted)]
        )

    rescue = copy.deepcopy(base)
    permutation_tensor = torch.as_tensor(final_permutation, dtype=torch.long)
    rescue.W_ca1_eo.copy_(base.W_ca1_eo[:, permutation_tensor])
    permuted = signals[:, permutation_tensor]
    for pattern, signal in zip(patterns, permuted):
        rescue.store(pattern, instructive_signal=signal)
    rescue_score = evaluate_memory(rescue, patterns, checkpoint)[0]
    return scores, alignments, rescue_score


def mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    if values.shape[0] < 2:
        return mean, np.zeros_like(mean)
    ci = 1.96 * np.nanstd(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, ci


def plot_condition_sweep(
    axis,
    grid: np.ndarray,
    scores: np.ndarray,
    reference_value: float,
    title: str,
    xlabel: str,
) -> None:
    for condition_index, condition in enumerate(CONDITIONS):
        mean, ci = mean_ci(scores[:, :, condition_index, 0])
        axis.plot(
            grid,
            mean,
            marker="o",
            color=COLORS[condition],
            linestyle=LINESTYLES.get(condition, "-"),
            label=DISPLAY_NAMES[condition],
        )
        axis.fill_between(grid, mean - ci, mean + ci, color=COLORS[condition], alpha=0.13)
    axis.axvline(reference_value, color="black", linestyle=":", linewidth=1)
    axis.axhline(0, color="0.6", linestyle="--", linewidth=0.8)
    axis.set(title=title, xlabel=xlabel, ylabel="Chance-corrected cosine")


def make_figure(results: dict, reference: dict, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    ref = reference["parameters"]
    panels = (
        ("alpha", axes[0, 0], ref["alpha"], "A  Learning rate", r"$\alpha$"),
        ("load", axes[0, 1], 28, "B  Memory load", "Stored memories"),
        ("dim_ca3", axes[0, 2], ref["dim_ca3"], "C  CA3 population size", "CA3 units"),
        ("ca3_active", axes[1, 0], ref["ca3_active"], "D  CA3 sparsity", "Active CA3 units"),
        ("ca1_active", axes[1, 1], ref["ca1_active"], "E  CA1/IS sparsity", "Active CA1 units"),
    )
    for name, axis, value, title, xlabel in panels:
        plot_condition_sweep(
            axis, results[name]["grid"], results[name]["scores"], value, title, xlabel
        )

    axis = axes[1, 2]
    alignment_scores = results["misalignment"]["scores"][:, :, 0]
    mean, ci = mean_ci(alignment_scores)
    axis.plot(
        results["misalignment"]["grid"], mean, marker="o", color=COLORS["aligned"],
        label="Mismatched decoder",
    )
    axis.fill_between(
        results["misalignment"]["grid"], mean - ci, mean + ci,
        color=COLORS["aligned"], alpha=0.13,
    )
    rescue_mean, rescue_ci = mean_ci(results["misalignment"]["rescue"])
    axis.errorbar(
        [1.0], [rescue_mean], yerr=[rescue_ci], marker="s", color=COLORS["decoder_rescue"],
        linestyle="none", capsize=3, label="Matched decoder rescue",
    )
    axis.axhline(0, color="0.6", linestyle="--", linewidth=0.8)
    axis.set(
        title="F  Coordinate misalignment",
        xlabel="Fraction of CA1 coordinates permuted",
        ylabel="Chance-corrected cosine",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.015))
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    figure.suptitle(
        "E4 — Central alignment result is robust around the saved optimized configuration",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def summarize_sweep(scores: np.ndarray) -> dict:
    aligned = scores[:, :, CONDITIONS.index("aligned"), 0]
    fixed = scores[:, :, CONDITIONS.index("fixed_permutation"), 0]
    rescued = scores[:, :, CONDITIONS.index("decoder_rescue"), 0]
    random = scores[:, :, CONDITIONS.index("random_matched"), 0]
    return {
        "aligned_mean_by_value": aligned.mean(axis=0),
        "aligned_minus_fixed_by_value": (aligned - fixed).mean(axis=0),
        "aligned_minus_random_by_value": (aligned - random).mean(axis=0),
        "rescue_minus_aligned_by_value": (rescued - aligned).mean(axis=0),
        "fraction_of_seed_value_pairs_aligned_above_random": float(np.mean(aligned > random)),
        "fraction_of_seed_value_pairs_rescue_within_0.05_of_aligned": float(np.mean(np.abs(rescued - aligned) < 0.05)),
    }


def run(args: argparse.Namespace) -> dict:
    reference = load_reference(args.config)
    info, autoencoder = load_autoencoder_session(reference["checkpoint"], map_location=args.device)
    checkpoint = checkpoint_parameters(info)
    validate_reference(reference, checkpoint)
    autoencoder.eval()
    ref_params = dict(reference["parameters"])
    seeds = np.arange(args.seed_start, args.seed_start + args.num_seeds)
    results = {}

    max_patterns = int(LOADS.max())
    materials = {
        int(seed): seed_material(int(seed), max_patterns, checkpoint) for seed in seeds
    }
    for sweep_name, grid in SWEEPS.items():
        print(f"E4 sweep: {sweep_name}", flush=True)
        scores = np.empty((len(seeds), len(grid), len(CONDITIONS), 3), dtype=np.float32)
        for seed_index, seed in enumerate(seeds):
            patterns, permutation, matching = materials[int(seed)]
            short_matching = derangement(
                args.num_patterns, np.random.default_rng(int(seed) + 71_000)
            )
            for value_index, value in enumerate(grid):
                parameters = dict(ref_params)
                parameters[sweep_name] = value.item()
                scores[seed_index, value_index] = run_setting(
                    int(seed), autoencoder, checkpoint, parameters,
                    patterns[: args.num_patterns], permutation, short_matching,
                )
        results[sweep_name] = {"grid": grid, "scores": scores}

    print("E4 sweep: memory load", flush=True)
    load_scores = np.empty((len(seeds), len(LOADS), len(CONDITIONS), 3), dtype=np.float32)
    for seed_index, seed in enumerate(seeds):
        patterns, permutation, matching = materials[int(seed)]
        load_scores[seed_index] = run_load_curve(
            int(seed), autoencoder, checkpoint, ref_params,
            patterns, permutation, matching, LOADS,
        )
    results["load"] = {"grid": LOADS, "scores": load_scores}

    print("E4 sweep: coordinate misalignment", flush=True)
    alignment_scores = np.empty((len(seeds), len(ALIGNMENT_FRACTIONS), 3), dtype=np.float32)
    realized_alignment = np.empty((len(seeds), len(ALIGNMENT_FRACTIONS)), dtype=np.float32)
    rescue = np.empty(len(seeds), dtype=np.float32)
    for seed_index, seed in enumerate(seeds):
        patterns, _, _ = materials[int(seed)]
        alignment_scores[seed_index], realized_alignment[seed_index], rescue[seed_index] = run_alignment_curve(
            int(seed), autoencoder, checkpoint, ref_params,
            patterns[: args.num_patterns], ALIGNMENT_FRACTIONS,
        )
    results["misalignment"] = {
        "grid": ALIGNMENT_FRACTIONS,
        "scores": alignment_scores,
        "realized_alignment": realized_alignment,
        "rescue": rescue,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "e4_sensitivity"
    npz_values = {"seeds": seeds, "conditions": np.asarray(CONDITIONS)}
    for name, data in results.items():
        npz_values[f"{name}_grid"] = data["grid"]
        npz_values[f"{name}_scores"] = data["scores"]
        if name == "misalignment":
            npz_values["misalignment_realized_alignment"] = data["realized_alignment"]
            npz_values["misalignment_rescue"] = data["rescue"]
    np.savez_compressed(prefix.with_suffix(".npz"), **npz_values)
    make_figure(results, reference, prefix.with_suffix(".png"))

    summaries = {
        name: summarize_sweep(data["scores"])
        for name, data in results.items()
        if name != "misalignment"
    }
    alignment_mean = alignment_scores[:, :, 0].mean(axis=0)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E4_focused_robustness",
        "interpretation": "one-factor-at-a-time robustness, not re-optimization",
        "configuration": vars(args),
        "reference": reference,
        "checkpoint_info": info,
        "checkpoint_parameters": checkpoint,
        "optimizer_artifact": {
            "path": str(OPTIMIZER_ARTIFACT),
            "sha256": hashlib.sha256(OPTIMIZER_ARTIFACT.read_bytes()).hexdigest(),
            "selection_note": "explicit best_params artifact; score is not stored locally",
        },
        "protocol": {
            "paired_seeds": seeds,
            "endpoint_load_for_parameter_sweeps": args.num_patterns,
            "data": "same row-wise K-hot patterns for every setting within seed",
            "model_rng": "separate from data RNG; identical CA3 sizes share projections within seed",
            "metrics": ["chance_corrected_cosine", "raw_cosine", "top_k_f1"],
            "uncertainty": "95% normal-approximation CI across independent seeds",
            "pattern_sha256_by_seed": [array_digest(materials[int(seed)][0]) for seed in seeds],
        },
        "sweeps": {
            name: {"grid": data["grid"], "summary": summaries[name]}
            for name, data in results.items()
            if name != "misalignment"
        },
        "misalignment": {
            "fraction_permuted": ALIGNMENT_FRACTIONS,
            "realized_code_alignment_mean": realized_alignment.mean(axis=0),
            "chance_corrected_decodability_mean": alignment_mean,
            "full_permutation_rescue_mean": float(rescue.mean()),
            "full_permutation_rescue_minus_mismatch": float(np.mean(rescue - alignment_scores[:, -1, 0])),
        },
        "robustness_checks": {
            "aligned_above_random_for_every_tested_sweep_mean": bool(
                all(np.all(np.asarray(summary["aligned_minus_random_by_value"]) > 0) for summary in summaries.values())
            ),
            "rescue_within_0.05_of_aligned_for_every_tested_sweep_mean": bool(
                all(np.all(np.abs(np.asarray(summary["rescue_minus_aligned_by_value"])) < 0.05) for summary in summaries.values())
            ),
            "decodability_decreases_with_misalignment": bool(alignment_mean[-1] < alignment_mean[0]),
            "full_permutation_decoder_rescue": bool(np.mean(rescue - alignment_scores[:, -1, 0]) > 0),
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed-start", type=int, default=1100)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--num-patterns", type=int, default=28)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.num_seeds < 2:
        parser.error("--num-seeds must be at least 2 for uncertainty estimates")
    if not 2 <= args.num_patterns <= int(LOADS.max()):
        parser.error(f"--num-patterns must lie in [2, {int(LOADS.max())}]")
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    report = run(args)
    print(json.dumps(json_ready({
        "reference": report["reference"]["parameters"],
        "robustness_checks": report["robustness_checks"],
        "misalignment": report["misalignment"],
        "figure": report["outputs"]["figure_png"],
    }), indent=2))


if __name__ == "__main__":
    main()
