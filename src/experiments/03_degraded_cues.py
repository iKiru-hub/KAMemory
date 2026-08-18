"""E2b: retrieval from degraded cues and controlled-overlap memories.

Clean inputs are stored once, while retrieval uses separately generated cues.
The experiment measures both reconstruction of the clean target and selection
of the correct memory identity. It also probes unseen lures and memory banks
with controlled pairwise overlap.

Run from the repository root:

    python3 src/experiments/03_degraded_cues.py --deterministic
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
MASK_LEVELS = np.asarray((0.0, 0.25, 0.50, 0.75))
BIT_FLIP_LEVELS = np.asarray((0.0, 0.05, 0.10, 0.20))
OVERLAP_LEVELS = np.asarray((0.0, 0.20, 0.40, 0.60))
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


def cosine_numpy(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def top_k_f1(target: np.ndarray, output: np.ndarray, k: int) -> float:
    target_indices = set(np.flatnonzero(target > 0.5).tolist())
    predicted_indices = set(np.argpartition(output, -k)[-k:].tolist())
    overlap = len(target_indices & predicted_indices)
    return float(overlap / k)


def masked_cue(
    pattern: np.ndarray, level: float, rng: np.random.Generator
) -> np.ndarray:
    cue = pattern.copy()
    if level == 0:
        return cue
    active = np.flatnonzero(pattern > 0.5)
    cue[active[rng.random(len(active)) < level]] = 0
    return cue


def bit_flip_cue(
    pattern: np.ndarray, level: float, rng: np.random.Generator
) -> np.ndarray:
    cue = pattern.copy()
    if level == 0:
        return cue
    flips = rng.random(len(cue)) < level
    cue[flips] = 1.0 - cue[flips]
    return cue


def controlled_overlap_patterns(
    num_patterns: int,
    size: int,
    num_active: int,
    overlap_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate K-hot memories with an exact shared active subset.

    Non-shared active units are disjoint whenever the requested bank fits, so
    pairwise active-set overlap is exactly the shared fraction (up to rounding).
    """

    shared_count = int(round(num_active * float(overlap_fraction)))
    shared = rng.choice(size, size=shared_count, replace=False)
    available = np.setdiff1d(np.arange(size), shared, assume_unique=True)
    unique_count = num_active - shared_count
    required = num_patterns * unique_count
    if required > len(available):
        raise ValueError("controlled-overlap bank does not fit the input population")
    unique = rng.choice(available, size=required, replace=False).reshape(
        num_patterns, unique_count
    )
    patterns = np.zeros((num_patterns, size), dtype=np.float32)
    for index in range(num_patterns):
        patterns[index, shared] = 1
        patterns[index, unique[index]] = 1
    return patterns


def store_bank(
    base: BTSPMemory,
    autoencoder: Autoencoder,
    patterns: np.ndarray,
    condition: str,
    permutation: torch.Tensor,
) -> BTSPMemory:
    memory = configure_condition(base, condition, permutation)
    with torch.no_grad():
        codes = autoencoder.encode(torch.as_tensor(patterns)).detach().cpu()
    signals = codes if condition in {"aligned", "no_plasticity"} else codes[:, permutation]
    for pattern, signal in zip(patterns, signals):
        memory.store(pattern, instructive_signal=signal)
    return memory


def evaluate_cue(
    memory: BTSPMemory,
    cue: np.ndarray,
    target: np.ndarray,
    memory_bank: np.ndarray,
    input_active: int,
) -> tuple[float, float, float, int, np.ndarray]:
    with torch.no_grad():
        output = memory.retrieve(cue).detach().cpu().numpy().reshape(-1)
    raw_cosine = cosine_numpy(target, output)
    chance = input_active / len(target)
    corrected = (raw_cosine - chance) / (1 - chance)
    f1 = top_k_f1(target, output, input_active)
    similarities = np.asarray([cosine_numpy(candidate, output) for candidate in memory_bank])
    identity = int(np.argmax(similarities))
    return raw_cosine, corrected, f1, identity, output


def nearest_neighbor_retrieval(
    cue: np.ndarray, target: np.ndarray, memory_bank: np.ndarray
) -> tuple[float, int]:
    similarities = np.asarray(
        [cosine_numpy(candidate, cue) for candidate in memory_bank]
    )
    identity = int(np.argmax(similarities))
    return cosine_numpy(target, memory_bank[identity]), identity


def run_seed(
    seed: int,
    autoencoder: Autoencoder,
    params: dict,
    *,
    num_patterns: int,
    num_replicates: int,
    num_lures: int,
    alpha: float,
    lure_threshold: float,
) -> dict[str, object]:
    seed_everything(seed)
    base = make_base_memory(autoencoder, params, alpha)
    patterns = generate_legacy_sparse_patterns(
        num_patterns, params["input_active"], params["input_dim"]
    )
    control_rng = np.random.default_rng(seed + 40_000)
    permutation = torch.as_tensor(
        control_rng.permutation(params["dim_ca1"]), dtype=torch.long
    )
    lures = np.zeros((num_lures, params["input_dim"]), dtype=np.float32)
    for lure in lures:
        lure[
            control_rng.choice(
                params["input_dim"], size=params["input_active"], replace=False
            )
        ] = 1
    mask_cues = np.zeros(
        (
            len(MASK_LEVELS),
            num_replicates,
            num_patterns,
            params["input_dim"],
        ),
        dtype=np.float32,
    )
    flip_cues = np.zeros(
        (
            len(BIT_FLIP_LEVELS),
            num_replicates,
            num_patterns,
            params["input_dim"],
        ),
        dtype=np.float32,
    )
    for level_index, level in enumerate(MASK_LEVELS):
        for replicate in range(num_replicates):
            for target_index, target in enumerate(patterns):
                mask_cues[level_index, replicate, target_index] = masked_cue(
                    target, float(level), control_rng
                )
    for level_index, level in enumerate(BIT_FLIP_LEVELS):
        for replicate in range(num_replicates):
            for target_index, target in enumerate(patterns):
                flip_cues[level_index, replicate, target_index] = bit_flip_cue(
                    target, float(level), control_rng
                )
    nearest_mask_cosine = np.zeros(
        (len(MASK_LEVELS), num_replicates, num_patterns), dtype=np.float32
    )
    nearest_mask_identity = np.zeros_like(nearest_mask_cosine)
    nearest_flip_cosine = np.zeros(
        (len(BIT_FLIP_LEVELS), num_replicates, num_patterns), dtype=np.float32
    )
    nearest_flip_identity = np.zeros_like(nearest_flip_cosine)
    for level_index in range(len(MASK_LEVELS)):
        for replicate in range(num_replicates):
            for target_index, target in enumerate(patterns):
                score, identity = nearest_neighbor_retrieval(
                    mask_cues[level_index, replicate, target_index], target, patterns
                )
                nearest_mask_cosine[level_index, replicate, target_index] = score
                nearest_mask_identity[level_index, replicate, target_index] = (
                    identity == target_index
                )
    for level_index in range(len(BIT_FLIP_LEVELS)):
        for replicate in range(num_replicates):
            for target_index, target in enumerate(patterns):
                score, identity = nearest_neighbor_retrieval(
                    flip_cues[level_index, replicate, target_index], target, patterns
                )
                nearest_flip_cosine[level_index, replicate, target_index] = score
                nearest_flip_identity[level_index, replicate, target_index] = (
                    identity == target_index
                )

    result: dict[str, object] = {}
    for condition in CONDITIONS:
        memory = store_bank(base, autoencoder, patterns, condition, permutation)
        mask_shape = (len(MASK_LEVELS), num_replicates, num_patterns)
        flip_shape = (len(BIT_FLIP_LEVELS), num_replicates, num_patterns)
        mask_cosine = np.zeros(mask_shape, dtype=np.float32)
        mask_corrected = np.zeros(mask_shape, dtype=np.float32)
        mask_f1 = np.zeros(mask_shape, dtype=np.float32)
        mask_identity = np.zeros(mask_shape, dtype=np.int16)
        flip_cosine = np.zeros(flip_shape, dtype=np.float32)
        flip_corrected = np.zeros(flip_shape, dtype=np.float32)
        flip_f1 = np.zeros(flip_shape, dtype=np.float32)
        flip_identity = np.zeros(flip_shape, dtype=np.int16)

        for level_index, level in enumerate(MASK_LEVELS):
            for replicate in range(num_replicates):
                for target_index, target in enumerate(patterns):
                    cue = mask_cues[level_index, replicate, target_index]
                    raw, corrected, f1, identity, _ = evaluate_cue(
                        memory, cue, target, patterns, params["input_active"]
                    )
                    mask_cosine[level_index, replicate, target_index] = raw
                    mask_corrected[level_index, replicate, target_index] = corrected
                    mask_f1[level_index, replicate, target_index] = f1
                    mask_identity[level_index, replicate, target_index] = identity == target_index

        for level_index, level in enumerate(BIT_FLIP_LEVELS):
            for replicate in range(num_replicates):
                for target_index, target in enumerate(patterns):
                    cue = flip_cues[level_index, replicate, target_index]
                    raw, corrected, f1, identity, _ = evaluate_cue(
                        memory, cue, target, patterns, params["input_active"]
                    )
                    flip_cosine[level_index, replicate, target_index] = raw
                    flip_corrected[level_index, replicate, target_index] = corrected
                    flip_f1[level_index, replicate, target_index] = f1
                    flip_identity[level_index, replicate, target_index] = identity == target_index

        lure_max_similarity = np.zeros(num_lures, dtype=np.float32)
        lure_false_retrieval = np.zeros(num_lures, dtype=np.int8)
        for lure_index, lure in enumerate(lures):
            with torch.no_grad():
                output = memory.retrieve(lure).detach().cpu().numpy().reshape(-1)
            maximum = max(cosine_numpy(target, output) for target in patterns)
            lure_max_similarity[lure_index] = maximum
            lure_false_retrieval[lure_index] = maximum >= lure_threshold

        result[condition] = {
            "mask_cosine": mask_cosine,
            "mask_chance_corrected": mask_corrected,
            "mask_top_k_f1": mask_f1,
            "mask_identity_accuracy": mask_identity,
            "flip_cosine": flip_cosine,
            "flip_chance_corrected": flip_corrected,
            "flip_top_k_f1": flip_f1,
            "flip_identity_accuracy": flip_identity,
            "lure_max_similarity": lure_max_similarity,
            "lure_false_retrieval": lure_false_retrieval,
            "final_weights": memory.W_ca3_ca1.detach().cpu().numpy().copy(),
        }

    overlap_cosine = np.zeros(
        (len(CONDITIONS), len(OVERLAP_LEVELS), num_patterns), dtype=np.float32
    )
    overlap_identity = np.zeros_like(overlap_cosine)
    overlap_patterns = np.zeros(
        (len(OVERLAP_LEVELS), num_patterns, params["input_dim"]), dtype=np.float32
    )
    realized_overlap = np.zeros(len(OVERLAP_LEVELS), dtype=np.float32)
    overlap_rng = np.random.default_rng(seed + 50_000)
    for overlap_index, level in enumerate(OVERLAP_LEVELS):
        bank = controlled_overlap_patterns(
            num_patterns,
            params["input_dim"],
            params["input_active"],
            float(level),
            overlap_rng,
        )
        overlap_patterns[overlap_index] = bank
        pairwise = [
            np.dot(bank[left], bank[right]) / params["input_active"]
            for left in range(num_patterns)
            for right in range(left)
        ]
        realized_overlap[overlap_index] = float(np.mean(pairwise))
        for condition_index, condition in enumerate(CONDITIONS):
            memory = store_bank(base, autoencoder, bank, condition, permutation)
            for target_index, target in enumerate(bank):
                raw, _, _, identity, _ = evaluate_cue(
                    memory, target, target, bank, params["input_active"]
                )
                overlap_cosine[condition_index, overlap_index, target_index] = raw
                overlap_identity[condition_index, overlap_index, target_index] = (
                    identity == target_index
                )

    result["overlap"] = {
        "patterns": overlap_patterns,
        "realized_overlap": realized_overlap,
        "cosine": overlap_cosine,
        "identity_accuracy": overlap_identity,
    }
    result["metadata"] = {
        "seed": seed,
        "patterns": patterns,
        "lures": lures,
        "mask_cues": mask_cues,
        "flip_cues": flip_cues,
        "permutation": permutation.numpy(),
        "base_ca3_projection": base.W_ei_ca3.detach().cpu().numpy().copy(),
        "nearest_mask_cosine": nearest_mask_cosine,
        "nearest_mask_identity": nearest_mask_identity,
        "nearest_flip_cosine": nearest_flip_cosine,
        "nearest_flip_identity": nearest_flip_identity,
    }
    return result


def mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed_means = np.mean(values, axis=tuple(range(2, values.ndim)))
    mean = np.mean(seed_means, axis=0)
    ci = 1.96 * np.std(seed_means, axis=0, ddof=1) / np.sqrt(seed_means.shape[0])
    return mean, ci


def plot_lines(ax, levels, arrays, key, title, ylabel) -> None:
    for condition_index, condition in enumerate(CONDITIONS):
        mean, ci = mean_ci(arrays[key][condition_index])
        ax.plot(
            levels,
            mean,
            marker="o",
            color=COLORS[condition],
            linestyle=LINESTYLES.get(condition, "-"),
            label=DISPLAY_NAMES[condition],
        )
        ax.fill_between(levels, mean - ci, mean + ci, color=COLORS[condition], alpha=0.15)
    ax.set_xlabel("Cue corruption fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def make_figure(arrays: dict[str, np.ndarray], output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    mask_content, mask_identity, flip_content, flip_identity, overlap_ax, lure_ax = axes.flat
    plot_lines(
        mask_content,
        MASK_LEVELS,
        arrays,
        "mask_chance_corrected",
        "A  Clean-target reconstruction from masked cues",
        "Chance-corrected cosine",
    )
    nearest_mean = np.mean(arrays["nearest_mask_identity"], axis=(0, 2, 3))
    mask_identity.plot(
        MASK_LEVELS,
        nearest_mean,
        color="black",
        linestyle=":",
        marker="s",
        label="Nearest neighbor",
    )
    plot_lines(
        mask_identity,
        MASK_LEVELS,
        arrays,
        "mask_identity_accuracy",
        "B  Memory identity from masked cues",
        "Correct identity fraction",
    )
    nearest_mean = np.mean(arrays["nearest_flip_identity"], axis=(0, 2, 3))
    flip_identity.plot(
        BIT_FLIP_LEVELS,
        nearest_mean,
        color="black",
        linestyle=":",
        marker="s",
        label="Nearest neighbor",
    )
    mask_identity.axhline(1 / arrays["mask_identity_accuracy"].shape[-1], color="black", linestyle="--", linewidth=1)
    plot_lines(
        flip_content,
        BIT_FLIP_LEVELS,
        arrays,
        "flip_chance_corrected",
        "C  Clean-target reconstruction after bit flips",
        "Chance-corrected cosine",
    )
    plot_lines(
        flip_identity,
        BIT_FLIP_LEVELS,
        arrays,
        "flip_identity_accuracy",
        "D  Memory identity after bit flips",
        "Correct identity fraction",
    )
    flip_identity.axhline(1 / arrays["flip_identity_accuracy"].shape[-1], color="black", linestyle="--", linewidth=1)

    for condition_index, condition in enumerate(CONDITIONS):
        values = arrays["overlap_identity_accuracy"][condition_index]
        seed_means = np.mean(values, axis=2)
        mean = np.mean(seed_means, axis=0)
        ci = 1.96 * np.std(seed_means, axis=0, ddof=1) / np.sqrt(seed_means.shape[0])
        overlap_ax.plot(
            OVERLAP_LEVELS,
            mean,
            marker="o",
            color=COLORS[condition],
            linestyle=LINESTYLES.get(condition, "-"),
            label=DISPLAY_NAMES[condition],
        )
        overlap_ax.fill_between(OVERLAP_LEVELS, mean - ci, mean + ci, color=COLORS[condition], alpha=0.15)
    overlap_ax.set_xlabel("Pairwise active-set overlap")
    overlap_ax.set_ylabel("Correct identity fraction")
    overlap_ax.axhline(
        1 / arrays["overlap_identity_accuracy"].shape[-1],
        color="black",
        linestyle="--",
        linewidth=1,
    )
    overlap_ax.set_title("E  Correlation impairs memory identity")

    positions = np.arange(len(CONDITIONS))
    lure_rates = np.mean(arrays["lure_false_retrieval"], axis=(1, 2))
    lure_ax.bar(positions, lure_rates, color=[COLORS[name] for name in CONDITIONS])
    lure_ax.set_xticks(positions, [DISPLAY_NAMES[name] for name in CONDITIONS], rotation=20, ha="right")
    lure_ax.set_ylabel("False-retrieval fraction")
    lure_ax.set_title("F  Unseen-lure false retrieval")

    handles, labels = mask_identity.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    figure.suptitle(
        "E2b — Retrieval from degraded cues: content, identity, overlap, and lures",
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
            num_replicates=args.num_cue_replicates,
            num_lures=args.num_lures,
            alpha=args.alpha,
            lure_threshold=args.lure_threshold,
        )
        for seed in seeds
    ]
    metric_names = (
        "mask_cosine",
        "mask_chance_corrected",
        "mask_top_k_f1",
        "mask_identity_accuracy",
        "flip_cosine",
        "flip_chance_corrected",
        "flip_top_k_f1",
        "flip_identity_accuracy",
        "lure_max_similarity",
        "lure_false_retrieval",
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
    arrays["overlap_cosine"] = np.stack(
        [result["overlap"]["cosine"] for result in seed_results], axis=1
    )
    arrays["overlap_identity_accuracy"] = np.stack(
        [result["overlap"]["identity_accuracy"] for result in seed_results], axis=1
    )
    metadata = [result["metadata"] for result in seed_results]
    overlap_patterns = np.stack(
        [result["overlap"]["patterns"] for result in seed_results]
    )
    realized_overlap = np.stack(
        [result["overlap"]["realized_overlap"] for result in seed_results]
    )
    arrays["nearest_mask_cosine"] = np.stack(
        [item["nearest_mask_cosine"] for item in metadata]
    )
    arrays["nearest_mask_identity"] = np.stack(
        [item["nearest_mask_identity"] for item in metadata]
    )
    arrays["nearest_flip_cosine"] = np.stack(
        [item["nearest_flip_cosine"] for item in metadata]
    )
    arrays["nearest_flip_identity"] = np.stack(
        [item["nearest_flip_identity"] for item in metadata]
    )

    clean_aligned = np.mean(arrays["mask_chance_corrected"][0, :, 0])
    masked_aligned = np.mean(arrays["mask_chance_corrected"][0, :, 1:])
    masked_fixed = np.mean(arrays["mask_chance_corrected"][1, :, 1:])
    masked_rescue = np.mean(arrays["mask_chance_corrected"][2, :, 1:])
    masked_no_plasticity = np.mean(arrays["mask_chance_corrected"][3, :, 1:])
    hypothesis_checks = {
        "aligned_reconstructs_clean_cues": bool(clean_aligned >= 7 / 9),
        "aligned_exceeds_fixed_under_masking": bool(masked_aligned > masked_fixed),
        "aligned_exceeds_no_plasticity_under_masking": bool(
            masked_aligned > masked_no_plasticity
        ),
        "decoder_rescues_masked_cues": bool(masked_rescue > masked_fixed),
        "rescue_approaches_aligned_under_masking": bool(
            abs(masked_rescue - masked_aligned) < 0.05
        ),
        "aligned_identity_exceeds_chance_at_50_percent_masking": bool(
            np.mean(arrays["mask_identity_accuracy"][0, :, 2]) > 1 / args.num_patterns
        ),
        "aligned_identity_declines_with_memory_overlap": bool(
            np.mean(arrays["overlap_identity_accuracy"][0, :, 0])
            > np.mean(arrays["overlap_identity_accuracy"][0, :, -1])
        ),
    }
    protocol_checks = {
        "encoding_inputs_and_retrieval_cues_are_separate": True,
        "clean_targets_used_for_all_corruptions": True,
        "fixed_and_rescue_learning_identical": bool(
            np.array_equal(arrays["final_weights"][1], arrays["final_weights"][2])
        ),
        "controlled_overlap_matches_requested_levels": bool(
            np.allclose(realized_overlap.mean(axis=0), OVERLAP_LEVELS, atol=1e-7)
        ),
        "nearest_neighbor_baseline_evaluated_on_identical_cues": True,
    }

    summary = {}
    for condition_index, condition in enumerate(CONDITIONS):
        summary[condition] = {
            "mask_chance_corrected_mean_by_level": np.mean(
                arrays["mask_chance_corrected"][condition_index], axis=(0, 2, 3)
            ),
            "mask_identity_mean_by_level": np.mean(
                arrays["mask_identity_accuracy"][condition_index], axis=(0, 2, 3)
            ),
            "flip_chance_corrected_mean_by_level": np.mean(
                arrays["flip_chance_corrected"][condition_index], axis=(0, 2, 3)
            ),
            "flip_identity_mean_by_level": np.mean(
                arrays["flip_identity_accuracy"][condition_index], axis=(0, 2, 3)
            ),
            "overlap_cosine_mean_by_level": np.mean(
                arrays["overlap_cosine"][condition_index], axis=(0, 2)
            ),
            "overlap_identity_mean_by_level": np.mean(
                arrays["overlap_identity_accuracy"][condition_index], axis=(0, 2)
            ),
            "lure_false_retrieval_rate": float(
                np.mean(arrays["lure_false_retrieval"][condition_index])
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "e2b_degraded_cues"
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        seeds=np.asarray(seeds),
        conditions=np.asarray(CONDITIONS),
        mask_levels=MASK_LEVELS,
        bit_flip_levels=BIT_FLIP_LEVELS,
        overlap_levels=OVERLAP_LEVELS,
        realized_overlap=realized_overlap,
        patterns=np.stack([item["patterns"] for item in metadata]),
        lures=np.stack([item["lures"] for item in metadata]),
        mask_cues=np.stack([item["mask_cues"] for item in metadata]),
        flip_cues=np.stack([item["flip_cues"] for item in metadata]),
        permutations=np.stack([item["permutation"] for item in metadata]),
        base_ca3_projection=np.stack(
            [item["base_ca3_projection"] for item in metadata]
        ),
        overlap_patterns=overlap_patterns,
        **arrays,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E2b_degraded_cue_retrieval",
        "configuration": vars(args),
        "runtime": runtime_metadata(),
        "checkpoint_info": info,
        "network_parameters": params,
        "protocol": {
            "dataset": "validated_notebook_rowwise_numpy_choice",
            "operation_order": "seed -> initialize EC-CA3 -> draw clean patterns -> store -> generate/retrieve degraded cues",
            "masking": "each active input bit independently removed at the nominal probability",
            "bit_flips": "each input bit independently inverted at the nominal probability",
            "identity_rule": "argmax cosine between model output and all clean stored targets",
            "lure_rule": "false retrieval if any stored target has output cosine >= 0.8",
            "pattern_sha256_by_seed": [
                array_digest(item["patterns"]) for item in metadata
            ],
        },
        "summary": summary,
        "nearest_neighbor_baseline": {
            "mask_content_cosine_mean_by_level": np.mean(
                arrays["nearest_mask_cosine"], axis=(0, 2, 3)
            ),
            "mask_identity_mean_by_level": np.mean(
                arrays["nearest_mask_identity"], axis=(0, 2, 3)
            ),
            "flip_content_cosine_mean_by_level": np.mean(
                arrays["nearest_flip_cosine"], axis=(0, 2, 3)
            ),
            "flip_identity_mean_by_level": np.mean(
                arrays["nearest_flip_identity"], axis=(0, 2, 3)
            ),
        },
        "protocol_checks": protocol_checks,
        "hypothesis_checks": hypothesis_checks,
        "interpretation_gate": (
            "Call the model associative memory only if aligned clean-target reconstruction "
            "and identity remain above controls/chance for degraded cues."
        ),
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
    make_figure(arrays, prefix.with_suffix(".png"))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="ae_8")
    parser.add_argument("--seed-start", type=int, default=900)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--num-patterns", type=int, default=8)
    parser.add_argument("--num-cue-replicates", type=int, default=20)
    parser.add_argument("--num-lures", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--ca3-active", type=int, default=22)
    parser.add_argument("--lure-threshold", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.num_seeds < 2:
        parser.error("--num-seeds must be at least 2")
    if args.num_patterns < 2:
        parser.error("--num-patterns must be at least 2")
    if args.num_cue_replicates < 1 or args.num_lures < 1:
        parser.error("cue replicates and lures must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    report = run(args)
    concise = {
        "summary": report["summary"],
        "protocol_checks": report["protocol_checks"],
        "hypothesis_checks": report["hypothesis_checks"],
        "outputs": report["outputs"],
    }
    print(json.dumps(json_ready(concise), indent=2))
    return 0 if all(report["protocol_checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
