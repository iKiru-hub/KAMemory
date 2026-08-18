"""E1: causal test of instructive-signal/decoder coordinate alignment.

The five paired conditions share memories, CA3 projections, initial weights,
and seeds. The script also replays the saved Figure 2 notebook protocol to
verify numerical backward compatibility before presenting the corrected
fixed-permutation and matched-decoder-rescue controls.

Run from the repository root:

    python3 src/experiments/01_alignment.py --deterministic
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
from kamemory.io import PATHS, load_autoencoder_session, runtime_metadata
from kamemory.models import Autoencoder, BTSPMemory
from kamemory.utils import array_digest, cosine_similarity, seed_everything


SCHEMA_VERSION = 1
CONDITIONS = (
    "aligned",
    "fixed_permutation",
    "decoder_rescue",
    "random_matched",
    "no_plasticity",
)
DISPLAY_NAMES = {
    "aligned": "Aligned IS",
    "fixed_permutation": "Fixed permutation",
    "decoder_rescue": "Permutation +\ndecoder rescue",
    "random_matched": "Random matched IS",
    "no_plasticity": "No plasticity",
}
COLORS = {
    "aligned": "#2878B5",
    "fixed_permutation": "#D95F59",
    "decoder_rescue": "#2A9D8F",
    "random_matched": "#8C6BB1",
    "no_plasticity": "#7A7A7A",
}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
LEGACY_RESULTS = PATHS.root / "media" / "generated" / "figure_2" / "figure_2_experiments.npz"


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
    autoencoder: Autoencoder, params: dict, alpha: float
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


def derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    if size < 2:
        raise ValueError("a matched random code requires at least two memories")
    identity = np.arange(size)
    for _ in range(1_000):
        candidate = rng.permutation(size)
        if np.all(candidate != identity):
            return candidate
    # Deterministic fallback that is always a derangement.
    return np.roll(identity, 1)


def partial_permutation(
    size: int, fraction: float, rng: np.random.Generator
) -> np.ndarray:
    permutation = np.arange(size)
    count = int(round(float(fraction) * size))
    if count < 2:
        return permutation
    selected = rng.choice(size, size=count, replace=False)
    permutation[selected] = np.roll(selected, 1)
    return permutation


def top_k_f1(target: np.ndarray, output: np.ndarray, k: int) -> float:
    target_indices = set(np.flatnonzero(target > 0.5).tolist())
    predicted_indices = set(np.argpartition(output, -k)[-k:].tolist())
    true_positive = len(target_indices & predicted_indices)
    precision = true_positive / max(len(predicted_indices), 1)
    recall = true_positive / max(len(target_indices), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def code_bank(autoencoder: Autoencoder, patterns: np.ndarray) -> torch.Tensor:
    device = next(autoencoder.parameters()).device
    with torch.no_grad():
        return autoencoder.encode(torch.as_tensor(patterns, device=device)).detach().cpu()


def configure_condition(
    base_memory: BTSPMemory,
    condition: str,
    permutation: torch.Tensor,
) -> BTSPMemory:
    memory = copy.deepcopy(base_memory)
    if condition == "decoder_rescue":
        # If c' = P c, D' = D P^T. For the index convention c'=c[perm],
        # this is implemented by permuting decoder columns in the same order.
        memory.W_ca1_eo.copy_(base_memory.W_ca1_eo[:, permutation])
        probe = torch.randn(base_memory._dim_ca1, 1)
        original = base_memory.W_ca1_eo @ probe
        rescued = memory.W_ca1_eo @ probe[permutation]
        if not torch.allclose(original, rescued, atol=1e-5, rtol=1e-5):
            raise AssertionError("matched decoder has the wrong permutation orientation")
    return memory


def signals_for_condition(
    codes: torch.Tensor,
    condition: str,
    permutation: torch.Tensor,
    random_matching: np.ndarray,
) -> torch.Tensor | None:
    if condition == "aligned":
        return codes
    if condition in {"fixed_permutation", "decoder_rescue"}:
        return codes[:, permutation]
    if condition == "random_matched":
        return codes[random_matching]
    if condition == "no_plasticity":
        return None
    raise ValueError(f"unknown condition: {condition}")


def evaluate_condition(
    memory: BTSPMemory,
    patterns: np.ndarray,
    target_codes: torch.Tensor,
    signals: torch.Tensor | None,
    params: dict,
) -> dict[str, np.ndarray | float]:
    device = memory.W_ca3_ca1.device
    if signals is not None:
        for pattern, signal in zip(patterns, signals):
            memory.store(
                torch.as_tensor(pattern, device=device),
                instructive_signal=signal.to(device),
            )

    cosine = []
    chance_corrected = []
    f1 = []
    mse = []
    ca1_similarity = []
    output_sparsity = []
    output_norm = []
    outputs = []
    ca1_codes = []
    chance = params["input_active"] / params["input_dim"]

    with torch.no_grad():
        for pattern, target_code in zip(patterns, target_codes):
            x = torch.as_tensor(pattern, device=device).reshape(-1, 1)
            output, ca1 = memory.retrieve(x, return_ca1=True)
            raw_cosine = float(cosine_similarity(x, output).item())
            cosine.append(raw_cosine)
            chance_corrected.append((raw_cosine - chance) / (1 - chance))
            output_np = output.detach().cpu().numpy().reshape(-1)
            f1.append(top_k_f1(pattern, output_np, params["input_active"]))
            mse.append(float(torch.mean((output - x) ** 2).item()))
            ca1_similarity.append(
                float(
                    cosine_similarity(
                        ca1.detach().cpu(), target_code.reshape(-1, 1)
                    ).item()
                )
            )
            output_sparsity.append(float((output > 0.5).float().mean().item()))
            output_norm.append(float(torch.linalg.vector_norm(output).item()))
            outputs.append(output_np)
            ca1_codes.append(ca1.detach().cpu().numpy().reshape(-1))

    signal_sparsity = np.nan
    signal_norm = np.nan
    if signals is not None:
        signal_sparsity = float((signals > 0.5).float().mean().item())
        signal_norm = float(torch.linalg.vector_norm(signals, dim=1).mean().item())
    return {
        "cosine": np.asarray(cosine),
        "chance_corrected": np.asarray(chance_corrected),
        "top_k_f1": np.asarray(f1),
        "mse": np.asarray(mse),
        "ca1_similarity": np.asarray(ca1_similarity),
        "output_sparsity": np.asarray(output_sparsity),
        "output_norm": np.asarray(output_norm),
        "signal_sparsity": signal_sparsity,
        "signal_norm": signal_norm,
        "outputs": np.asarray(outputs),
        "ca1_codes": np.asarray(ca1_codes),
        "final_weights": memory.W_ca3_ca1.detach().cpu().numpy().copy(),
    }


def run_seed(
    seed: int,
    autoencoder: Autoencoder,
    params: dict,
    *,
    num_patterns: int,
    alpha: float,
) -> dict[str, dict]:
    seed_everything(seed)
    # Preserve the validated notebook order: initialize EC→CA3 before drawing
    # the row-by-row sparse memory set from NumPy's global RNG.
    base_memory = make_base_memory(autoencoder, params, alpha)
    patterns = generate_legacy_sparse_patterns(
        num_patterns,
        params["input_active"],
        params["input_dim"],
    )
    control_rng = np.random.default_rng(seed + 10_000)
    codes = code_bank(autoencoder, patterns)
    permutation = torch.as_tensor(
        control_rng.permutation(params["dim_ca1"]), dtype=torch.long
    )
    matching = derangement(num_patterns, control_rng)

    results = {}
    for condition in CONDITIONS:
        memory = configure_condition(base_memory, condition, permutation)
        signals = signals_for_condition(codes, condition, permutation, matching)
        results[condition] = evaluate_condition(
            memory, patterns, codes, signals, params
        )
    results["metadata"] = {
        "seed": seed,
        "patterns": patterns,
        "permutation": permutation.numpy(),
        "random_matching": matching,
        "base_ca3_projection": base_memory.W_ei_ca3.detach().cpu().numpy().copy(),
    }
    return results


def run_alignment_sweep(
    seeds: list[int],
    autoencoder: Autoencoder,
    params: dict,
    *,
    num_patterns: int,
    alpha: float,
    fractions: np.ndarray,
) -> dict[str, np.ndarray]:
    records = {"seed": [], "fraction": [], "alignment": [], "decodability": []}
    for seed in seeds:
        seed_everything(seed)
        base = make_base_memory(autoencoder, params, alpha)
        patterns = generate_legacy_sparse_patterns(
            num_patterns,
            params["input_active"],
            params["input_dim"],
        )
        control_rng = np.random.default_rng(seed + 20_000)
        codes = code_bank(autoencoder, patterns)
        for fraction in fractions:
            permutation = torch.as_tensor(
                partial_permutation(
                    params["dim_ca1"], float(fraction), control_rng
                ),
                dtype=torch.long,
            )
            signals = codes[:, permutation]
            memory = copy.deepcopy(base)
            result = evaluate_condition(memory, patterns, codes, signals, params)
            alignment = torch.stack(
                [cosine_similarity(left, right) for left, right in zip(codes, signals)]
            ).mean()
            records["seed"].append(seed)
            records["fraction"].append(float(fraction))
            records["alignment"].append(float(alignment.item()))
            records["decodability"].append(float(np.mean(result["cosine"])))
    return {key: np.asarray(value) for key, value in records.items()}


def replay_legacy_condition(
    seed: int,
    autoencoder: Autoencoder,
    params: dict,
    *,
    shuffled: bool,
    num_patterns: int = 28,
    alpha: float = 0.35,
) -> float:
    """Replay the original notebook, including its weight-only row shuffle."""

    seed_everything(seed)
    memory = make_base_memory(autoencoder, params, alpha)
    if shuffled:
        num_rows = memory.W_ei_ca1.shape[0]
        rows = torch.randperm(num_rows)[:num_rows]
        permuted = rows[torch.randperm(len(rows))]
        memory.W_ei_ca1[rows] = memory.W_ei_ca1[permuted].clone()
        # The notebook did not permute encoder bias. This is retained only for
        # backward validation, not as an E1 scientific control.
    patterns = generate_legacy_sparse_patterns(
        num_patterns, params["input_active"], params["input_dim"]
    )
    for pattern in patterns:
        memory.store(pattern)
    return float(
        np.mean(
            [
                cosine_similarity(
                    torch.as_tensor(pattern).reshape(-1, 1), memory.retrieve(pattern)
                ).item()
                for pattern in patterns
            ]
        )
    )


def verify_legacy_notebook(
    autoencoder: Autoencoder, params: dict, saved_path: Path = LEGACY_RESULTS
) -> dict[str, object]:
    if not saved_path.exists():
        return {"available": False, "passed": None, "saved_path": str(saved_path)}
    saved = np.load(saved_path)
    saved_conditions = saved["comparison_condition"]
    saved_values = saved["comparison_decodability"]
    reproduced = []
    for seed in range(100, 112):
        reproduced.extend(
            [
                replay_legacy_condition(seed, autoencoder, params, shuffled=False),
                replay_legacy_condition(seed, autoencoder, params, shuffled=True),
            ]
        )
    reproduced = np.asarray(reproduced)
    maximum_error = float(np.max(np.abs(saved_values - reproduced)))
    aligned_mask = saved_conditions == "content-aware IS"
    shuffled_mask = saved_conditions == "shuffled IS"
    return {
        "available": True,
        "passed": maximum_error < 1e-7,
        "saved_path": str(saved_path),
        "maximum_absolute_error": maximum_error,
        "saved_values": saved_values,
        "reproduced_values": reproduced,
        "saved_aligned_mean": float(saved_values[aligned_mask].mean()),
        "saved_shuffled_mean": float(saved_values[shuffled_mask].mean()),
        "reproduced_aligned_mean": float(reproduced[0::2].mean()),
        "reproduced_shuffled_mean": float(reproduced[1::2].mean()),
    }


def aggregate(seed_results: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    metrics = (
        "cosine",
        "chance_corrected",
        "top_k_f1",
        "mse",
        "ca1_similarity",
        "output_sparsity",
        "output_norm",
        "signal_sparsity",
        "signal_norm",
    )
    combined = {}
    for condition in CONDITIONS:
        combined[condition] = {}
        for metric in metrics:
            values = [result[condition][metric] for result in seed_results]
            array = np.asarray(values)
            combined[condition][metric] = array
            combined[condition][f"mean_{metric}"] = (
                np.nanmean(array, axis=1) if array.ndim > 1 else array
            )
    return combined


def summarize(aggregated: dict) -> dict[str, dict[str, dict[str, float]]]:
    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        for metric in ("cosine", "chance_corrected", "top_k_f1", "mse", "ca1_similarity"):
            values = aggregated[condition][f"mean_{metric}"]
            finite = values[np.isfinite(values)]
            sem = float(np.std(finite, ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else 0.0
            summary[condition][metric] = {
                "mean": float(np.mean(finite)),
                "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                "ci95_low": float(np.mean(finite) - 1.96 * sem),
                "ci95_high": float(np.mean(finite) + 1.96 * sem),
            }
    return summary


def paired_jitter(count: int) -> np.ndarray:
    if count == 1:
        return np.zeros(1)
    return np.linspace(-0.10, 0.10, count)


def plot_condition_metric(ax, aggregated, metric: str, title: str, ylabel: str) -> None:
    for index, condition in enumerate(CONDITIONS):
        values = aggregated[condition][f"mean_{metric}"]
        jitter = paired_jitter(len(values))
        ax.scatter(
            index + jitter,
            values,
            s=24,
            alpha=0.70,
            color=COLORS[condition],
            edgecolors="none",
        )
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        ax.errorbar(index, mean, yerr=1.96 * sem, fmt="o", color="black", capsize=4, zorder=4)
    ax.set_xticks(range(len(CONDITIONS)), [DISPLAY_NAMES[name] for name in CONDITIONS], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def make_figure(
    aggregated: dict,
    sweep: dict[str, np.ndarray],
    parity: dict[str, object],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    ax_cosine, ax_corrected, ax_rescue, ax_sweep, ax_f1, ax_parity = axes.flat

    plot_condition_metric(ax_cosine, aggregated, "cosine", "A  Final content decodability", "Cosine similarity")
    plot_condition_metric(
        ax_corrected,
        aggregated,
        "chance_corrected",
        "B  Chance-corrected endpoint",
        "Chance-corrected cosine",
    )
    ax_corrected.axhline(0, color="black", linewidth=1, linestyle="--")

    fixed = aggregated["fixed_permutation"]["mean_cosine"]
    rescued = aggregated["decoder_rescue"]["mean_cosine"]
    for left, right in zip(fixed, rescued):
        ax_rescue.plot([0, 1], [left, right], color="#777777", alpha=0.45, linewidth=1)
    ax_rescue.scatter(np.zeros_like(fixed), fixed, color=COLORS["fixed_permutation"], s=32)
    ax_rescue.scatter(np.ones_like(rescued), rescued, color=COLORS["decoder_rescue"], s=32)
    ax_rescue.set_xticks([0, 1], ["Fixed\npermutation", "Matched\ndecoder"])
    ax_rescue.set_ylabel("Cosine similarity")
    ax_rescue.set_title("C  Decoder rescue within seed")

    scatter = ax_sweep.scatter(
        sweep["alignment"],
        sweep["decodability"],
        c=sweep["fraction"],
        cmap="viridis_r",
        s=30,
        alpha=0.8,
        edgecolors="none",
    )
    coefficients = np.polyfit(sweep["alignment"], sweep["decodability"], deg=1)
    x_line = np.linspace(sweep["alignment"].min(), sweep["alignment"].max(), 100)
    ax_sweep.plot(x_line, np.polyval(coefficients, x_line), color="black", linestyle="--")
    correlation = float(np.corrcoef(sweep["alignment"], sweep["decodability"])[0, 1])
    ax_sweep.text(0.04, 0.94, f"r = {correlation:.3f}", transform=ax_sweep.transAxes, va="top")
    ax_sweep.set_xlabel("Target/IS code alignment")
    ax_sweep.set_ylabel("Cosine similarity")
    ax_sweep.set_title("D  Decodability tracks alignment")
    colorbar = figure.colorbar(scatter, ax=ax_sweep, fraction=0.046, pad=0.04)
    colorbar.set_label("Permuted fraction")

    plot_condition_metric(ax_f1, aggregated, "top_k_f1", "E  Active-content recovery", "Top-K F1")
    chance = 0.1
    ax_f1.axhline(chance, color="black", linewidth=1, linestyle="--", label="chance")
    ax_f1.legend(frameon=False)

    if parity.get("available"):
        saved = np.asarray(parity["saved_values"])
        reproduced = np.asarray(parity["reproduced_values"])
        ax_parity.scatter(saved[0::2], reproduced[0::2], label="aligned", color=COLORS["aligned"], s=38)
        ax_parity.scatter(saved[1::2], reproduced[1::2], label="legacy shuffled", color=COLORS["fixed_permutation"], s=38)
        lower = min(saved.min(), reproduced.min())
        upper = max(saved.max(), reproduced.max())
        ax_parity.plot([lower, upper], [lower, upper], color="black", linestyle="--")
        ax_parity.text(
            0.04,
            0.95,
            f"max |Δ| = {parity['maximum_absolute_error']:.2e}",
            transform=ax_parity.transAxes,
            va="top",
        )
        ax_parity.set_xlabel("Saved notebook value")
        ax_parity.set_ylabel("Refactored backend value")
        ax_parity.legend(frameon=False)
    else:
        ax_parity.text(0.5, 0.5, "Saved notebook data unavailable", ha="center", va="center")
    ax_parity.set_title("F  Backward numerical parity")

    figure.suptitle("E1 — Instructive-signal alignment causally controls stable readout", fontsize=16, fontweight="bold")
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    info, autoencoder = load_autoencoder_session(args.checkpoint, map_location=args.device)
    params = session_parameters(info)
    if args.ca3_active is not None:
        params["ca3_active"] = args.ca3_active
    autoencoder.eval()

    parity = (
        {"available": False, "passed": None, "skipped": True}
        if args.skip_legacy_parity
        else verify_legacy_notebook(autoencoder.cpu(), params)
    )
    autoencoder.to(args.device)

    seeds = [args.seed_start + index for index in range(args.num_seeds)]
    seed_results = [
        run_seed(
            seed,
            autoencoder,
            params,
            num_patterns=args.num_patterns,
            alpha=args.alpha,
        )
        for seed in seeds
    ]
    aggregated = aggregate(seed_results)
    summary = summarize(aggregated)
    fractions = np.linspace(0, 1, args.num_alignment_levels)
    sweep = run_alignment_sweep(
        seeds[: min(args.sweep_seeds, len(seeds))],
        autoencoder,
        params,
        num_patterns=args.num_patterns,
        alpha=args.alpha,
        fractions=fractions,
    )

    aligned = aggregated["aligned"]["mean_cosine"]
    fixed = aggregated["fixed_permutation"]["mean_cosine"]
    rescued = aggregated["decoder_rescue"]["mean_cosine"]
    random_matched = aggregated["random_matched"]["mean_cosine"]
    no_plasticity = aggregated["no_plasticity"]["mean_cosine"]
    signal_conditions = ("aligned", "fixed_permutation", "decoder_rescue", "random_matched")
    reference_sparsity = aggregated["aligned"]["signal_sparsity"]
    reference_norm = aggregated["aligned"]["signal_norm"]
    signal_statistics_matched = all(
        np.allclose(aggregated[name]["signal_sparsity"], reference_sparsity, atol=1e-7)
        and np.allclose(aggregated[name]["signal_norm"], reference_norm, atol=1e-7)
        for name in signal_conditions
    )
    fixed_rescue_weights_identical = all(
        np.array_equal(
            result["fixed_permutation"]["final_weights"],
            result["decoder_rescue"]["final_weights"],
        )
        for result in seed_results
    )
    causal_checks = {
        "aligned_exceeds_fixed_permutation": bool(np.mean(aligned - fixed) > 0),
        "aligned_exceeds_random_matched": bool(np.mean(aligned - random_matched) > 0),
        "aligned_exceeds_no_plasticity": bool(np.mean(aligned - no_plasticity) > 0),
        "decoder_rescues_fixed_permutation": bool(np.mean(rescued - fixed) > 0),
        "rescue_approaches_aligned": bool(abs(np.mean(rescued) - np.mean(aligned)) < 0.05),
        "matched_signal_statistics": bool(signal_statistics_matched),
        "fixed_and_rescue_learning_identical": bool(fixed_rescue_weights_identical),
        "legacy_notebook_parity": bool(parity.get("passed")) if parity.get("available") else None,
    }
    correlation = float(np.corrcoef(sweep["alignment"], sweep["decodability"])[0, 1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "e1_alignment"
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        seeds=np.asarray(seeds),
        conditions=np.asarray(CONDITIONS),
        cosine=np.stack([aggregated[name]["cosine"] for name in CONDITIONS]),
        chance_corrected=np.stack([aggregated[name]["chance_corrected"] for name in CONDITIONS]),
        top_k_f1=np.stack([aggregated[name]["top_k_f1"] for name in CONDITIONS]),
        mse=np.stack([aggregated[name]["mse"] for name in CONDITIONS]),
        ca1_similarity=np.stack([aggregated[name]["ca1_similarity"] for name in CONDITIONS]),
        patterns=np.stack([result["metadata"]["patterns"] for result in seed_results]),
        permutations=np.stack([result["metadata"]["permutation"] for result in seed_results]),
        random_matchings=np.stack([result["metadata"]["random_matching"] for result in seed_results]),
        base_ca3_projection=np.stack([result["metadata"]["base_ca3_projection"] for result in seed_results]),
        outputs=np.stack(
            [
                np.stack([result[name]["outputs"] for result in seed_results])
                for name in CONDITIONS
            ]
        ),
        ca1_codes=np.stack(
            [
                np.stack([result[name]["ca1_codes"] for result in seed_results])
                for name in CONDITIONS
            ]
        ),
        final_weights=np.stack(
            [
                np.stack([result[name]["final_weights"] for result in seed_results])
                for name in CONDITIONS
            ]
        ),
        signal_sparsity=np.stack([aggregated[name]["signal_sparsity"] for name in CONDITIONS]),
        signal_norm=np.stack([aggregated[name]["signal_norm"] for name in CONDITIONS]),
        sweep_seed=sweep["seed"],
        sweep_fraction=sweep["fraction"],
        sweep_alignment=sweep["alignment"],
        sweep_decodability=sweep["decodability"],
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E1_alignment",
        "configuration": vars(args),
        "runtime": runtime_metadata(),
        "checkpoint_info": info,
        "network_parameters": params,
        "protocol": {
            "dataset": "validated_notebook_rowwise_numpy_choice",
            "operation_order": "seed -> initialize EC-CA3 -> draw patterns -> store -> recall",
            "pattern_sha256_by_seed": [
                array_digest(result["metadata"]["patterns"])
                for result in seed_results
            ],
        },
        "summary": summary,
        "paired_effects": {
            "aligned_minus_fixed": (aligned - fixed),
            "aligned_minus_random_matched": (aligned - random_matched),
            "rescue_minus_fixed": (rescued - fixed),
            "rescue_minus_aligned": (rescued - aligned),
        },
        "alignment_decodability_correlation": correlation,
        "causal_checks": causal_checks,
        "legacy_parity": parity,
    }
    report["outputs"] = {
        "metrics_json": str(prefix.with_suffix(".json")),
        "raw_npz": str(prefix.with_suffix(".npz")),
        "figure_png": str(prefix.with_suffix(".png")),
        "figure_pdf": str(prefix.with_suffix(".pdf")),
    }
    with prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(report), handle, indent=2, sort_keys=True)
        handle.write("\n")
    make_figure(aggregated, sweep, parity, prefix.with_suffix(".png"))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="ae_8")
    parser.add_argument("--seed-start", type=int, default=500)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--sweep-seeds", type=int, default=8)
    parser.add_argument("--num-patterns", type=int, default=28)
    parser.add_argument("--num-alignment-levels", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--ca3-active", type=int, default=22)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-legacy-parity", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.num_seeds < 2:
        parser.error("--num-seeds must be at least 2")
    if args.sweep_seeds < 1:
        parser.error("--sweep-seeds must be positive")
    if args.num_patterns < 2:
        parser.error("--num-patterns must be at least 2")
    return args


def main() -> int:
    args = parse_args()
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    report = run(args)
    concise = {
        "causal_checks": report["causal_checks"],
        "alignment_decodability_correlation": report["alignment_decodability_correlation"],
        "condition_cosine_means": {
            condition: report["summary"][condition]["cosine"]["mean"]
            for condition in CONDITIONS
        },
        "legacy_parity": {
            key: report["legacy_parity"].get(key)
            for key in (
                "available",
                "passed",
                "maximum_absolute_error",
                "saved_aligned_mean",
                "saved_shuffled_mean",
            )
        },
        "outputs": report["outputs"],
    }
    print(json.dumps(json_ready(concise), indent=2))
    required = [value for value in report["causal_checks"].values() if value is not None]
    return 0 if all(required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
