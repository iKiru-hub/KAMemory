"""E3 companion: clean reproduction of the notebook cue-remapping result.

This script replays ``lab_fig_c1.ipynb`` experiment II: the same sensory cue
is learned on two otherwise identical tracks, first at position 10 and then
at position 30. It preserves the notebook's generator, checkpoint, RNG call
order, pre-update activity recording, cell threshold, and effective-input
calculation while adding seed-level quantification and machine-readable data.

Run from the repository root:

    python3 src/experiments/04b_legacy_remapping.py --deterministic
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

from kamemory.data import generate_legacy_remapping_track
from kamemory.io import load_autoencoder_session, runtime_metadata
from kamemory.models import BTSPMemory
from kamemory.utils import array_digest, seed_everything


SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
NOTEBOOK = "notebooks/lab_fig_c1.ipynb, experiment II"


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


def session_parameters(info: dict) -> dict[str, int | float | bool]:
    params = info["network_params"]
    return {
        "track_length": int(params["mec_N_x"]),
        "spatial_dim": int(params["dim_mec"]),
        "sensory_dim": int(params["dim_lec"]),
        "dim_ca1": int(params["dim_ca1"]),
        "dim_ca3": int(params["dim_ca3"]),
        "sensory_active": int(params["K_lec"]),
        "output_active": int(params["K_eo"]),
        "ca1_active": int(params["K_ca1"]),
        "ca3_active": int(params["K_ca3"]),
        "beta": float(params["beta_ca1"]),
        "place_field_sigma": float(params["mec_sigma"]),
        "bias": bool(params.get("bias", False)),
    }


def make_memory(autoencoder, params: dict, alpha: float) -> BTSPMemory:
    return BTSPMemory.from_autoencoder(
        autoencoder,
        K_lat=params["ca1_active"],
        K_out=params["output_active"],
        dim_ca3=params["dim_ca3"],
        K_ca3=params["ca3_active"],
        beta=params["beta"],
        alpha=alpha,
    )


def replay_track(
    memory: BTSPMemory,
    inputs: np.ndarray,
    alpha_samples: np.ndarray,
    *,
    alpha: float,
    alpha_baseline: float,
    num_laps: int,
    track_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the legacy loop, including activity sampled before each update."""

    activity = np.empty((len(inputs), memory._dim_ca1), dtype=np.float32)
    with torch.no_grad():
        for index, pattern in enumerate(inputs):
            # Legacy MTL.forward computed and recorded CA1 before updating W.
            _, ca1 = memory.retrieve(pattern, return_ca1=True)
            activity[index] = ca1.detach().cpu().numpy().reshape(-1)
            memory.set_alpha(max(alpha_baseline, alpha * float(alpha_samples[index])))
            memory.store(pattern)
    return (
        activity.reshape(num_laps, track_length, memory._dim_ca1),
        memory.W_ca3_ca1.detach().cpu().numpy().copy(),
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left.reshape(-1), right.reshape(-1)) / max(denominator, 1e-12))


def signed_circular_shift(start: float, stop: float, period: int) -> float:
    return float((stop - start + period / 2) % period - period / 2)


def analyze_run(
    activity1: np.ndarray,
    activity2: np.ndarray,
    weight1: np.ndarray,
    weight2: np.ndarray,
    projection: np.ndarray,
    cue_positions: tuple[int, int],
    threshold: float,
) -> dict[str, np.ndarray | float | int]:
    mean1 = activity1.mean(axis=0)
    mean2 = activity2.mean(axis=0)
    cue1, cue2 = cue_positions
    selected1 = np.flatnonzero(mean1[cue1] > threshold)
    selected2 = np.flatnonzero(mean2[cue2] > threshold)
    if not len(selected1) or not len(selected2):
        raise RuntimeError("legacy cell threshold selected no cue-responsive cells")

    profile1 = mean1[:, selected1].mean(axis=1)
    profile2 = mean2[:, selected2].mean(axis=1)
    profile_peak1 = int(np.argmax(profile1))
    profile_peak2 = int(np.argmax(profile2))
    preferred1 = np.argmax(mean1, axis=0)
    preferred2 = np.argmax(mean2, axis=0)
    intersection = np.intersect1d(selected1, selected2)

    effective1 = projection.T @ weight1.T
    effective2 = projection.T @ weight2.T
    overlap = len(intersection) / max(len(np.union1d(selected1, selected2)), 1)
    same_cell_shifts = np.asarray(
        [
            signed_circular_shift(preferred1[index], preferred2[index], len(mean1))
            for index in intersection
        ],
        dtype=np.float32,
    )
    return {
        "mean1": mean1,
        "mean2": mean2,
        "selected1": selected1,
        "selected2": selected2,
        "preferred1": preferred1,
        "preferred2": preferred2,
        "profile1": profile1,
        "profile2": profile2,
        "profile_peak1": profile_peak1,
        "profile_peak2": profile_peak2,
        "profile_shift": signed_circular_shift(profile_peak1, profile_peak2, len(mean1)),
        "selected_count1": len(selected1),
        "selected_count2": len(selected2),
        "ensemble_jaccard": overlap,
        "same_cell_count": len(intersection),
        "same_cell_shift_mean": float(same_cell_shifts.mean()) if len(same_cell_shifts) else np.nan,
        "same_cell_shift_median": float(np.median(same_cell_shifts)) if len(same_cell_shifts) else np.nan,
        "cue_population_cosine": cosine(mean1[cue1], mean2[cue2]),
        "same_position_cosine": cosine(mean1[cue1], mean2[cue1]),
        "effective1": effective1,
        "effective2": effective2,
        "effective_change_norm": float(np.linalg.norm(effective2 - effective1)),
    }


def run_seed(
    seed: int,
    autoencoder,
    params: dict,
    *,
    cue_positions: tuple[int, int],
    num_laps: int,
    alpha: float,
    alpha_baseline: float,
    threshold: float,
    deterministic: bool,
) -> dict:
    seed_everything(seed, deterministic=deterministic)
    inputs1, lap_cues1, alpha_samples1 = generate_legacy_remapping_track(
        track_length=params["track_length"],
        num_laps=num_laps,
        cue_position=cue_positions[0],
        spatial_size=params["spatial_dim"],
        sensory_size=params["sensory_dim"],
        sensory_active=params["sensory_active"],
        place_field_sigma=params["place_field_sigma"],
    )
    inputs2, lap_cues2, alpha_samples2 = generate_legacy_remapping_track(
        track_length=params["track_length"],
        num_laps=num_laps,
        cue_position=cue_positions[1],
        spatial_size=params["spatial_dim"],
        sensory_size=params["sensory_dim"],
        sensory_active=params["sensory_active"],
        place_field_sigma=params["place_field_sigma"],
    )
    memory = make_memory(autoencoder, params, alpha)
    projection = memory.W_ei_ca3.detach().cpu().numpy().copy()
    activity1, weight1 = replay_track(
        memory,
        inputs1,
        alpha_samples1,
        alpha=alpha,
        alpha_baseline=alpha_baseline,
        num_laps=num_laps,
        track_length=params["track_length"],
    )
    memory.reset_memory()
    activity2, weight2 = replay_track(
        memory,
        inputs2,
        alpha_samples2,
        alpha=alpha,
        alpha_baseline=alpha_baseline,
        num_laps=num_laps,
        track_length=params["track_length"],
    )
    analysis = analyze_run(
        activity1, activity2, weight1, weight2, projection, cue_positions, threshold
    )
    analysis.update(
        {
            "seed": seed,
            "inputs1": inputs1,
            "inputs2": inputs2,
            "lap_cues1": lap_cues1,
            "lap_cues2": lap_cues2,
            "alpha_samples1": alpha_samples1,
            "alpha_samples2": alpha_samples2,
            "activity1": activity1,
            "activity2": activity2,
            "weight1": weight1,
            "weight2": weight2,
            "projection": projection,
        }
    )
    return analysis


def save_notebook_place_fields(run: dict, output_dir: Path) -> None:
    mean1, mean2 = run["mean1"], run["mean2"]
    order = np.argsort(np.argmax(mean1, axis=0))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharex=True, sharey=True)
    for index, (axis, activity, cue) in enumerate(zip(axes, (mean1, mean2), (10, 30)), 1):
        image = axis.imshow(
            activity[:, order].T[:500], aspect="auto", cmap="plasma",
            interpolation="nearest", vmin=0,
        )
        axis.axvline(10, lw=1.5, color="white", alpha=0.65)
        axis.axvline(30, lw=1.5, color="white", alpha=0.65)
        axis.axvline(cue, lw=2.5, color="#58D3F7")
        axis.set(title=f"Cue at position {cue}", xlabel="Position on track")
        axis.set_xlim(0, 49)
        axis.set_xticks([0, 10, 30, 49], [0, 10, 30, 50])
        if index == 1:
            axis.set_ylabel("CA1 cells (trial-1 field order)")
    fig.colorbar(image, ax=axes, label="Mean CA1 activity", fraction=0.025, pad=0.02)
    fig.suptitle("Legacy notebook reproduction: place-field remapping", y=1.02)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"e3_legacy_remapping_place_fields.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_notebook_effective_inputs(run: dict, output_dir: Path, spatial_dim: int) -> None:
    effective1, effective2 = run["effective1"], run["effective2"]
    order = np.argsort(effective1[spatial_dim:].mean(axis=0))
    cutoff = 600
    fig, axes = plt.subplots(3, 1, figsize=(5.5, 5), sharex=True)
    panels = (effective1.T[order][cutoff:], effective2.T[order][cutoff:], (effective2 - effective1).T[order][cutoff:])
    labels = ("Cue at 10", "Cue at 30", "Difference")
    for index, (axis, panel, label) in enumerate(zip(axes, panels, labels)):
        if index < 2:
            lower, upper = np.quantile(panel, (0.01, 0.99))
            cmap = "viridis"
        else:
            upper = float(np.quantile(np.abs(panel), 0.99))
            lower, cmap = -upper, "seismic"
        image = axis.imshow(
            panel, aspect="auto", cmap=cmap, vmin=lower, vmax=upper,
            interpolation="nearest",
        )
        axis.axvline(spatial_dim, lw=1, color="black")
        axis.set_ylabel(label)
        axis.set_yticks([])
    axes[-1].set_xticks([0, 10, 30, 50, 99], [0, 10, 30, 50, 100])
    axes[-1].set_xlabel("EC input (spatial | sensory)")
    fig.colorbar(image, ax=axes, label="Change in effective input weight", fraction=0.025, pad=0.02)
    fig.suptitle("Legacy effective EC→CA3→CA1 receptive fields", y=0.97)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"e3_legacy_remapping_effective_inputs.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary_figure(run: dict, seed_metrics: dict[str, np.ndarray], output_dir: Path) -> None:
    cue1, cue2 = 10, 30
    mean1, mean2 = run["mean1"], run["mean2"]
    order = np.argsort(np.argmax(mean1, axis=0))[::-1]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for axis, activity, cue, title in zip(
        axes[0, :2], (mean1, mean2), (cue1, cue2), ("Cue at 10", "Same cue at 30")
    ):
        axis.imshow(activity[:, order].T[:500], aspect="auto", cmap="plasma", vmin=0)
        axis.axvline(cue, color="#58D3F7", lw=2)
        axis.set(title=title, xlabel="Track position", ylabel="CA1 cells")

    axes[0, 2].plot(run["profile1"], color="#2878B5", lw=2, label="Cue at 10 ensemble")
    axes[0, 2].plot(run["profile2"], color="#D95F59", lw=2, label="Cue at 30 ensemble")
    axes[0, 2].axvline(cue1, color="#2878B5", ls="--", alpha=0.7)
    axes[0, 2].axvline(cue2, color="#D95F59", ls="--", alpha=0.7)
    axes[0, 2].set(title="Cue-responsive population fields", xlabel="Track position", ylabel="Mean CA1 activity")
    axes[0, 2].legend(frameon=False, fontsize=8)

    bins = np.arange(51) - 0.5
    axes[1, 0].hist(run["preferred1"][run["selected1"]], bins=bins, alpha=0.65, color="#2878B5", label="Cue at 10")
    axes[1, 0].hist(run["preferred2"][run["selected2"]], bins=bins, alpha=0.65, color="#D95F59", label="Cue at 30")
    axes[1, 0].set(title="Fields of threshold-selected cells", xlabel="Preferred position", ylabel="Cell count")
    axes[1, 0].legend(frameon=False)

    effective1, effective2 = run["effective1"], run["effective2"]
    effective_change = effective2 - effective1
    effective_order = np.argsort(effective1[50:].mean(axis=0))
    change_panel = effective_change.T[effective_order][600:]
    change_limit = float(np.quantile(np.abs(change_panel), 0.99))
    image = axes[1, 1].imshow(change_panel, aspect="auto", cmap="seismic", vmin=-change_limit, vmax=change_limit)
    axes[1, 1].axvline(50, color="black", lw=1)
    axes[1, 1].set(title="Effective receptive-field change", xlabel="EC input (spatial | sensory)", ylabel="CA1 cells")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.03)

    x = np.arange(len(seed_metrics["seed"]))
    axes[1, 2].plot(x, seed_metrics["profile_peak1"], "o", color="#2878B5", label="Cue at 10")
    axes[1, 2].plot(x, seed_metrics["profile_peak2"], "o", color="#D95F59", label="Cue at 30")
    for index in x:
        axes[1, 2].plot([index, index], [seed_metrics["profile_peak1"][index], seed_metrics["profile_peak2"][index]], color="0.75", zorder=0)
    axes[1, 2].axhline(cue1, color="#2878B5", ls="--", alpha=0.6)
    axes[1, 2].axhline(cue2, color="#D95F59", ls="--", alpha=0.6)
    axes[1, 2].set(title="Population-field shift across seeds", xlabel="Paired seed", ylabel="Population-field peak", ylim=(-1, 50))
    axes[1, 2].legend(frameon=False, fontsize=8)

    for label, axis in zip("ABCDEF", axes.flat):
        axis.text(-0.12, 1.08, label, transform=axis.transAxes, weight="bold", fontsize=13)
    fig.suptitle("E3 legacy companion: moving one cue remaps CA1 fields", y=1.01, fontsize=15)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"e3_legacy_remapping.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="ae_6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=12)
    parser.add_argument("--num-laps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--alpha-baseline", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if args.num_seeds < 1:
        parser.error("--num-seeds must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    info, autoencoder = load_autoencoder_session(args.session)
    params = session_parameters(info)
    if params["track_length"] != 50 or params["spatial_dim"] != 50:
        raise ValueError("the notebook reproduction requires the 50-bin ae_6 track model")
    cue_positions = (10, 30)
    runs = []
    for seed in range(args.seed, args.seed + args.num_seeds):
        print(f"legacy remapping seed {seed} ({len(runs) + 1}/{args.num_seeds})", flush=True)
        runs.append(
            run_seed(
                seed, autoencoder, params,
                cue_positions=cue_positions,
                num_laps=args.num_laps,
                alpha=args.alpha,
                alpha_baseline=args.alpha_baseline,
                threshold=args.threshold,
                deterministic=args.deterministic,
            )
        )

    metric_names = (
        "profile_peak1", "profile_peak2", "profile_shift", "selected_count1",
        "selected_count2", "ensemble_jaccard", "same_cell_count",
        "same_cell_shift_mean", "same_cell_shift_median", "cue_population_cosine",
        "same_position_cosine", "effective_change_norm",
    )
    seed_metrics = {"seed": np.asarray([run["seed"] for run in runs], dtype=int)}
    for name in metric_names:
        seed_metrics[name] = np.asarray([run[name] for run in runs])
    representative = runs[0]

    save_notebook_place_fields(representative, args.output_dir)
    save_notebook_effective_inputs(representative, args.output_dir, params["spatial_dim"])
    save_summary_figure(representative, seed_metrics, args.output_dir)

    np.savez_compressed(
        args.output_dir / "e3_legacy_remapping.npz",
        **seed_metrics,
        representative_inputs1=representative["inputs1"],
        representative_inputs2=representative["inputs2"],
        representative_alpha_samples1=representative["alpha_samples1"],
        representative_alpha_samples2=representative["alpha_samples2"],
        representative_activity1=representative["activity1"],
        representative_activity2=representative["activity2"],
        representative_weight1=representative["weight1"],
        representative_weight2=representative["weight2"],
        representative_projection=representative["projection"],
        representative_selected1=representative["selected1"],
        representative_selected2=representative["selected2"],
    )
    constant_profile = bool(
        np.all(np.maximum(args.alpha_baseline, args.alpha * representative["alpha_samples1"]) == args.alpha_baseline)
        and np.all(np.maximum(args.alpha_baseline, args.alpha * representative["alpha_samples2"]) == args.alpha_baseline)
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E3 legacy cue-position remapping companion",
        "source_notebook": NOTEBOOK,
        "status": "protocol reproduction; prior notebook arrays were not saved for bitwise comparison",
        "checkpoint": args.session,
        "checkpoint_metadata": {
            "date": info.get("date"), "loss_ae": info.get("loss_ae"),
            "metadata_matches_notebook_printout": info.get("date") == "14/10/2024 19:03:10" and info.get("loss_ae") == 0.00197,
        },
        "notebook_visible_references": {
            "selected_cell_counts": [24, 22],
            "cue_population_cosine": 0.08468431,
            "representative_selected_cell_counts": [
                representative["selected_count1"], representative["selected_count2"]
            ],
            "representative_cue_population_cosine": representative["cue_population_cosine"],
            "cosine_absolute_difference": abs(
                representative["cue_population_cosine"] - 0.08468431
            ),
            "interpretation": (
                "close protocol/visual reproduction; not bitwise parity because the "
                "notebook consumed undocumented global RNG state before experiment II"
            ),
        },
        "protocol": {
            "cue_positions": cue_positions, "num_laps": args.num_laps,
            "alpha": args.alpha, "alpha_baseline": args.alpha_baseline,
            "effective_alpha_is_constant": constant_profile,
            "activity_recording": "immediately before each storage update (legacy MTL order)",
            "cell_selection": f"mean activity at the condition's cue position > {args.threshold}",
            "effective_input": "W_ei_ca3.T @ W_ca3_ca1.T",
            "representative_seed": args.seed,
            "paired_seeds": seed_metrics["seed"],
        },
        "representative": {name: representative[name] for name in metric_names},
        "across_seeds": {
            name: {
                "mean": float(np.nanmean(seed_metrics[name])),
                "sd": float(np.nanstd(seed_metrics[name], ddof=1)) if args.num_seeds > 1 else 0.0,
                "values": seed_metrics[name],
            }
            for name in metric_names
        },
        "digests": {
            "representative_inputs1": array_digest(representative["inputs1"]),
            "representative_inputs2": array_digest(representative["inputs2"]),
            "representative_activity1": array_digest(representative["activity1"]),
            "representative_activity2": array_digest(representative["activity2"]),
        },
        "parameters": params,
        "runtime": runtime_metadata(),
    }
    with (args.output_dir / "e3_legacy_remapping.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(json_ready({
        "checkpoint_match": summary["checkpoint_metadata"]["metadata_matches_notebook_printout"],
        "constant_effective_alpha": constant_profile,
        "profile_peak1_mean": summary["across_seeds"]["profile_peak1"]["mean"],
        "profile_peak2_mean": summary["across_seeds"]["profile_peak2"]["mean"],
        "profile_shift_mean": summary["across_seeds"]["profile_shift"]["mean"],
        "selected_counts_representative": [representative["selected_count1"], representative["selected_count2"]],
        "output": str(args.output_dir / "e3_legacy_remapping.png"),
    }), indent=2))


if __name__ == "__main__":
    main()
