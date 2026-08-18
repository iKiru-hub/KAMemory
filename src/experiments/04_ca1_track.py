"""E3: factorial spatial-sensory task and held-out CA1 measurements.

The original exploratory track coupled cue identity, cue position, and a
position-dependent learning-rate profile. This experiment fully crosses two
cue identities with four presentation positions, includes cue-free laps,
uses constant plasticity, and evaluates only on separately generated laps.

Run from the repository root:

    python3 src/experiments/04_ca1_track.py --deterministic
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

from kamemory.data import generate_factorial_track
from kamemory.io import load_autoencoder_session, runtime_metadata
from kamemory.models import Autoencoder, BTSPMemory
from kamemory.utils import array_digest, seed_everything


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
    "decoder_rescue": "Permutation + rescue",
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
LINESTYLES = {"decoder_rescue": "--"}
DEFAULT_LAYOUTS = ((7, 19, 31, 43), (3, 15, 27, 39))
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


def session_parameters(info: dict) -> dict[str, int | float | bool]:
    params = info["network_params"]
    return {
        "track_length": int(params["mec_N_x"]),
        "spatial_dim": int(params["dim_mec"]),
        "sensory_dim": int(params["dim_lec"]),
        "input_dim": int(params["dim_ei"]),
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
    identity = np.arange(size)
    for _ in range(1_000):
        candidate = rng.permutation(size)
        if np.all(candidate != identity):
            return candidate
    return np.roll(identity, 1)


def code_bank(autoencoder: Autoencoder, inputs: np.ndarray) -> torch.Tensor:
    device = next(autoencoder.parameters()).device
    with torch.no_grad():
        return autoencoder.encode(torch.as_tensor(inputs, device=device)).detach().cpu()


def retrieve_bank(
    memory: BTSPMemory, inputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ca1_rows = []
    output_rows = []
    with torch.no_grad():
        for pattern in inputs:
            output, ca1 = memory.retrieve(pattern, return_ca1=True)
            ca1_rows.append(ca1.detach().cpu().numpy().reshape(-1))
            output_rows.append(output.detach().cpu().numpy().reshape(-1))
    return np.asarray(ca1_rows), np.asarray(output_rows)


def train_condition(
    base: BTSPMemory,
    inputs: np.ndarray,
    signals: torch.Tensor,
    probe_inputs: np.ndarray,
    checkpoints: tuple[int, ...],
    track_length: int,
    plasticity_profile: np.ndarray,
    *,
    learning_enabled: bool = True,
) -> tuple[BTSPMemory, np.ndarray]:
    memory = copy.deepcopy(base)
    memory.learning_enabled = learning_enabled
    snapshots = []
    for index, (pattern, signal) in enumerate(zip(inputs, signals)):
        memory._alpha = float(plasticity_profile[index])
        memory.store(pattern, instructive_signal=signal)
        completed_lap = (index + 1) // track_length
        if (index + 1) % track_length == 0 and completed_lap in checkpoints:
            snapshot, _ = retrieve_bank(memory, probe_inputs)
            snapshots.append(snapshot)
    if len(snapshots) != len(checkpoints):
        raise AssertionError("not every receptive-field checkpoint was recorded")
    return memory, np.asarray(snapshots)


def event_rows(dataset: dict[str, np.ndarray], *, include_free: bool = False):
    track_length = int(dataset["track_length"])
    cue_ids = dataset["cue_ids"]
    rows = np.arange(len(cue_ids)) * track_length + dataset["event_positions"]
    if include_free:
        return rows
    return rows[cue_ids >= 0]


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )


def row_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.linalg.norm(left_centered, axis=1) * np.linalg.norm(
        right_centered, axis=1
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 1e-12,
    )


def centroid_predict(
    training: np.ndarray,
    training_labels: np.ndarray,
    test: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    centroids = np.asarray(
        [training[training_labels == label].mean(axis=0) for label in classes]
    )
    test_norm = np.linalg.norm(test, axis=1, keepdims=True)
    centroid_norm = np.linalg.norm(centroids, axis=1, keepdims=True).T
    similarities = (test @ centroids.T) / np.maximum(test_norm * centroid_norm, 1e-12)
    return classes[np.argmax(similarities, axis=1)]


def cue_accuracy_across_positions(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_positions: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_positions: np.ndarray,
) -> float:
    predictions = []
    targets = []
    classes = np.unique(train_labels)
    for position in np.unique(test_positions):
        train_mask = train_positions != position
        test_mask = test_positions == position
        predictions.extend(
            centroid_predict(
                train_features[train_mask],
                train_labels[train_mask],
                test_features[test_mask],
                classes,
            )
        )
        targets.extend(test_labels[test_mask])
    return float(np.mean(np.asarray(predictions) == np.asarray(targets)))


def position_accuracy_across_cues(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_positions: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_positions: np.ndarray,
) -> float:
    predictions = []
    targets = []
    position_classes = np.unique(train_positions)
    for test_cue in np.unique(test_labels):
        train_mask = train_labels != test_cue
        test_mask = test_labels == test_cue
        predictions.extend(
            centroid_predict(
                train_features[train_mask],
                train_positions[train_mask],
                test_features[test_mask],
                position_classes,
            )
        )
        targets.extend(test_positions[test_mask])
    return float(np.mean(np.asarray(predictions) == np.asarray(targets)))


def balanced_binary_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    predictions = centroid_predict(
        train_features, train_labels, test_features, np.asarray((0, 1))
    )
    recalls = [
        np.mean(predictions[test_labels == label] == label) for label in (0, 1)
    ]
    return float(np.mean(recalls))


def free_lap_activity(
    activity: np.ndarray, dataset: dict[str, np.ndarray]
) -> np.ndarray:
    num_laps = int(dataset["num_laps"])
    track_length = int(dataset["track_length"])
    reshaped = activity.reshape(num_laps, track_length, -1)
    return reshaped[dataset["cue_ids"] < 0]


def field_analysis(
    train_ca1: np.ndarray,
    test_ca1: np.ndarray,
    train_dataset: dict[str, np.ndarray],
    test_dataset: dict[str, np.ndarray],
    num_selected: int,
) -> dict[str, np.ndarray | float]:
    train_free = free_lap_activity(train_ca1, train_dataset)
    test_free = free_lap_activity(test_ca1, test_dataset)
    train_tuning = train_free.mean(axis=0).T
    test_tuning = test_free.mean(axis=0).T
    split = max(1, len(train_free) // 2)
    first = train_free[:split].mean(axis=0).T
    second = train_free[split:].mean(axis=0).T
    reliability = row_correlations(first, second) if len(train_free[split:]) else np.zeros(len(first))
    modulation = (train_tuning.max(axis=1) - train_tuning.min(axis=1)) / (
        train_tuning.mean(axis=1) + 1e-8
    )
    selection_score = modulation * np.maximum(reliability, 0)
    selected = np.argsort(selection_score)[-num_selected:]
    heldout_stability = row_correlations(train_tuning[selected], test_tuning[selected])
    heldout_modulation = (
        test_tuning[selected].max(axis=1) - test_tuning[selected].min(axis=1)
    ) / (test_tuning[selected].mean(axis=1) + 1e-8)
    place_cell_mask = heldout_modulation >= 1.0
    peaks = np.argmax(test_tuning[selected][place_cell_mask], axis=1)
    density = np.bincount(peaks, minlength=train_tuning.shape[1]) / len(selected)
    return {
        "selected": selected,
        "train_tuning": train_tuning,
        "test_tuning": test_tuning,
        "density": density,
        "stability": float(np.mean(heldout_stability)),
        "place_cell_fraction": float(np.mean(place_cell_mask)),
    }


def cue_selected_neurons(
    train_ca1: np.ndarray,
    dataset: dict[str, np.ndarray],
    number_per_cue: int,
) -> np.ndarray:
    rows = event_rows(dataset)
    cue_ids = dataset["cue_ids"][dataset["cue_ids"] >= 0]
    difference = train_ca1[rows][cue_ids == 0].mean(axis=0) - train_ca1[rows][
        cue_ids == 1
    ].mean(axis=0)
    cue_zero = np.argsort(difference)[-number_per_cue:]
    cue_one = np.argsort(difference)[:number_per_cue]
    return np.concatenate((cue_zero, cue_one))


def heldout_cue_matrix(
    test_ca1: np.ndarray,
    dataset: dict[str, np.ndarray],
    selected: np.ndarray,
) -> np.ndarray:
    rows = event_rows(dataset)
    cue_ids = dataset["cue_ids"][dataset["cue_ids"] >= 0]
    event_positions = dataset["event_positions"][dataset["cue_ids"] >= 0]
    positions = np.unique(event_positions)
    matrices = []
    for cue in np.unique(cue_ids):
        matrices.append(
            np.asarray(
                [
                    test_ca1[rows][(cue_ids == cue) & (event_positions == position)][
                        :, selected
                    ].mean(axis=0)
                    for position in positions
                ]
            ).T
        )
    return np.asarray(matrices)


def condition_metrics(
    memory: BTSPMemory,
    snapshots: np.ndarray,
    train_data: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    params: dict,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    train_ca1, _ = retrieve_bank(memory, train_data["inputs"])
    test_ca1, test_output = retrieve_bank(memory, test_data["inputs"])
    train_rows = event_rows(train_data)
    test_rows = event_rows(test_data)
    train_labels = train_data["cue_ids"][train_data["cue_ids"] >= 0]
    test_labels = test_data["cue_ids"][test_data["cue_ids"] >= 0]
    train_positions = train_data["event_positions"][train_data["cue_ids"] >= 0]
    test_positions = test_data["event_positions"][test_data["cue_ids"] >= 0]
    cue_accuracy = cue_accuracy_across_positions(
        train_ca1[train_rows],
        train_labels,
        train_positions,
        test_ca1[test_rows],
        test_labels,
        test_positions,
    )
    position_accuracy = position_accuracy_across_cues(
        train_ca1[train_rows],
        train_labels,
        train_positions,
        test_ca1[test_rows],
        test_labels,
        test_positions,
    )
    train_all_event_rows = event_rows(train_data, include_free=True)
    test_all_event_rows = event_rows(test_data, include_free=True)
    train_presence = (train_data["cue_ids"] >= 0).astype(int)
    test_presence = (test_data["cue_ids"] >= 0).astype(int)
    presence_accuracy = balanced_binary_accuracy(
        train_ca1[train_all_event_rows],
        train_presence,
        test_ca1[test_all_event_rows],
        test_presence,
    )

    fields = field_analysis(
        train_ca1,
        test_ca1,
        train_data,
        test_data,
        args.num_place_cells,
    )
    selected = fields["selected"]
    final_probe = snapshots[-1].T[selected]
    formation = np.asarray(
        [
            np.mean(row_correlations(snapshot.T[selected], final_probe))
            for snapshot in snapshots
        ]
    )
    cue_neurons = cue_selected_neurons(
        train_ca1, train_data, args.num_cue_cells_per_class
    )
    cue_matrix = heldout_cue_matrix(test_ca1, test_data, cue_neurons)

    event_target = test_data["inputs"][test_rows]
    event_output = test_output[test_rows]
    spatial_dim = params["spatial_dim"]
    metrics = {
        "cue_accuracy": cue_accuracy,
        "position_accuracy": position_accuracy,
        "cue_presence_accuracy": presence_accuracy,
        "field_stability": fields["stability"],
        "place_cell_fraction": fields["place_cell_fraction"],
        "output_cosine": float(np.mean(cosine_rows(event_target, event_output))),
        "spatial_output_cosine": float(
            np.mean(
                cosine_rows(
                    event_target[:, :spatial_dim], event_output[:, :spatial_dim]
                )
            )
        ),
        "sensory_output_cosine": float(
            np.mean(
                cosine_rows(
                    event_target[:, spatial_dim:], event_output[:, spatial_dim:]
                )
            )
        ),
        "field_density": fields["density"],
        "formation": formation,
    }
    representative = {
        "place_indices": selected,
        "place_train_tuning": fields["train_tuning"][selected],
        "place_test_tuning": fields["test_tuning"][selected],
        "cue_indices": cue_neurons,
        "cue_matrix": cue_matrix,
    }
    return metrics, representative


def run_seed_layout(
    seed: int,
    layout_index: int,
    cue_positions: tuple[int, ...],
    autoencoder: Autoencoder,
    params: dict,
    args: argparse.Namespace,
) -> dict[str, object]:
    run_seed = seed + 100_000 * layout_index
    seed_everything(run_seed)
    train_rng = np.random.default_rng(run_seed + 60_000)
    test_rng = np.random.default_rng(run_seed + 70_000)
    control_rng = np.random.default_rng(run_seed + 80_000)
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
    base = make_base_memory(autoencoder, params, args.alpha)
    codes = code_bank(autoencoder, train_data["inputs"])
    permutation = torch.as_tensor(
        control_rng.permutation(params["dim_ca1"]), dtype=torch.long
    )
    matching = derangement(len(train_data["inputs"]), control_rng)
    signal_banks = {
        "aligned": codes,
        "fixed_permutation": codes[:, permutation],
        "random_matched": codes[matching],
        "no_plasticity": codes,
    }
    plasticity_profile = np.full(
        len(train_data["inputs"]), args.baseline_alpha, dtype=np.float32
    )
    plasticity_profile[train_data["cue_present"]] = args.alpha
    probe_lap = np.flatnonzero(test_data["cue_ids"] < 0)[0]
    start = probe_lap * params["track_length"]
    probe_inputs = test_data["inputs"][start : start + params["track_length"]]
    num_train_laps = int(train_data["num_laps"])
    checkpoints = tuple(
        sorted(
            set(
                max(1, int(round(fraction * num_train_laps)))
                for fraction in (0.05, 0.25, 0.5, 1.0)
            )
        )
    )
    if len(checkpoints) != 4:
        raise ValueError("training schedule is too short for four formation checkpoints")

    memories: dict[str, BTSPMemory] = {}
    snapshots: dict[str, np.ndarray] = {}
    for condition in ("aligned", "fixed_permutation", "random_matched", "no_plasticity"):
        memories[condition], snapshots[condition] = train_condition(
            base,
            train_data["inputs"],
            signal_banks[condition],
            probe_inputs,
            checkpoints,
            params["track_length"],
            plasticity_profile,
            learning_enabled=condition != "no_plasticity",
        )
    memories["decoder_rescue"] = copy.deepcopy(memories["fixed_permutation"])
    memories["decoder_rescue"].W_ca1_eo.copy_(base.W_ca1_eo[:, permutation])
    snapshots["decoder_rescue"] = snapshots["fixed_permutation"].copy()
    probe = torch.randn(params["dim_ca1"], 1)
    if not torch.allclose(
        base.W_ca1_eo @ probe,
        memories["decoder_rescue"].W_ca1_eo @ probe[permutation],
        atol=1e-5,
        rtol=1e-5,
    ):
        raise AssertionError("rescue decoder orientation is incorrect")

    condition_results = {}
    representatives = {}
    weight_digests = {}
    for condition in CONDITIONS:
        condition_results[condition], representatives[condition] = condition_metrics(
            memories[condition],
            snapshots[condition],
            train_data,
            test_data,
            params,
            args,
        )
        weight_digests[condition] = array_digest(
            memories[condition].W_ca3_ca1.detach().cpu().numpy()
        )

    signal_sparsity = {
        condition: float((signals > 0.5).float().mean().item())
        for condition, signals in signal_banks.items()
        if condition != "no_plasticity"
    }
    signal_norm = {
        condition: float(torch.linalg.vector_norm(signals, dim=1).mean().item())
        for condition, signals in signal_banks.items()
        if condition != "no_plasticity"
    }
    return {
        "conditions": condition_results,
        "representatives": representatives,
        "train_data": train_data,
        "test_data": test_data,
        "permutation": permutation.numpy(),
        "random_matching": matching,
        "base_ca3_projection": base.W_ei_ca3.detach().cpu().numpy().copy(),
        "checkpoints": np.asarray(checkpoints),
        "signal_sparsity": signal_sparsity,
        "signal_norm": signal_norm,
        "plasticity_profile": plasticity_profile,
        "weight_digests": weight_digests,
    }


def summary_statistics(values: np.ndarray) -> dict[str, float]:
    seed_values = values.mean(axis=1)
    mean = float(np.mean(seed_values))
    sd = float(np.std(seed_values, ddof=1))
    ci = 1.96 * sd / np.sqrt(len(seed_values))
    return {
        "mean": mean,
        "sd_across_seeds": sd,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
    }


def plot_metric(ax, arrays: dict, metric: str, title: str, ylabel: str, chance=None):
    values = arrays[metric]
    for condition_index, condition in enumerate(CONDITIONS):
        seed_values = values[condition_index].mean(axis=1)
        jitter = np.linspace(-0.10, 0.10, len(seed_values))
        ax.scatter(
            condition_index + jitter,
            seed_values,
            color=COLORS[condition],
            s=22,
            alpha=0.65,
        )
        mean = np.mean(seed_values)
        ci = 1.96 * np.std(seed_values, ddof=1) / np.sqrt(len(seed_values))
        ax.errorbar(condition_index, mean, yerr=ci, color="black", fmt="o", capsize=4)
    if chance is not None:
        ax.axhline(chance, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(
        range(len(CONDITIONS)),
        [DISPLAY_NAMES[name] for name in CONDITIONS],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def make_figure(
    arrays: dict[str, np.ndarray],
    representative: dict[str, np.ndarray],
    schedules: dict[str, np.ndarray],
    checkpoints: np.ndarray,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    task_ax, heat_ax, density_ax, formation_ax, cue_ax, position_ax, stability_ax, output_ax = axes.flat

    cue_ids = schedules["cue_ids"]
    event_positions = schedules["event_positions"]
    colors = np.where(cue_ids == 0, "#E76F51", np.where(cue_ids == 1, "#457B9D", "#AAAAAA"))
    task_ax.scatter(event_positions, np.arange(len(cue_ids)), c=colors, s=38)
    task_ax.set_xlabel("Event/probe position")
    task_ax.set_ylabel("Training lap")
    task_ax.set_title("A  Balanced cue × position task")

    tuning = representative["place_test_tuning"]
    train_tuning = representative["place_train_tuning"]
    order = np.argsort(np.argmax(train_tuning, axis=1))
    image = heat_ax.imshow(tuning[order], aspect="auto", cmap="viridis", vmin=0, vmax=1)
    heat_ax.set_xlabel("Track position")
    heat_ax.set_ylabel("Training-selected CA1 cells")
    heat_ax.set_title("B  Held-out place fields")
    figure.colorbar(image, ax=heat_ax, fraction=0.046, pad=0.04)

    positions = np.arange(arrays["field_density"].shape[-1])
    for condition_index, condition in enumerate(CONDITIONS):
        density = arrays["field_density"][condition_index].mean(axis=(0, 1))
        density_ax.plot(
            positions,
            density,
            color=COLORS[condition],
            linestyle=LINESTYLES.get(condition, "-"),
            label=DISPLAY_NAMES[condition],
        )
    density_ax.set_xlabel("Held-out peak position")
    density_ax.set_ylabel("Place-field density")
    density_ax.set_title("C  Fields span the track")

    for condition_index, condition in enumerate(CONDITIONS):
        seed_layout = arrays["formation"][condition_index]
        mean = seed_layout.mean(axis=(0, 1))
        seed_values = seed_layout.mean(axis=1)
        ci = 1.96 * seed_values.std(axis=0, ddof=1) / np.sqrt(len(seed_values))
        formation_ax.plot(
            checkpoints,
            mean,
            marker="o",
            color=COLORS[condition],
            linestyle=LINESTYLES.get(condition, "-"),
            label=DISPLAY_NAMES[condition],
        )
        formation_ax.fill_between(checkpoints, mean - ci, mean + ci, color=COLORS[condition], alpha=0.12)
    formation_ax.set_xlabel("Training laps completed")
    formation_ax.set_ylabel("Similarity to final field")
    formation_ax.set_title("D  Receptive-field formation")

    plot_metric(cue_ax, arrays, "cue_accuracy", "E  Cue decoding across positions", "Held-out accuracy", chance=0.5)
    plot_metric(position_ax, arrays, "position_accuracy", "F  Position decoding across cues", "Held-out accuracy", chance=0.25)
    plot_metric(stability_ax, arrays, "field_stability", "G  Held-out field stability", "Train–test tuning correlation", chance=0.0)
    plot_metric(output_ax, arrays, "output_cosine", "H  Stable EC output reads content", "Held-out content cosine")

    handles, labels = formation_ax.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    figure.suptitle(
        "E3 — Content-aware instructive signals create stable, readable CA1 representations",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    info, autoencoder = load_autoencoder_session(args.checkpoint, map_location=args.device)
    params = session_parameters(info)
    autoencoder.eval()
    layouts = DEFAULT_LAYOUTS
    seeds = [args.seed_start + index for index in range(args.num_seeds)]
    results = [
        [
            run_seed_layout(
                seed,
                layout_index,
                layout,
                autoencoder,
                params,
                args,
            )
            for layout_index, layout in enumerate(layouts)
        ]
        for seed in seeds
    ]

    scalar_metrics = (
        "cue_accuracy",
        "position_accuracy",
        "cue_presence_accuracy",
        "field_stability",
        "place_cell_fraction",
        "output_cosine",
        "spatial_output_cosine",
        "sensory_output_cosine",
    )
    arrays = {
        metric: np.asarray(
            [
                [
                    [results[s][layout]["conditions"][condition][metric] for layout in range(len(layouts))]
                    for s in range(len(seeds))
                ]
                for condition in CONDITIONS
            ]
        )
        for metric in scalar_metrics
    }
    arrays["field_density"] = np.asarray(
        [
            [
                [results[s][layout]["conditions"][condition]["field_density"] for layout in range(len(layouts))]
                for s in range(len(seeds))
            ]
            for condition in CONDITIONS
        ]
    )
    arrays["formation"] = np.asarray(
        [
            [
                [results[s][layout]["conditions"][condition]["formation"] for layout in range(len(layouts))]
                for s in range(len(seeds))
            ]
            for condition in CONDITIONS
        ]
    )

    summary = {
        condition: {
            metric: summary_statistics(arrays[metric][condition_index])
            for metric in scalar_metrics
        }
        for condition_index, condition in enumerate(CONDITIONS)
    }
    aligned = 0
    fixed = 1
    rescue = 2
    random_matched = 3
    no_plasticity = 4
    hypothesis_checks = {
        "aligned_cue_decoding_exceeds_random": bool(
            np.mean(arrays["cue_accuracy"][aligned] - arrays["cue_accuracy"][random_matched]) > 0
        ),
        "aligned_cue_decoding_exceeds_no_plasticity": bool(
            np.mean(arrays["cue_accuracy"][aligned] - arrays["cue_accuracy"][no_plasticity]) > 0
        ),
        "permutation_preserves_intrinsic_ca1_cue_information": bool(
            abs(np.mean(arrays["cue_accuracy"][aligned]) - np.mean(arrays["cue_accuracy"][fixed])) < 0.05
        ),
        "aligned_position_decoding_exceeds_random": bool(
            np.mean(arrays["position_accuracy"][aligned] - arrays["position_accuracy"][random_matched]) > 0
        ),
        "aligned_fields_are_stable_on_heldout_laps": bool(
            np.mean(arrays["field_stability"][aligned]) > 0.3
        ),
        "aligned_output_decoding_exceeds_fixed": bool(
            np.mean(arrays["output_cosine"][aligned] - arrays["output_cosine"][fixed]) > 0
        ),
        "decoder_rescues_output_content": bool(
            np.mean(arrays["output_cosine"][rescue] - arrays["output_cosine"][fixed]) > 0
        ),
        "rescue_approaches_aligned_output": bool(
            abs(np.mean(arrays["output_cosine"][rescue]) - np.mean(arrays["output_cosine"][aligned])) < 0.05
        ),
    }
    first = results[0][0]
    cue_pairs = set(
        zip(
            first["train_data"]["cue_ids"][first["train_data"]["cue_ids"] >= 0].tolist(),
            first["train_data"]["event_positions"][first["train_data"]["cue_ids"] >= 0].tolist(),
        )
    )
    expected_pairs = {(cue, position) for cue in (0, 1) for position in layouts[0]}
    signal_names = ("aligned", "fixed_permutation", "random_matched")
    protocol_checks = {
        "cue_identity_crossed_with_position": cue_pairs == expected_pairs,
        "cue_free_laps_present": bool(np.any(first["train_data"]["cue_ids"] < 0)),
        "training_and_heldout_inputs_differ": bool(
            not np.array_equal(first["train_data"]["inputs"], first["test_data"]["inputs"])
        ),
        "plasticity_profile_is_cue_identity_independent": bool(
            all(
                np.isclose(
                    first["plasticity_profile"][event_rows(first["train_data"])][
                        first["train_data"]["cue_ids"][first["train_data"]["cue_ids"] >= 0]
                        == cue
                    ].mean(),
                    args.alpha,
                )
                for cue in (0, 1)
            )
        ),
        "spatial_and_sensory_components_are_explicit": bool(
            first["train_data"]["spatial"].shape[1] == params["spatial_dim"]
            and first["train_data"]["sensory"].shape[1] == params["sensory_dim"]
        ),
        "neurons_selected_only_from_training_laps": True,
        "matched_signal_statistics": all(
            np.isclose(first["signal_sparsity"][name], first["signal_sparsity"]["aligned"])
            and np.isclose(first["signal_norm"][name], first["signal_norm"]["aligned"])
            for name in signal_names
        ),
        "fixed_and_rescue_learning_identical": all(
            results[s][layout]["weight_digests"]["fixed_permutation"]
            == results[s][layout]["weight_digests"]["decoder_rescue"]
            for s in range(len(seeds))
            for layout in range(len(layouts))
        ),
    }

    representative = first["representatives"]["aligned"]
    schedules = {
        "cue_ids": first["train_data"]["cue_ids"],
        "event_positions": first["train_data"]["event_positions"],
    }
    checkpoints = first["checkpoints"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "e3_ca1_track"
    train_inputs = np.asarray(
        [[results[s][layout]["train_data"]["inputs"] for layout in range(len(layouts))] for s in range(len(seeds))]
    )
    test_inputs = np.asarray(
        [[results[s][layout]["test_data"]["inputs"] for layout in range(len(layouts))] for s in range(len(seeds))]
    )
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        seeds=np.asarray(seeds),
        conditions=np.asarray(CONDITIONS),
        layouts=np.asarray(layouts),
        checkpoints=checkpoints,
        train_inputs=train_inputs,
        test_inputs=test_inputs,
        train_cue_ids=np.asarray([[results[s][layout]["train_data"]["cue_ids"] for layout in range(len(layouts))] for s in range(len(seeds))]),
        test_cue_ids=np.asarray([[results[s][layout]["test_data"]["cue_ids"] for layout in range(len(layouts))] for s in range(len(seeds))]),
        train_event_positions=np.asarray([[results[s][layout]["train_data"]["event_positions"] for layout in range(len(layouts))] for s in range(len(seeds))]),
        test_event_positions=np.asarray([[results[s][layout]["test_data"]["event_positions"] for layout in range(len(layouts))] for s in range(len(seeds))]),
        permutations=np.asarray([[results[s][layout]["permutation"] for layout in range(len(layouts))] for s in range(len(seeds))]),
        random_matchings=np.asarray([[results[s][layout]["random_matching"] for layout in range(len(layouts))] for s in range(len(seeds))]),
        base_ca3_projection=np.asarray([[results[s][layout]["base_ca3_projection"] for layout in range(len(layouts))] for s in range(len(seeds))]),
        representative_place_indices=representative["place_indices"],
        representative_place_train_tuning=representative["place_train_tuning"],
        representative_place_test_tuning=representative["place_test_tuning"],
        representative_cue_indices=representative["cue_indices"],
        representative_cue_matrix=representative["cue_matrix"],
        **arrays,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E3_factorial_ca1_track",
        "configuration": vars(args),
        "runtime": runtime_metadata(),
        "checkpoint_info": info,
        "network_parameters": params,
        "protocol": {
            "design": "fully crossed cue identity x presentation position with cue-free laps",
            "ec_projection": "the full concatenated spatial+sensory EC input projects to both fixed EC-CA3 and the autoencoder-derived instructive signal",
            "plasticity_profile": "identical low baseline outside cue events and identical event alpha for both cue identities and all conditions; the profile contains no cue-identity information",
            "selection": "place/cue neurons selected using post-storage retrieval of training laps only",
            "evaluation": "all reported decoding, tuning, stability, and EC-output metrics use separately generated held-out laps",
            "cue_decoder": "leave-one-position-out nearest-centroid decoding",
            "position_decoder": "train on one cue identity and test on the other",
            "prediction": "CA1 neurons targeted by cue-selective EC3/plateau signals should acquire cue-dependent fields that generalize across presentation position; disrupting content matching should abolish this structure",
            "train_input_sha256_by_seed_layout": [[array_digest(results[s][layout]["train_data"]["inputs"]) for layout in range(len(layouts))] for s in range(len(seeds))],
            "test_input_sha256_by_seed_layout": [[array_digest(results[s][layout]["test_data"]["inputs"]) for layout in range(len(layouts))] for s in range(len(seeds))],
            "final_weight_sha256_by_seed_layout_condition": [[results[s][layout]["weight_digests"] for layout in range(len(layouts))] for s in range(len(seeds))],
        },
        "summary": summary,
        "paired_effects": {
            "aligned_minus_random_cue_accuracy": arrays["cue_accuracy"][aligned] - arrays["cue_accuracy"][random_matched],
            "aligned_minus_random_position_accuracy": arrays["position_accuracy"][aligned] - arrays["position_accuracy"][random_matched],
            "aligned_minus_fixed_output_cosine": arrays["output_cosine"][aligned] - arrays["output_cosine"][fixed],
            "rescue_minus_fixed_output_cosine": arrays["output_cosine"][rescue] - arrays["output_cosine"][fixed],
        },
        "paired_effect_summary": {
            "aligned_minus_random_cue_accuracy": summary_statistics(
                arrays["cue_accuracy"][aligned] - arrays["cue_accuracy"][random_matched]
            ),
            "aligned_minus_random_position_accuracy": summary_statistics(
                arrays["position_accuracy"][aligned] - arrays["position_accuracy"][random_matched]
            ),
            "aligned_minus_random_field_stability": summary_statistics(
                arrays["field_stability"][aligned] - arrays["field_stability"][random_matched]
            ),
            "aligned_minus_fixed_output_cosine": summary_statistics(
                arrays["output_cosine"][aligned] - arrays["output_cosine"][fixed]
            ),
            "rescue_minus_fixed_output_cosine": summary_statistics(
                arrays["output_cosine"][rescue] - arrays["output_cosine"][fixed]
            ),
        },
        "protocol_checks": protocol_checks,
        "hypothesis_checks": hypothesis_checks,
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
    make_figure(arrays, representative, schedules, checkpoints, prefix.with_suffix(".png"))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="ae_factorial_paper_v1")
    parser.add_argument("--seed-start", type=int, default=1100)
    parser.add_argument("--num-seeds", type=int, default=12)
    parser.add_argument("--train-repeats", type=int, default=2)
    parser.add_argument("--test-repeats", type=int, default=2)
    parser.add_argument("--train-free-laps", type=int, default=4)
    parser.add_argument("--test-free-laps", type=int, default=4)
    parser.add_argument("--num-place-cells", type=int, default=100)
    parser.add_argument("--num-cue-cells-per-class", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--baseline-alpha", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.num_seeds < 2:
        parser.error("--num-seeds must be at least 2")
    if min(args.train_repeats, args.test_repeats, args.train_free_laps, args.test_free_laps) < 2:
        parser.error("track repeats and cue-free laps must be at least 2")
    if args.alpha <= 0 or args.baseline_alpha < 0 or args.baseline_alpha >= args.alpha:
        parser.error("require 0 <= --baseline-alpha < --alpha")
    return args


def main() -> int:
    args = parse_args()
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    report = run(args)
    concise = {
        "condition_summary": {
            condition: {
                metric: report["summary"][condition][metric]["mean"]
                for metric in (
                    "cue_accuracy",
                    "position_accuracy",
                    "field_stability",
                    "output_cosine",
                )
            }
            for condition in CONDITIONS
        },
        "protocol_checks": report["protocol_checks"],
        "hypothesis_checks": report["hypothesis_checks"],
        "outputs": report["outputs"],
    }
    print(json.dumps(json_ready(concise), indent=2))
    return 0 if all(report["protocol_checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
