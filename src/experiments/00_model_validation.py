"""E0: validate the KAMemory backend and generate a diagnostic figure.

Run from the repository root after installing the package in editable mode:

    python3 src/experiments/00_model_validation.py

The experiment is deliberately self-contained and does not read notebook state
or previously generated panels. It writes a JSON report and a PNG/PDF figure to
``src/experiments/plots`` by default.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kamemory-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from kamemory.data import generate_legacy_sparse_patterns
from kamemory.io import (
    check_project_paths,
    load_autoencoder_session,
    load_config,
    runtime_metadata,
    save_autoencoder_session,
)
from kamemory.models import Autoencoder, BTSPMemory
from kamemory.training import evaluate_autoencoder, fit_autoencoder
from kamemory.utils import array_digest, seed_everything


SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


@contextmanager
def temporary_working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def active_fraction(activity: torch.Tensor, threshold: float = 0.5) -> float:
    return float((activity > threshold).float().mean().item())


def top_k_scores(
    targets: np.ndarray, predictions: np.ndarray, num_active: int
) -> dict[str, float]:
    """Measure recovery of active input units without rewarding true negatives."""

    target_active = targets > 0.5
    predicted_active = np.zeros_like(target_active, dtype=bool)
    top_indices = np.argpartition(predictions, -num_active, axis=1)[:, -num_active:]
    rows = np.arange(len(predictions))[:, None]
    predicted_active[rows, top_indices] = True
    true_positive = np.logical_and(target_active, predicted_active).sum(axis=1)
    precision = true_positive / np.maximum(predicted_active.sum(axis=1), 1)
    recall = true_positive / np.maximum(target_active.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }


def make_autoencoder(
    input_dim: int,
    encoding_dim: int,
    latent_active: int,
    beta: float,
) -> Autoencoder:
    return Autoencoder(
        input_dim=input_dim,
        encoding_dim=encoding_dim,
        K=latent_active,
        beta=beta,
        use_bias=True,
    )


def make_memory(
    autoencoder: Autoencoder,
    *,
    input_active: int,
    latent_active: int,
    dim_ca3: int,
    ca3_active: int,
    beta: float,
    alpha: float,
) -> BTSPMemory:
    return BTSPMemory.from_autoencoder(
        autoencoder,
        K_lat=latent_active,
        K_out=input_active,
        dim_ca3=dim_ca3,
        K_ca3=ca3_active,
        beta=beta,
        alpha=alpha,
    )


def check_hand_computed_update(
    memory: BTSPMemory, pattern: torch.Tensor
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compare one model update with the equation evaluated independently."""

    pattern = pattern.reshape(-1, 1).to(memory.W_ca3_ca1)
    before = memory.W_ca3_ca1.clone()
    ca3 = memory.ca3_activity(pattern)
    instructive_signal = memory.instructive_signal(pattern)
    expected = (1 - memory._alpha * instructive_signal) * before + memory._alpha * (
        instructive_signal @ ca3.T
    )
    memory.store(pattern)
    observed = memory.W_ca3_ca1.clone()
    maximum_error = float(torch.max(torch.abs(expected - observed)).item())
    return maximum_error, expected.cpu().numpy(), observed.cpu().numpy()


def check_alpha_zero(memory: BTSPMemory, pattern: torch.Tensor) -> bool:
    memory.set_alpha(0.0)
    before = memory.W_ca3_ca1.clone()
    memory.store(pattern)
    return torch.equal(before, memory.W_ca3_ca1)


def reproducibility_signature(
    seed: int,
    *,
    input_dim: int,
    input_active: int,
    latent_active: int,
    beta: float,
) -> tuple[np.ndarray, dict[str, torch.Tensor], np.ndarray]:
    """Repeat a minimal end-to-end run for exact seed comparison."""

    seed_everything(seed)
    patterns = generate_legacy_sparse_patterns(16, input_active, input_dim)
    model = make_autoencoder(input_dim, input_dim, latent_active, beta)
    fit_autoencoder(
        model,
        patterns,
        patterns,
        epochs=3,
        batch_size=8,
        learning_rate=1e-2,
    )
    with torch.no_grad():
        outputs = model(torch.as_tensor(patterns)).cpu().numpy()
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return patterns, state, outputs


def check_reproducibility(**kwargs) -> bool:
    first_patterns, first_state, first_outputs = reproducibility_signature(**kwargs)
    second_patterns, second_state, second_outputs = reproducibility_signature(**kwargs)
    same_state = all(
        torch.equal(first_state[name], second_state[name]) for name in first_state
    )
    return (
        np.array_equal(first_patterns, second_patterns)
        and same_state
        and np.array_equal(first_outputs, second_outputs)
    )


def check_paths_and_checkpoint(model: Autoencoder) -> dict[str, object]:
    """Test configuration and checkpoint IO from outside the repository cwd."""

    path_report = check_project_paths()
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        checkpoint_root = temporary_path / "checkpoints"
        info = {
            "dim_ei": model._input_dim,
            "dim_ca1": model._encoding_dim,
            "K_lat": model._K,
            "beta": model._beta,
            "bias": model._use_bias,
        }
        session = save_autoencoder_session(
            model,
            info,
            name="ae_e0_roundtrip",
            directory=checkpoint_root,
        )
        with temporary_working_directory(temporary_path):
            config = load_config("base_configs.json")
            loaded_info, loaded_model = load_autoencoder_session(
                session, directory=checkpoint_root
            )

        checkpoint_equal = loaded_info == info and all(
            torch.equal(source, loaded)
            for source, loaded in zip(
                model.state_dict().values(), loaded_model.state_dict().values()
            )
        )

    required_paths = ("root", "source", "configs", "data", "autoencoders", "experiments")
    paths_ok = all(path_report[name]["is_dir"] for name in required_paths)
    return {
        "ok": bool(paths_ok and checkpoint_equal and "hyperparameters" in config),
        "project_paths": path_report,
        "checkpoint_roundtrip": bool(checkpoint_equal),
        "config_loaded_outside_project": "hyperparameters" in config,
    }


def storage_diagnostics(
    memory: BTSPMemory, patterns: np.ndarray
) -> dict[str, np.ndarray | float | bool]:
    weight_minimum = []
    weight_mean = []
    weight_maximum = []
    ca3_sparsity = []
    instructive_sparsity = []
    ca1_sparsity = []

    for pattern_array in patterns:
        pattern = torch.as_tensor(pattern_array).reshape(-1, 1)
        ca3 = memory.ca3_activity(pattern)
        signal = memory.instructive_signal(pattern)
        memory.store(pattern)
        ca1 = memory.ca1_activity(ca3)
        weights = memory.W_ca3_ca1
        weight_minimum.append(float(weights.min().item()))
        weight_mean.append(float(weights.mean().item()))
        weight_maximum.append(float(weights.max().item()))
        ca3_sparsity.append(active_fraction(ca3))
        instructive_sparsity.append(active_fraction(signal))
        ca1_sparsity.append(active_fraction(ca1))

    all_values = np.concatenate(
        [weight_minimum, weight_mean, weight_maximum, ca3_sparsity, instructive_sparsity, ca1_sparsity]
    )
    weights_bounded = min(weight_minimum) >= -1e-7 and max(weight_maximum) <= 1 + 1e-7
    mean_sparsities = {
        "ca3": float(np.mean(ca3_sparsity)),
        "instructive_signal": float(np.mean(instructive_sparsity)),
        "ca1": float(np.mean(ca1_sparsity)),
    }
    sparsity_sensible = all(0 < value < 0.75 for value in mean_sparsities.values())
    return {
        "finite": bool(np.isfinite(all_values).all()),
        "weights_bounded_0_1": bool(weights_bounded),
        "activity_sparsity_sensible": bool(sparsity_sensible),
        "mean_sparsity": mean_sparsities,
        "weight_minimum": np.asarray(weight_minimum),
        "weight_mean": np.asarray(weight_mean),
        "weight_maximum": np.asarray(weight_maximum),
        "ca3_sparsity": np.asarray(ca3_sparsity),
        "instructive_sparsity": np.asarray(instructive_sparsity),
        "ca1_sparsity": np.asarray(ca1_sparsity),
    }


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
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def make_validation_figure(
    *,
    history,
    examples: np.ndarray,
    reconstructions: np.ndarray,
    expected_update: np.ndarray,
    observed_update: np.ndarray,
    storage: dict,
    checks: dict[str, bool],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    ax_learning, ax_examples, ax_update, ax_weights, ax_sparsity, ax_checks = axes.flat

    epochs = np.arange(1, len(history.train_loss) + 1)
    ax_learning.plot(epochs, history.train_loss, label="train", linewidth=2)
    ax_learning.plot(epochs, history.validation_loss, label="held-out", linewidth=2)
    ax_learning.set_yscale("log")
    ax_learning.set_xlabel("Epoch")
    ax_learning.set_ylabel("MSE")
    ax_learning.set_title("A  Autoencoder learning")
    ax_learning.legend(frameon=False)

    example_image = np.vstack((examples, np.full((1, examples.shape[1]), np.nan), reconstructions))
    image = ax_examples.imshow(example_image, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    separator = examples.shape[0] - 0.5
    ax_examples.axhline(separator + 1, color="white", linewidth=2)
    ax_examples.set_yticks(
        [examples.shape[0] / 2 - 0.5, examples.shape[0] * 1.5 + 0.5],
        ["targets", "reconstructions"],
    )
    ax_examples.set_xlabel("EC unit")
    ax_examples.set_title("B  Held-out reconstruction")
    figure.colorbar(image, ax=ax_examples, fraction=0.046, pad=0.04)

    ax_update.scatter(
        expected_update.reshape(-1),
        observed_update.reshape(-1),
        s=8,
        alpha=0.35,
        edgecolors="none",
    )
    limit = max(float(expected_update.max()), float(observed_update.max()), 1e-6)
    ax_update.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    ax_update.set_xlabel("Hand-computed weight")
    ax_update.set_ylabel("Implemented weight")
    ax_update.set_title("C  BTSP update equivalence")
    ax_update.set_aspect("equal", adjustable="box")

    storage_events = np.arange(1, len(storage["weight_mean"]) + 1)
    ax_weights.fill_between(
        storage_events,
        storage["weight_minimum"],
        storage["weight_maximum"],
        alpha=0.2,
        label="min–max",
    )
    ax_weights.plot(storage_events, storage["weight_mean"], linewidth=2, label="mean")
    ax_weights.set_xlabel("Stored memories")
    ax_weights.set_ylabel("CA3→CA1 weight")
    ax_weights.set_ylim(-0.02, 1.02)
    ax_weights.set_title("D  Synaptic evolution")
    ax_weights.legend(frameon=False)

    ax_sparsity.plot(storage_events, storage["ca3_sparsity"], label="CA3", linewidth=2)
    ax_sparsity.plot(
        storage_events, storage["instructive_sparsity"], label="IS", linewidth=2
    )
    ax_sparsity.plot(storage_events, storage["ca1_sparsity"], label="CA1", linewidth=2)
    ax_sparsity.set_xlabel("Stored memories")
    ax_sparsity.set_ylabel("Fraction activity > 0.5")
    ax_sparsity.set_ylim(-0.02, 1.02)
    ax_sparsity.set_title("E  Population sparsity")
    ax_sparsity.legend(frameon=False, ncol=3)

    labels = list(checks)
    values = np.asarray([checks[label] for label in labels], dtype=float)
    colors = ["#2a9d8f" if value else "#e76f51" for value in values]
    positions = np.arange(len(labels))
    ax_checks.barh(positions, values, color=colors)
    ax_checks.set_yticks(positions, [label.replace("_", " ") for label in labels])
    ax_checks.set_xlim(0, 1.05)
    ax_checks.set_xticks([0, 1], ["fail", "pass"])
    ax_checks.invert_yaxis()
    ax_checks.set_title("F  Validation summary")
    for position, passed in zip(positions, values):
        ax_checks.text(0.5, position, "PASS" if passed else "FAIL", ha="center", va="center", color="white", fontweight="bold")

    figure.suptitle("E0 — KAMemory model validation", fontsize=16, fontweight="bold")
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed, deterministic=args.deterministic)
    training_patterns = generate_legacy_sparse_patterns(
        args.num_training_patterns,
        args.input_active,
        args.input_dim,
    )
    held_out_patterns = generate_legacy_sparse_patterns(
        args.num_test_patterns,
        args.input_active,
        args.input_dim,
    )

    autoencoder = make_autoencoder(
        args.input_dim, args.encoding_dim, args.latent_active, args.beta
    )
    initial_mse = evaluate_autoencoder(autoencoder, held_out_patterns)
    history = fit_autoencoder(
        autoencoder,
        training_patterns,
        held_out_patterns,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    final_mse = evaluate_autoencoder(autoencoder, held_out_patterns, device=args.device)
    autoencoder.eval()
    with torch.no_grad():
        reconstructions = autoencoder(
            torch.as_tensor(held_out_patterns, device=args.device)
        ).cpu().numpy()
    recovery = top_k_scores(held_out_patterns, reconstructions, args.input_active)

    update_memory = make_memory(
        autoencoder,
        input_active=args.input_active,
        latent_active=args.latent_active,
        dim_ca3=args.dim_ca3,
        ca3_active=args.ca3_active,
        beta=args.beta,
        alpha=args.alpha,
    ).to(args.device)
    update_error, expected_update, observed_update = check_hand_computed_update(
        update_memory, torch.as_tensor(training_patterns[0], device=args.device)
    )

    zero_memory = make_memory(
        autoencoder,
        input_active=args.input_active,
        latent_active=args.latent_active,
        dim_ca3=args.dim_ca3,
        ca3_active=args.ca3_active,
        beta=args.beta,
        alpha=args.alpha,
    ).to(args.device)
    alpha_zero_unchanged = check_alpha_zero(
        zero_memory, torch.as_tensor(training_patterns[0], device=args.device)
    )

    retrieval_memory = make_memory(
        autoencoder,
        input_active=args.input_active,
        latent_active=args.latent_active,
        dim_ca3=args.dim_ca3,
        ca3_active=args.ca3_active,
        beta=args.beta,
        alpha=args.alpha,
    ).to(args.device)
    retrieval_memory.store(torch.as_tensor(training_patterns[0], device=args.device))
    before_retrieval = retrieval_memory.W_ca3_ca1.clone()
    retrieval_memory.retrieve(torch.as_tensor(training_patterns[0], device=args.device))
    retrieval_is_read_only = torch.equal(before_retrieval, retrieval_memory.W_ca3_ca1)

    reproducible = check_reproducibility(
        seed=args.seed,
        input_dim=args.input_dim,
        input_active=args.input_active,
        latent_active=args.latent_active,
        beta=args.beta,
    )
    path_checkpoint = check_paths_and_checkpoint(autoencoder.cpu())

    diagnostic_memory = make_memory(
        autoencoder,
        input_active=args.input_active,
        latent_active=args.latent_active,
        dim_ca3=args.dim_ca3,
        ca3_active=args.ca3_active,
        beta=args.beta,
        alpha=args.alpha,
    )
    storage = storage_diagnostics(
        diagnostic_memory, training_patterns[: args.num_storage_patterns]
    )

    chance_f1 = args.input_active / args.input_dim
    checks = {
        "AE_loss_improves": bool(final_mse < initial_mse),
        "CA1_code_decodable": bool(recovery["f1"] > chance_f1),
        "retrieve_read_only": bool(retrieval_is_read_only),
        "update_matches_equation": bool(update_error <= args.update_tolerance),
        "alpha_zero_unchanged": bool(alpha_zero_unchanged),
        "seed_reproducible": bool(reproducible),
        "paths_and_checkpoint": bool(path_checkpoint["ok"]),
        "weights_and_activity_finite": bool(storage["finite"]),
        "weights_bounded": bool(storage["weights_bounded_0_1"]),
        "activity_sparsity_sensible": bool(storage["activity_sparsity_sensible"]),
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E0_model_validation",
        "passed": bool(all(checks.values())),
        "configuration": vars(args),
        "runtime": runtime_metadata(),
        "data": {
            "protocol": "validated_notebook_rowwise_numpy_choice",
            "training_sha256": array_digest(training_patterns),
            "held_out_sha256": array_digest(held_out_patterns),
        },
        "metrics": {
            "autoencoder_initial_mse": initial_mse,
            "autoencoder_final_mse": final_mse,
            "top_k": recovery,
            "top_k_chance": chance_f1,
            "maximum_update_error": update_error,
        },
        "checks": checks,
        "paths_and_checkpoint": path_checkpoint,
        "storage": storage,
        "training_history": {
            "train_loss": history.train_loss,
            "validation_loss": history.validation_loss,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "e0_model_validation_metrics.json"
    figure_path = args.output_dir / "e0_model_validation.png"
    raw_path = args.output_dir / "e0_model_validation.npz"
    np.savez_compressed(
        raw_path,
        training_patterns=training_patterns,
        held_out_patterns=held_out_patterns,
        held_out_reconstructions=reconstructions,
        train_loss=np.asarray(history.train_loss),
        validation_loss=np.asarray(history.validation_loss),
        weight_minimum=storage["weight_minimum"],
        weight_mean=storage["weight_mean"],
        weight_maximum=storage["weight_maximum"],
        ca3_sparsity=storage["ca3_sparsity"],
        instructive_sparsity=storage["instructive_sparsity"],
        ca1_sparsity=storage["ca1_sparsity"],
    )
    report["outputs"] = {
        "metrics": str(metrics_path),
        "raw_npz": str(raw_path),
        "figure_png": str(figure_path),
        "figure_pdf": str(figure_path.with_suffix(".pdf")),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(report), handle, indent=2, sort_keys=True)
        handle.write("\n")
    make_validation_figure(
        history=history,
        examples=held_out_patterns[: args.num_examples],
        reconstructions=reconstructions[: args.num_examples],
        expected_update=expected_update,
        observed_update=observed_update,
        storage=storage,
        checks=checks,
        output_path=figure_path,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--encoding-dim", type=int, default=32)
    parser.add_argument("--dim-ca3", type=int, default=32)
    parser.add_argument("--input-active", type=int, default=4)
    parser.add_argument("--latent-active", type=int, default=10)
    parser.add_argument("--ca3-active", type=int, default=10)
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--num-training-patterns", type=int, default=160)
    parser.add_argument("--num-test-patterns", type=int, default=64)
    parser.add_argument("--num-storage-patterns", type=int, default=40)
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--update-tolerance", type=float, default=1e-7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.encoding_dim != args.input_dim:
        parser.error("E0 currently requires --encoding-dim to equal --input-dim")
    if args.num_examples > args.num_test_patterns:
        parser.error("--num-examples cannot exceed --num-test-patterns")
    if args.num_storage_patterns > args.num_training_patterns:
        parser.error("--num-storage-patterns cannot exceed --num-training-patterns")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    concise = {
        "passed": report["passed"],
        "checks": report["checks"],
        "metrics": report["metrics"],
        "outputs": report["outputs"],
    }
    print(json.dumps(json_ready(concise), indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
