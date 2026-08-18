"""Fast backend diagnostics runnable before starting a long simulation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile

import torch

from .data import generate_sparse_patterns
from .io import PATHS, check_project_paths, load_autoencoder_session, save_autoencoder_session
from .models import Autoencoder, BTSPMemory
from .training import evaluate_autoencoder, evaluate_memory, fit_autoencoder, store_patterns
from .utils import seed_everything


def smoke_test_training(seed: int = 7) -> dict[str, object]:
    """Run a tiny AE + BTSP training pass and check mutation boundaries."""

    rng = seed_everything(seed)
    patterns = generate_sparse_patterns(24, 3, 12, rng=rng)
    model = Autoencoder(input_dim=12, encoding_dim=12, K=5, beta=20)
    initial_loss = evaluate_autoencoder(model, patterns)
    history = fit_autoencoder(
        model,
        patterns,
        patterns,
        epochs=4,
        batch_size=8,
        learning_rate=1e-2,
    )
    final_loss = evaluate_autoencoder(model, patterns)

    memory = BTSPMemory.from_autoencoder(
        model,
        K_lat=5,
        K_out=3,
        dim_ca3=12,
        K_ca3=5,
        beta=20,
        alpha=0.2,
    )
    before_retrieval = memory.W_ca3_ca1.clone()
    memory(patterns[0])
    retrieval_is_read_only = torch.equal(before_retrieval, memory.W_ca3_ca1)
    store_patterns(memory, patterns[:4])
    storage_changed_weights = not torch.equal(before_retrieval, memory.W_ca3_ca1)
    memory_result = evaluate_memory(memory, patterns[:4])

    finite = all(
        math.isfinite(value)
        for value in (initial_loss, final_loss, memory_result["mse"])
    )
    return {
        "ok": finite and retrieval_is_read_only and storage_changed_weights,
        "autoencoder_initial_mse": initial_loss,
        "autoencoder_final_mse": final_loss,
        "epochs": len(history.train_loss),
        "retrieval_is_read_only": retrieval_is_read_only,
        "storage_changed_weights": storage_changed_weights,
        "memory_mse": memory_result["mse"],
    }


def smoke_test_checkpoint(seed: int = 7) -> dict[str, object]:
    """Round-trip a model through a temporary checkpoint directory."""

    seed_everything(seed)
    model = Autoencoder(input_dim=6, encoding_dim=5, K=2, beta=10)
    info = {"dim_ei": 6, "dim_ca1": 5, "K_lat": 2, "beta": 10, "bias": True}
    with tempfile.TemporaryDirectory() as temp_dir:
        session = save_autoencoder_session(
            model, info, name="ae_smoke", directory=Path(temp_dir)
        )
        loaded_info, loaded = load_autoencoder_session(
            session, directory=Path(temp_dir)
        )
        equal = all(
            torch.equal(left, right)
            for left, right in zip(model.state_dict().values(), loaded.state_dict().values())
        )
    return {"ok": equal and loaded_info == info, "state_dict_equal": equal}


def run_backend_checks() -> dict[str, object]:
    paths = check_project_paths(PATHS)
    required = ("root", "source", "configs", "data", "autoencoders", "experiments")
    paths_ok = all(paths[name]["is_dir"] for name in required)
    training = smoke_test_training()
    checkpoint = smoke_test_checkpoint()
    return {
        "ok": paths_ok and training["ok"] and checkpoint["ok"],
        "paths": paths,
        "training": training,
        "checkpoint": checkpoint,
    }


def main() -> int:
    report = run_backend_checks()
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

