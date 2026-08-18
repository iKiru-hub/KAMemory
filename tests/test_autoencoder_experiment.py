from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from kamemory.autoencoder_experiment import (
    make_factorial_autoencoder_datasets,
    normalize_autoencoder_settings,
    run_autoencoder_experiment,
)
from kamemory.io import load_autoencoder_session


def tiny_settings(directory: Path | None = None) -> dict:
    return {
        "seed": 17,
        "deterministic": True,
        "device": "cpu",
        "verbose": False,
        "network_params": {
            "mec_N_x": 8,
            "mec_N_y": 1,
            "dim_mec": 8,
            "mec_sigma": 1.5,
            "dim_lec": 8,
            "num_cues": 2,
            "bias": False,
            "dim_ei": 16,
            "dim_ca3": 16,
            "dim_ca1": 16,
            "dim_eo": 16,
            "K_lec": 2,
            "K_ei": 4,
            "K_ca3": 4,
            "K_ca1": 4,
            "K_eo": 4,
            "beta_ei": 20.0,
            "beta_ca3": 20.0,
            "beta_ca1": 20.0,
            "beta_eo": 20.0,
            "alpha": 0.2,
        },
        "dataset": {
            "layouts": [[1, 4]],
            "train_sessions_per_layout": 1,
            "validation_sessions_per_layout": 1,
            "test_sessions_per_layout": 1,
            "train_repeats_per_combination": 1,
            "validation_repeats_per_combination": 1,
            "test_repeats_per_combination": 1,
            "train_cue_free_laps": 1,
            "validation_cue_free_laps": 1,
            "test_cue_free_laps": 1,
        },
        "training": {
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 0.01,
            "early_stopping_patience": None,
            "log_every": 1,
        },
        "evaluation": {
            "batch_size": 16,
            "reference_checkpoint": None,
        },
        "save": {
            "enabled": directory is not None,
            "name": "tiny_factorial",
            "directory": None if directory is None else str(directory),
        },
    }


class AutoencoderExperimentTests(unittest.TestCase):
    def test_flat_training_aliases_are_supported(self):
        settings = normalize_autoencoder_settings(
            {"epochs": 3, "lr": 0.02, "batch_size": 7, "save_name": "alias"}
        )
        self.assertEqual(settings["training"]["epochs"], 3)
        self.assertEqual(settings["training"]["learning_rate"], 0.02)
        self.assertEqual(settings["training"]["batch_size"], 7)
        self.assertEqual(settings["save"]["name"], "alias")

    def test_factorial_splits_are_reproducible_and_independent(self):
        first = make_factorial_autoencoder_datasets(tiny_settings())
        second = make_factorial_autoencoder_datasets(tiny_settings())
        for split in ("train", "validation", "test"):
            self.assertTrue(np.array_equal(first[split], second[split]))
        self.assertNotEqual(
            first["summary"]["sha256"]["train"],
            first["summary"]["sha256"]["test"],
        )

    def test_end_to_end_checkpoint_can_be_reloaded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_autoencoder_experiment(tiny_settings(Path(temp_dir)))
            session = result["session_path"]
            self.assertIsNotNone(session)
            self.assertTrue((session / "autoencoder.pt").is_file())
            self.assertTrue((session / "info.json").is_file())
            info, loaded = load_autoencoder_session(session)
            self.assertEqual(info["experiment"], "factorial_track_autoencoder_training")
            for expected, observed in zip(
                result["model"].state_dict().values(),
                loaded.state_dict().values(),
            ):
                self.assertTrue(torch.equal(expected.cpu(), observed.cpu()))
            metrics = result["report"]["metrics"]["trained"]["test"]
            self.assertTrue(np.isfinite(metrics["mse"]))
            self.assertEqual(len(result["history"]["train_loss"]), 2)


if __name__ == "__main__":
    unittest.main()
