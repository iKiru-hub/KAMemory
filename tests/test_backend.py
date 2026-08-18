from pathlib import Path
import os
import tempfile
import unittest

import torch

from kamemory.checks import smoke_test_checkpoint, smoke_test_training
from kamemory.io import PATHS, check_project_paths, load_config
from kamemory.models import Autoencoder, BTSPMemory


class BackendSmokeTests(unittest.TestCase):
    def test_paths_do_not_depend_on_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                report = check_project_paths()
                config = load_config("base_configs.json")
            finally:
                os.chdir(original)
        self.assertEqual(Path(report["root"]["path"]), PATHS.root)
        self.assertTrue(report["configs"]["is_dir"])
        self.assertIn("hyperparameters", config)

    def test_training_boundaries(self):
        report = smoke_test_training()
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["retrieval_is_read_only"])
        self.assertTrue(report["storage_changed_weights"])

    def test_checkpoint_roundtrip(self):
        self.assertTrue(smoke_test_checkpoint()["ok"])

    def test_explicit_instructive_signal_matches_update_equation(self):
        torch.manual_seed(3)
        autoencoder = Autoencoder(input_dim=6, encoding_dim=6, K=2, beta=10)
        memory = BTSPMemory.from_autoencoder(
            autoencoder,
            K_lat=2,
            K_out=2,
            dim_ca3=6,
            K_ca3=2,
            beta=10,
            alpha=0.25,
        )
        pattern = torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.float32)
        signal = torch.tensor([0, 1, 0, 1, 0, 0], dtype=torch.float32).reshape(-1, 1)
        ca3 = memory.ca3_activity(pattern)
        before = memory.W_ca3_ca1.clone()
        expected = (1 - 0.25 * signal) * before + 0.25 * (signal @ ca3.T)
        memory.store(pattern, instructive_signal=signal)
        self.assertTrue(torch.equal(expected, memory.W_ca3_ca1))


if __name__ == "__main__":
    unittest.main()
