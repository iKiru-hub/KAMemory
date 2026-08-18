"""Numerical contract between the exploratory and refactored backends."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kamemory-matplotlib")
)
import numpy as np
import torch

from kamemory.data import generate_legacy_sparse_patterns
from kamemory.models import Autoencoder, BTSPMemory
from kamemory.training import fit_autoencoder


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
import models as legacy_models
import training as legacy_training
import utils as legacy_utils


class LegacyParityTests(unittest.TestCase):
    def test_sparse_dataset_matches_notebook_generator(self):
        np.random.seed(41)
        expected = legacy_utils.sparse_stimulus_generator(
            N=24, K=4, size=20, plot=False
        )
        np.random.seed(41)
        observed = generate_legacy_sparse_patterns(24, 4, 20)
        self.assertTrue(np.array_equal(expected, observed))

    def test_autoencoder_training_matches_legacy_loop(self):
        np.random.seed(12)
        torch.manual_seed(12)
        training_data = generate_legacy_sparse_patterns(32, 4, 20)
        test_data = generate_legacy_sparse_patterns(16, 4, 20)
        legacy = legacy_models.Autoencoder(
            input_dim=20, encoding_dim=20, K=7, beta=30
        )
        refactored = Autoencoder(input_dim=20, encoding_dim=20, K=7, beta=30)
        refactored.load_state_dict(legacy.state_dict())

        torch.manual_seed(99)
        _, legacy = legacy_training.train_autoencoder(
            training_data,
            test_data,
            legacy,
            epochs=3,
            batch_size=8,
            learning_rate=0.003,
        )
        torch.manual_seed(99)
        fit_autoencoder(
            refactored,
            training_data,
            test_data,
            epochs=3,
            batch_size=8,
            learning_rate=0.003,
            legacy_validation_order=True,
        )
        for old_value, new_value in zip(
            legacy.state_dict().values(), refactored.state_dict().values()
        ):
            self.assertTrue(torch.equal(old_value, new_value))

    def test_mtl_storage_and_recall_match_legacy_model(self):
        torch.manual_seed(21)
        autoencoder = Autoencoder(input_dim=12, encoding_dim=12, K=5, beta=20)
        weights = autoencoder.get_weights(bias=True)

        np.random.seed(55)
        legacy = legacy_models.MTL(
            W_ei_ca1=weights[0].clone(),
            W_ca1_eo=weights[1].clone(),
            B_ei_ca1=weights[2].clone(),
            B_ca1_eo=weights[3].clone(),
            K_lat=5,
            K_out=3,
            dim_ca3=12,
            K_ca3=5,
            beta=20,
            alpha=0.2,
        )
        legacy.record = lambda *_: None
        np.random.seed(55)
        refactored = BTSPMemory(
            W_ei_ca1=weights[0],
            W_ca1_eo=weights[1],
            B_ei_ca1=weights[2],
            B_ca1_eo=weights[3],
            K_lat=5,
            K_out=3,
            dim_ca3=12,
            K_ca3=5,
            beta=20,
            alpha=0.2,
        )
        self.assertTrue(torch.equal(legacy.W_ei_ca3, refactored.W_ei_ca3))

        np.random.seed(72)
        patterns = generate_legacy_sparse_patterns(8, 3, 12)
        with torch.no_grad():
            for pattern in patterns:
                column = torch.as_tensor(pattern).reshape(-1, 1)
                legacy(column)
                refactored.store(column)
        self.assertTrue(torch.equal(legacy.W_ca3_ca1, refactored.W_ca3_ca1))

        legacy.pause_lr()
        with torch.no_grad():
            for pattern in patterns:
                column = torch.as_tensor(pattern).reshape(-1, 1)
                self.assertTrue(
                    torch.equal(legacy(column), refactored.retrieve(column))
                )


if __name__ == "__main__":
    unittest.main()

