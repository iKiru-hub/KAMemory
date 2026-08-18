import unittest

import torch

from kamemory.models import Autoencoder, BTSPMemory
from kamemory.plasticity import (
    apply_plasticity,
    delta_update,
    hebbian_update,
    target_gated_update,
)


class PlasticityRuleTests(unittest.TestCase):
    def setUp(self):
        self.weights = torch.tensor(
            [[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]], dtype=torch.float32
        )
        self.presynaptic = torch.tensor([1.0, 0.0, 1.0])
        self.target = torch.tensor([1.0, 0.0])

    def test_target_gated_matches_validated_equation(self):
        observed = target_gated_update(
            self.weights, self.presynaptic, self.target, 0.25
        )
        column_target = self.target.reshape(-1, 1)
        column_pre = self.presynaptic.reshape(-1, 1)
        expected = (1 - 0.25 * column_target) * self.weights + 0.25 * (
            column_target @ column_pre.T
        )
        self.assertTrue(torch.equal(observed, expected))

    def test_hebbian_is_potentiation_only_and_bounded(self):
        observed = hebbian_update(
            self.weights, self.presynaptic, self.target, 0.75
        )
        expected = torch.tensor(
            [[0.95, 0.4, 1.0], [0.1, 0.3, 0.5]], dtype=torch.float32
        )
        self.assertTrue(torch.allclose(observed, expected))

    def test_delta_corrects_a_fraction_of_linear_prediction_error(self):
        zero = torch.zeros_like(self.weights)
        observed = delta_update(zero, self.presynaptic, self.target, 0.5)
        expected = torch.tensor(
            [[0.25, 0.0, 0.25], [0.0, 0.0, 0.0]], dtype=torch.float32
        )
        self.assertTrue(torch.allclose(observed, expected))

    def test_named_dispatch_rejects_unknown_rule(self):
        with self.assertRaisesRegex(ValueError, "unknown plasticity rule"):
            apply_plasticity(
                "missing", self.weights, self.presynaptic, self.target, 0.1
            )

    def test_memory_dispatch_preserves_default_and_supports_baselines(self):
        torch.manual_seed(4)
        autoencoder = Autoencoder(input_dim=6, encoding_dim=6, K=2, beta=10)
        default = BTSPMemory.from_autoencoder(
            autoencoder,
            K_lat=2,
            K_out=2,
            dim_ca3=6,
            K_ca3=2,
            beta=10,
            alpha=0.25,
        )
        hebbian = BTSPMemory.from_autoencoder(
            autoencoder,
            K_lat=2,
            K_out=2,
            dim_ca3=6,
            K_ca3=2,
            beta=10,
            alpha=0.25,
            plasticity_rule="hebbian",
        )
        hebbian.W_ei_ca3.copy_(default.W_ei_ca3)
        pattern = torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.float32)
        signal = torch.tensor([0, 1, 0, 1, 0, 0], dtype=torch.float32)
        ca3 = default.ca3_activity(pattern)

        default.store(pattern, instructive_signal=signal)
        expected_default = target_gated_update(
            torch.zeros_like(default.W_ca3_ca1), ca3, signal, 0.25
        )
        self.assertTrue(torch.equal(default.W_ca3_ca1, expected_default))

        hebbian.store(pattern, instructive_signal=signal)
        expected_hebbian = hebbian_update(
            torch.zeros_like(hebbian.W_ca3_ca1), ca3, signal, 0.25
        )
        self.assertTrue(torch.equal(hebbian.W_ca3_ca1, expected_hebbian))


if __name__ == "__main__":
    unittest.main()
