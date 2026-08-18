"""Mathematical and analysis contracts for E5 interference."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "e5_interference", ROOT / "src" / "experiments" / "06_interference_analysis.py"
)
E5 = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(E5)


class E5InterferenceTests(unittest.TestCase):
    def test_contribution_decomposition_exactly_reconstructs_weight_update(self):
        signals = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        ca3 = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
        alpha = 0.2
        coefficients = torch.zeros((2, 2))
        weights = torch.zeros((2, 2))
        for load in range(2):
            coefficients, weights = E5.contribution_step(
                coefficients, weights, signals[load], ca3[load], load, alpha
            )
        self.assertTrue(torch.allclose(weights, coefficients.T @ ca3))
        self.assertTrue(torch.allclose(coefficients[0], torch.tensor([0.16, 0.0])))

    def test_age_curve_uses_lower_triangular_diagonals(self):
        matrix = np.asarray(
            [[1.0, np.nan, np.nan], [0.8, 1.0, np.nan], [0.6, 0.7, 1.0]]
        )
        self.assertTrue(np.allclose(E5.age_curve(matrix), (1.0, 0.75, 0.6)))

    def test_half_life_is_first_below_half_immediate(self):
        self.assertEqual(E5.first_half_age(np.asarray([1.0, 0.7, 0.49, 0.2])), 2)
        self.assertTrue(np.isnan(E5.first_half_age(np.asarray([1.0, 0.8, 0.6]))))


if __name__ == "__main__":
    unittest.main()
