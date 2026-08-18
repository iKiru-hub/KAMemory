"""Contracts for the frozen optimized configuration and E4 sweep helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "e4_sensitivity", ROOT / "src" / "experiments" / "05_sensitivity.py"
)
E4 = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(E4)


class E4ProtocolTests(unittest.TestCase):
    def test_frozen_reference_matches_best_params_artifact(self):
        reference = E4.load_reference(ROOT / "src" / "configs" / "optimized_memory.json")
        source = (ROOT / "src" / "optim_wb" / "best_params.yaml").read_text()
        parsed = {
            name: float(re.search(rf"{name}:\n\s+value:\s+([^\n]+)", source).group(1))
            for name in ("K_ca3", "K_lat", "alpha", "beta")
        }
        self.assertEqual(reference["parameters"]["ca3_active"], int(parsed["K_ca3"]))
        self.assertEqual(reference["parameters"]["ca1_active"], int(parsed["K_lat"]))
        self.assertEqual(reference["parameters"]["alpha"], parsed["alpha"])
        self.assertEqual(reference["parameters"]["beta"], parsed["beta"])

    def test_partial_permutation_has_prespecified_mismatch(self):
        identity = E4.partial_permutation(20, 0.0, np.random.default_rng(3))
        half = E4.partial_permutation(20, 0.5, np.random.default_rng(3))
        full = E4.partial_permutation(20, 1.0, np.random.default_rng(3))
        self.assertTrue(np.array_equal(identity, np.arange(20)))
        self.assertEqual(np.count_nonzero(half != np.arange(20)), 10)
        self.assertEqual(np.count_nonzero(full != np.arange(20)), 20)
        self.assertEqual(sorted(full.tolist()), list(range(20)))

    def test_uncertainty_uses_seed_axis(self):
        values = np.asarray([[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]])
        mean, ci = E4.mean_ci(values)
        self.assertTrue(np.array_equal(mean, np.asarray([3.0, 5.0])))
        self.assertEqual(ci.shape, mean.shape)
        self.assertTrue(np.all(ci > 0))


if __name__ == "__main__":
    unittest.main()
