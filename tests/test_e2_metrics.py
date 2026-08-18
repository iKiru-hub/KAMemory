"""Small contracts for the E2 capacity and cue-construction helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_experiment(name: str, filename: str):
    spec = spec_from_file_location(name, ROOT / "src" / "experiments" / filename)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


retention = load_experiment("e2_retention", "02_retention.py")
degraded = load_experiment("e2_degraded", "03_degraded_cues.py")


class E2MetricTests(unittest.TestCase):
    def test_contiguous_capacity_stops_at_first_failure(self):
        self.assertEqual(retention.contiguous_capacity([0.9, 0.8, 0.7], 0.75), 2)
        self.assertEqual(retention.contiguous_capacity([0.9, 0.8], 0.75), 2)
        self.assertEqual(retention.contiguous_capacity([0.7, 0.9], 0.75), 0)

    def test_retention_lifetime_uses_only_post_storage_trace(self):
        matrix = np.asarray(
            [
                [0.9, np.nan, np.nan],
                [0.8, 0.9, np.nan],
                [0.6, 0.8, 0.9],
            ]
        )
        observed = retention.retention_lifetimes(matrix, threshold=0.75)
        self.assertTrue(np.array_equal(observed, [1, 1, 0]))

    def test_controlled_overlap_is_exact_and_patterns_remain_sparse(self):
        for level in degraded.OVERLAP_LEVELS:
            patterns = degraded.controlled_overlap_patterns(
                num_patterns=8,
                size=50,
                num_active=5,
                overlap_fraction=float(level),
                rng=np.random.default_rng(7),
            )
            self.assertTrue(np.all(patterns.sum(axis=1) == 5))
            pairwise = [
                np.dot(patterns[left], patterns[right]) / 5
                for left in range(len(patterns))
                for right in range(left)
            ]
            self.assertTrue(np.allclose(pairwise, level))

    def test_nearest_neighbor_returns_clean_target_identity(self):
        bank = np.eye(4, dtype=np.float32)
        score, identity = degraded.nearest_neighbor_retrieval(
            bank[2], bank[2], bank
        )
        self.assertEqual(identity, 2)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
