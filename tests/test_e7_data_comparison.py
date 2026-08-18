"""Contracts for the prespecified E7 published-data comparison."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "e7_comparison", ROOT / "src" / "experiments" / "08_ca1_data_comparison.py"
)
E7 = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(E7)


class E7DataComparisonTests(unittest.TestCase):
    def test_reference_counts_match_published_numerators(self):
        with E7.DEFAULT_REFERENCE.open("r", encoding="utf-8") as handle:
            reference = json.load(handle)
        table = E7.empirical_table(reference)
        self.assertTrue(np.array_equal(table, ((71, 37), (273, 204))))
        self.assertEqual(table[0].sum(), 108)
        self.assertEqual(table[1].sum(), 477)

    def test_contingency_orientation_and_conditionals(self):
        cue = np.asarray((1, 1, 0, 0, 0), dtype=bool)
        spatial = np.asarray((1, 0, 1, 1, 0), dtype=bool)
        table = E7.contingency(cue, spatial)
        self.assertTrue(np.array_equal(table, ((1, 1), (2, 1))))
        self.assertTrue(np.allclose(E7.conditional_spatial(table), (0.5, 2 / 3)))

    def test_odds_ratio_uses_prespecified_correction_and_handles_absence(self):
        table = np.asarray(((1, 1), (1, 3)))
        expected = np.log((1.5 * 3.5) / (1.5 * 1.5))
        self.assertAlmostEqual(E7.log_odds_ratio(table), expected)
        self.assertTrue(np.isnan(E7.log_odds_ratio(np.asarray(((0, 0), (2, 3))))))

    def test_published_association_is_positive(self):
        with E7.DEFAULT_REFERENCE.open("r", encoding="utf-8") as handle:
            table = E7.empirical_table(json.load(handle))
        active, inactive = E7.conditional_spatial(table)
        self.assertGreater(active, inactive)
        self.assertGreater(E7.log_odds_ratio(table), 0)


if __name__ == "__main__":
    unittest.main()
