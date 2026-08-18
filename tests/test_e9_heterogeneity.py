from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1] / "src" / "experiments" / "10_is_heterogeneity.py"
SPEC = spec_from_file_location("e9_heterogeneity", EXPERIMENT)
E9 = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(E9)


class E9HeterogeneityTests(unittest.TestCase):
    def test_background_masks_are_nested_and_exclude_cues(self):
        uniforms = np.asarray((0.0, 0.001, 0.004, 0.02, 0.2))
        cue = np.asarray((False, True, False, False, False))
        rates = np.asarray((0.0, 0.002, 0.005, 0.05))
        masks = E9.background_masks(uniforms, cue, rates)

        self.assertFalse(np.any(masks[:, cue]))
        self.assertFalse(np.any(masks[0]))
        self.assertTrue(np.all(masks[:-1] <= masks[1:]))
        np.testing.assert_array_equal(masks[-1], (uniforms < 0.05) & ~cue)

    def test_zero_rate_profile_is_exact_e3_profile(self):
        cue = np.asarray((False, True, False, False, True))
        masks = np.zeros((2, len(cue)), dtype=bool)
        masks[1, (0, 2)] = True
        profiles = E9.plasticity_profiles(
            cue, masks, cue_alpha=0.1, baseline_alpha=0.001
        )

        np.testing.assert_array_equal(
            profiles[0],
            np.asarray((0.001, 0.1, 0.001, 0.001, 0.1), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            profiles[1],
            np.asarray((0.1, 0.1, 0.1, 0.001, 0.1), dtype=np.float32),
        )

    def test_pareto_frontier_minimizes_error_and_maximizes_readout(self):
        error = np.asarray((4.0, 2.0, 1.0, 1.5))
        output = np.asarray((0.9, 0.85, 0.5, 0.8))
        np.testing.assert_array_equal(
            E9.pareto_frontier(error, output),
            np.asarray((True, True, True, True)),
        )

        output[-1] = 0.4
        np.testing.assert_array_equal(
            E9.pareto_frontier(error, output),
            np.asarray((True, True, True, False)),
        )

    def test_biological_comparison_has_zero_error_for_empirical_masks(self):
        empirical = np.asarray(((2, 1), (1, 2)))
        cue = np.asarray([[[[True, True, True, False, False, False]]]])
        spatial = np.asarray([[[[1.0, 1.0, 0.0, 1.0, 0.0, 0.0]]]])
        arrays = {"train_cue_active": cue, "test_position_eta": spatial}
        result = E9.biological_comparison(arrays, empirical, min_eta=0.5)

        np.testing.assert_array_equal(result["tables"][0, 0], empirical)
        self.assertTrue(
            np.isclose(result["primary_absolute_log_odds_ratio_error"][0], 0)
        )
        self.assertTrue(np.isclose(result["prevalence_rmse"][0, 0], 0))


if __name__ == "__main__":
    unittest.main()
