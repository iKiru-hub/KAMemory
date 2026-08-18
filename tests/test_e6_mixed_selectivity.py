"""Analysis contracts for E6 cross-validated mixed selectivity."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "e6_mixed", ROOT / "src" / "experiments" / "07_ca1_mixed_selectivity.py"
)
E6 = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(E6)


class E6MixedSelectivityTests(unittest.TestCase):
    def test_f_survival_matches_standard_five_percent_critical_values(self):
        self.assertAlmostEqual(E6.f_survival(np.asarray([4.844]), 1, 11)[0], 0.05, places=3)
        self.assertAlmostEqual(E6.f_survival(np.asarray([4.066]), 3, 8)[0], 0.05, places=3)

    def test_factorial_classifier_resolves_prespecified_classes(self):
        cues = np.repeat((0, 1), 8)
        positions = np.tile(np.repeat(np.arange(4), 2), 2)
        position = positions.astype(float)
        cue = cues.astype(float)
        activity = np.column_stack(
            (
                position,
                cue,
                position + cue,
                (cue == 1) & (positions == 3),
                np.ones(len(cue)),
            )
        ).astype(float)
        effects = E6.factorial_effects(activity, cues, positions, min_eta=0.05)
        self.assertTrue(np.array_equal(E6.classify_cells(effects), np.arange(5)))

    def test_conjunctive_cell_retains_nonexclusive_main_effects(self):
        cues = np.repeat((0, 1), 8)
        positions = np.tile(np.repeat(np.arange(4), 2), 2)
        activity = (((cues == 1) & (positions == 3)).astype(float))[:, None]
        effects = E6.factorial_effects(activity, cues, positions, min_eta=0.05)
        self.assertTrue(effects["cue_active"][0])
        self.assertTrue(effects["position_active"][0])
        self.assertTrue(effects["interaction_active"][0])
        self.assertEqual(E6.classify_cells(effects)[0], 3)

    def test_bh_and_average_ranks_handle_boundaries_and_ties(self):
        significant = E6.bh_significant(np.asarray([0.001, 0.02, 0.2]), 0.05)
        self.assertTrue(np.array_equal(significant, (True, True, False)))
        self.assertTrue(np.allclose(E6.average_ranks(np.asarray([2, 1, 1, 4])), (3, 1.5, 1.5, 4)))

    def test_sparse_weight_replay_is_numerically_equivalent(self):
        ca3 = torch.tensor([[1.0, 0.0], [0.5, 1.0]])
        signals = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        alphas = np.asarray([0.2, 0.1])
        observed = E6.sparse_final_weights(ca3, signals, alphas)
        expected = torch.zeros((2, 2))
        for presynaptic, signal, alpha in zip(ca3, signals, alphas):
            expected = (1 - alpha * signal[:, None]) * expected + alpha * torch.outer(signal, presynaptic)
        self.assertTrue(torch.allclose(observed, expected))

    def test_cue_tuning_dose_response_aggregates_within_seed(self):
        tuning = np.asarray([[[0.0, 0.25, 0.5, 0.75, 1.0]]])
        remapping = np.asarray([[[0.0, 0.2, 0.4, 0.6, 0.8]]])
        observed = E6.cue_tuning_dose_response(tuning, remapping)
        self.assertTrue(np.allclose(observed, ((0.0, 0.2, 0.4, 0.7),)))


if __name__ == "__main__":
    unittest.main()
