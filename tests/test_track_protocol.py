"""Contracts for factorial E3 and the legacy remapping reproduction."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import numpy as np

from kamemory.data import generate_factorial_track, generate_legacy_remapping_track


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "e3_track", ROOT / "src" / "experiments" / "04_ca1_track.py"
)
E3 = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(E3)


class FactorialTrackTests(unittest.TestCase):
    def make_track(self, seed=4):
        return generate_factorial_track(
            track_length=12,
            cue_positions=(2, 5, 8, 11),
            repeats_per_combination=2,
            cue_free_laps=4,
            spatial_size=12,
            sensory_size=10,
            sensory_active=2,
            place_field_sigma=2,
            rng=np.random.default_rng(seed),
        )

    def test_every_cue_is_crossed_with_every_position(self):
        track = self.make_track()
        cue_mask = track["cue_ids"] >= 0
        pairs = list(
            zip(track["cue_ids"][cue_mask], track["event_positions"][cue_mask])
        )
        for cue in (0, 1):
            for position in (2, 5, 8, 11):
                self.assertEqual(pairs.count((cue, position)), 2)
        self.assertEqual(np.count_nonzero(track["cue_ids"] < 0), 4)

    def test_components_and_cue_free_laps_are_explicit(self):
        track = self.make_track()
        self.assertTrue(
            np.array_equal(
                track["inputs"],
                np.concatenate((track["spatial"], track["sensory"]), axis=1),
            )
        )
        self.assertEqual(
            np.count_nonzero(track["cue_present"]),
            np.count_nonzero(track["cue_ids"] >= 0),
        )
        free_laps = np.flatnonzero(track["cue_ids"] < 0)
        for lap in free_laps:
            start = lap * int(track["track_length"])
            stop = start + int(track["track_length"])
            self.assertFalse(np.any(track["cue_present"][start:stop]))

    def test_heldout_background_is_new_but_cues_are_shared(self):
        train = self.make_track(seed=3)
        test = generate_factorial_track(
            track_length=12,
            cue_positions=(2, 5, 8, 11),
            repeats_per_combination=2,
            cue_free_laps=4,
            spatial_size=12,
            sensory_size=10,
            sensory_active=2,
            place_field_sigma=2,
            rng=np.random.default_rng(9),
            cue_patterns=train["cue_patterns"],
        )
        self.assertTrue(np.array_equal(train["cue_patterns"], test["cue_patterns"]))
        self.assertFalse(np.array_equal(train["inputs"], test["inputs"]))

    def test_decoders_remove_the_nuisance_factor(self):
        cues = np.repeat((0, 1), 4)
        positions = np.tile(np.arange(4), 2)
        features = np.concatenate(
            (np.eye(2)[cues], np.eye(4)[positions]), axis=1
        ).astype(np.float32)
        self.assertEqual(
            E3.cue_accuracy_across_positions(
                features, cues, positions, features, cues, positions
            ),
            1.0,
        )
        self.assertEqual(
            E3.position_accuracy_across_cues(
                features, cues, positions, features, cues, positions
            ),
            1.0,
        )

    def test_legacy_remapping_generator_matches_notebook_backend(self):
        import training as legacy_training

        track = {
            "length": 12,
            "num_laps": 3,
            "reward": "random",
            "cue": "random",
            "cue_position": [4],
        }
        network = {
            "K_lec": 2,
            "dim_mec": 12,
            "dim_lec": 10,
            "mec_N_x": 12,
            "mec_N_y": 1,
            "mec_sigma": 2,
            "num_cues": 1,
        }
        np.random.seed(17)
        expected = legacy_training.get_track_input(track, network, cue_duration=1)
        np.random.seed(17)
        observed = generate_legacy_remapping_track(
            track_length=12,
            num_laps=3,
            cue_position=4,
            spatial_size=12,
            sensory_size=10,
            sensory_active=2,
            place_field_sigma=2,
        )
        for expected_array, observed_array in zip(expected, observed):
            self.assertTrue(np.array_equal(expected_array, observed_array))


if __name__ == "__main__":
    unittest.main()
