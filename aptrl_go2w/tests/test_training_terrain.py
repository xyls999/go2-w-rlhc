"""Training-only gap widths must never mutate the fixed acceptance NPZ."""
import hashlib
import os
from pathlib import Path
import unittest

import numpy as np

from common.terrain import load_npz_track


TRACK = os.environ.get("APTRL_TRACK_NPZ")


@unittest.skipUnless(TRACK, "APTRL_TRACK_NPZ required for geometry tests")
class TrainingTerrainTests(unittest.TestCase):
    def test_first_gap_narrows_only_on_a_private_copy(self):
        # Catches accidental alteration of the source, second gap, or lane sides.
        from common.training_terrain import training_heightmap

        before_hash = hashlib.sha256(Path(TRACK).read_bytes()).hexdigest()
        original, _, _, _ = load_npz_track(TRACK)
        for width in (0.1, 0.2, 0.3, 0.4, 0.5):
            with self.subTest(width=width):
                actual = training_heightmap(TRACK, width)
                kept = round(width / 0.1)
                self.assertIsNot(actual, original)
                np.testing.assert_array_equal(actual[:265], original[:265])
                np.testing.assert_array_equal(actual[265:265+kept],
                                              original[265:265+kept])
                np.testing.assert_array_equal(actual[265+kept:270],
                                              np.broadcast_to(original[270],
                                                              actual[265+kept:270].shape))
                np.testing.assert_array_equal(actual[270:], original[270:])
                self.assertLess(actual[265, 40], -4.5)
                self.assertLess(actual[265+kept, 20], -4.5)
                np.testing.assert_array_equal(actual[300:304], original[300:304])
        self.assertEqual(hashlib.sha256(Path(TRACK).read_bytes()).hexdigest(),
                         before_hash)
        np.testing.assert_array_equal(training_heightmap(TRACK, 0.5), original)

    def test_rejects_unapproved_training_widths(self):
        from common.training_terrain import training_heightmap

        for width in (0.0, 0.15, 0.6, -0.1):
            with self.subTest(width=width), self.assertRaises(ValueError):
                training_heightmap(TRACK, width)


if __name__ == "__main__":
    unittest.main()
