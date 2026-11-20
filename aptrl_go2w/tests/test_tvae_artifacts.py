import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage2_tvae.train_tvae import load_data, prepare_output_path


class TVAEArtifactTests(unittest.TestCase):
    def test_existing_checkpoint_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tvae.pt"
            checkpoint.write_bytes(b"existing checkpoint")
            with self.assertRaises(FileExistsError):
                prepare_output_path(str(checkpoint))
            self.assertEqual(checkpoint.read_bytes(), b"existing checkpoint")

    def test_windows_do_not_cross_trajectory_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            for gait in ("trot", "bound"):
                states = np.concatenate([np.ones((4, 78)), np.full((4, 78), 2.0)])
                torque = np.concatenate([np.ones((4, 12)), np.full((4, 12), 2.0)])
                np.savez(Path(directory) / f"dataset_{gait}.npz",
                         state=states, torque=torque, lens=np.array([4, 4]))
            data = load_data(directory)
            self.assertEqual(data["trot"]["win"].shape, (4, 3, 78))
            self.assertEqual(data["trot"]["window_traj"].tolist(), [0, 0, 1, 1])
            self.assertTrue(np.all(data["trot"]["win"][:2] == 1.0))
            self.assertTrue(np.all(data["trot"]["win"][2:] == 2.0))

    def test_new_checkpoint_parent_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "new-run" / "tvae.pt"
            self.assertEqual(prepare_output_path(str(checkpoint)), str(checkpoint))
            self.assertTrue(checkpoint.parent.is_dir())
            self.assertFalse(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
