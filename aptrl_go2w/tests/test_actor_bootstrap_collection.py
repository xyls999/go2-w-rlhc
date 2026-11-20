import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from stage3_rl.actor_bootstrap import collect_demonstrations


class FakeEncoder:
    def load_state_dict(self, state):
        return None

    def eval(self):
        return self

    def __call__(self, windows):
        batch = windows.shape[0]
        return torch.zeros((batch, 16)), torch.zeros((batch, 16))


class FakeEnv:
    instances = []
    terminate = False

    def __init__(self, seed, **kwargs):
        self.seed = seed
        self.kwargs = kwargs
        self.reset_called = False
        self.steps = 0
        self.vx_cmd = 1.5
        self.instances.append(self)

    def _get_obs(self):
        obs = np.full(426, (self.seed + self.steps) % 7, dtype=np.float32)
        obs[15] = self.vx_cmd
        return obs

    def reset(self):
        self.reset_called = True
        return self._get_obs()

    def step(self, action):
        self.steps += 1
        obs = self._get_obs()
        done = bool(self.terminate and self.steps == 1)
        return obs, 0.0, done, {"reason": "body_contact" if done else None}


class ActorBootstrapCollectionTests(unittest.TestCase):
    def setUp(self):
        FakeEnv.instances = []
        FakeEnv.terminate = False
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        state = np.arange(12 * 78, dtype=np.float32).reshape(12, 78)
        np.savez_compressed(
            self.data_dir / "dataset_trot.npz",
            state=state,
            lens=np.array([6, 6]),
            vx=np.array([0.5, 1.0]),
            seed=np.array([100, 200]),
        )
        self.checkpoint = self.root / "tvae.pt"
        torch.save({"enc": {}, "s_mean": np.zeros(78, np.float32),
                    "s_std": np.ones(78, np.float32)}, self.checkpoint)
        self.decoder = self.root / "decoder.npz"
        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        np.savez(self.decoder, checkpoint_sha256=np.array(digest))

    def tearDown(self):
        self.tmp.cleanup()

    def collect(self, out_dir, **kwargs):
        with patch("stage3_rl.actor_bootstrap.TVAEEncoder", FakeEncoder), \
             patch("stage3_rl.actor_bootstrap.Go2WEnv", FakeEnv):
            return collect_demonstrations(
                data_dir=str(self.data_dir),
                tvae_checkpoint=str(self.checkpoint),
                decoder_path=str(self.decoder),
                out_dir=str(out_dir),
                seed=9,
                episodes_per_speed=10,
                episode_len=3,
                **kwargs,
            )

    def test_collector_records_aligned_full_episodes_from_normal_reset(self):
        out = self.root / "run"
        summary = self.collect(out)

        self.assertEqual(summary["complete_episodes_by_speed"], {"0.5": 10, "1.0": 10})
        self.assertTrue(all(env.reset_called for env in FakeEnv.instances))
        self.assertTrue(all(env.kwargs["eval_mode"] is True for env in FakeEnv.instances))
        self.assertTrue(all(env.kwargs["train_stage"] == "flat" for env in FakeEnv.instances))
        with np.load(out / "demonstrations.npz") as data:
            self.assertEqual(data["observation"].shape, (20, 3, 426))
            self.assertEqual(data["action"].shape, (20, 3, 29))
            np.testing.assert_array_equal(data["action"][:, :, 16:28], 0.0)
            np.testing.assert_array_equal(data["action"][:, :, 28], -8.0)
            for speed, episode_obs in zip(data["speed"], data["observation"]):
                self.assertAlmostEqual(float(episode_obs[0, 15]), float(speed))
        with np.load(out / "demonstrations.npz") as data:
            trajectory_ids_a = data["trajectory_id"].copy()
            phase_offsets_a = data["phase_offset"].copy()
            episode_seeds_a = data["episode_seed"].copy()
        episodes = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
        self.assertEqual(len(episodes), 20)
        self.assertTrue(all(item["complete"] for item in episodes))
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["decoder_sha256"], hashlib.sha256(self.decoder.read_bytes()).hexdigest())
        self.assertTrue(all(item["reference_state_injected"] is False for item in episodes))

    def test_collector_selection_is_deterministic_for_same_seed(self):
        a, b = self.root / "run-a", self.root / "run-b"
        self.collect(a)
        self.collect(b)
        with np.load(a / "demonstrations.npz") as da, np.load(b / "demonstrations.npz") as db:
            np.testing.assert_array_equal(da["trajectory_id"], db["trajectory_id"])
            np.testing.assert_array_equal(da["phase_offset"], db["phase_offset"])
            np.testing.assert_array_equal(da["episode_seed"], db["episode_seed"])

    def test_incomplete_episodes_are_not_written_as_training_data(self):
        FakeEnv.terminate = True
        out = self.root / "failed-run"
        with self.assertRaisesRegex(RuntimeError, "10 complete"):
            self.collect(out)
        self.assertFalse((out / "demonstrations.npz").exists())
        self.assertTrue((out / "episodes.jsonl").exists())
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "gate_failed")

    def test_collector_refuses_nonempty_output_directory(self):
        out = self.root / "existing"
        out.mkdir()
        sentinel = out / "sentinel.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.collect(out)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_collector_rejects_decoder_export_from_another_checkpoint(self):
        np.savez(self.decoder, checkpoint_sha256=np.array("0" * 64))
        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            self.collect(self.root / "wrong-pair")


if __name__ == "__main__":
    unittest.main()
