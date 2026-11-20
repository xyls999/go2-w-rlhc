import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage3_rl.validate_bootstrap import evaluate_bootstrap


class FakeEnv:
    lane_fail = False
    progress_per_speed = {0.5: 0.005, 1.0: 0.010}
    terminal_at = None

    def __init__(self, seed, episode_len, **kwargs):
        self.seed = seed
        self.episode_len = episode_len
        self.kwargs = kwargs
        self.steps = 0
        self.vx_cmd = 1.5
        self.data = type("Data", (), {})()
        self.data.qpos = np.array([0.5, 4.0, 0.34], dtype=np.float64)

    def _get_obs(self):
        obs = np.zeros(426, np.float32)
        obs[15] = self.vx_cmd
        return obs

    def reset(self):
        self.steps = 0
        return self._get_obs()

    def step(self, action):
        self.steps += 1
        self.data.qpos[0] += self.progress_per_speed[self.vx_cmd]
        if self.lane_fail and self.steps >= 2:
            self.data.qpos[1] = 5.91
        done = self.terminal_at == self.steps
        return self._get_obs(), 0.0, done, {"reason": "body_contact" if done else None}


class FakeRunner:
    def __init__(self, checkpoint, device=None, decoder_path=None):
        self.checkpoint = checkpoint
        self.decoder_path = decoder_path

    def act(self, observation, deterministic=True):
        return np.zeros(29, np.float32)


class BootstrapGateTests(unittest.TestCase):
    def setUp(self):
        FakeEnv.lane_fail = False
        FakeEnv.progress_per_speed = {0.5: 0.005, 1.0: 0.010}
        FakeEnv.terminal_at = None
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "model.pt"
        self.decoder = self.root / "decoder.npz"
        self.checkpoint.write_bytes(b"ckpt")
        self.decoder.write_bytes(b"decoder")

    def tearDown(self):
        self.tmp.cleanup()

    def evaluate(self, output, **kwargs):
        options = dict(
            checkpoint=str(self.checkpoint), decoder_path=str(self.decoder),
            out_dir=str(output), episodes_per_speed=10, episode_len=1000,
            seed_base=100, device="cpu", env_factory=FakeEnv,
            runner_factory=FakeRunner,
        )
        options.update(kwargs)
        return evaluate_bootstrap(**options)

    def test_gate_checks_speed_specific_displacement_and_full_episode(self):
        result = self.evaluate(self.root / "pass")

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["successful_episodes_by_speed"], {"0.5": 10, "1.0": 10})
        self.assertEqual(result["episodes_per_speed"], 10)

    def test_gate_fails_if_fast_speed_misses_its_displacement_threshold(self):
        FakeEnv.progress_per_speed = {0.5: 0.005, 1.0: 0.007}

        result = self.evaluate(self.root / "fail-distance")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["successful_episodes_by_speed"]["0.5"], 10)
        self.assertEqual(result["successful_episodes_by_speed"]["1.0"], 0)

    def test_gate_enforces_strict_lane(self):
        FakeEnv.lane_fail = True
        result = self.evaluate(self.root / "fail-lane")
        self.assertEqual(result["successful_episodes_by_speed"], {"0.5": 0, "1.0": 0})

    def test_gate_rejects_early_terminal_even_if_distance_would_pass(self):
        FakeEnv.terminal_at = 1
        result = self.evaluate(self.root / "early-terminal")
        self.assertEqual(result["successful_episodes_by_speed"], {"0.5": 0, "1.0": 0})

    def test_gate_requires_at_least_ten_episodes_per_speed(self):
        with self.assertRaisesRegex(ValueError, "at least 10"):
            self.evaluate(self.root / "too-few", episodes_per_speed=9)

    def test_output_directory_is_never_overwritten(self):
        out = self.root / "existing"
        out.mkdir()
        marker = out / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.evaluate(out)
        self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
