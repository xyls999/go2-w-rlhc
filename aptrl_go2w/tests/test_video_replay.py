"""Rendering must not change fixed-track MuJoCo dynamics or hide replay failure."""
import io
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from stage3_rl.env import Go2WEnv
from stage3_rl.evaluate import PolicyRunner, VideoRecorder, run_episode


TRACK = os.environ.get("APTRL_TRACK_NPZ")
CKPT = os.path.join(os.path.dirname(__file__), "..", "stage3_rl", "runs_v21",
                    "model_latest.pt")


class FakeRenderer:
    def update_scene(self, data, camera):
        self.lookat = tuple(camera.lookat)

    def render(self):
        return np.zeros((1, 1, 3), dtype=np.uint8)


class FakeInput:
    def write(self, frame):
        return len(frame)

    def close(self):
        pass


class FakeProcess:
    stdin = FakeInput()
    stderr = io.BytesIO()

    def wait(self):
        return 0


class VideoReplayTests(unittest.TestCase):
    def test_finish_drains_large_ffmpeg_stderr_without_hanging(self):
        # Catches waiting for ffmpeg before draining its bounded stderr pipe.
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "diagnostic.mp4")
            with open(path, "wb") as stream:
                stream.write(b"diagnostic fixture")
            proc = subprocess.Popen([
                sys.executable, "-c",
                "import sys; sys.stdin.buffer.read(); "
                "sys.stderr.buffer.write(b'x'*200000); sys.stderr.flush()",
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE)
            recorder = object.__new__(VideoRecorder)
            recorder.active = True
            recorder.path = path
            recorder.proc = proc
            errors = []
            def finish():
                try:
                    recorder.finish(True)
                except Exception as exc:
                    errors.append(exc)
            thread = threading.Thread(target=finish, daemon=True)
            thread.start()
            thread.join(timeout=5)
            blocked = thread.is_alive()
            if blocked:
                proc.kill()
                thread.join(timeout=5)
            self.assertFalse(blocked, "ffmpeg stderr pipe blocked finish()")
            self.assertEqual(errors, [])
            self.assertTrue(proc.stderr.closed)

    @unittest.skipUnless(TRACK, "APTRL_TRACK_NPZ is required for real-physics replay")
    def test_recording_callback_does_not_recompute_or_change_trajectory(self):
        # Catches the extra mj_forward that changed the v21 failure location.
        runner = PolicyRunner(CKPT, device="cpu")
        plain = Go2WEnv(seed=9001, fixed_track=TRACK, eval_mode=True,
                        episode_len=4000, eval_protocol="pose-v1")
        baseline = run_episode(plain, runner, max_steps=4000)
        recorded = Go2WEnv(seed=9001, fixed_track=TRACK, eval_mode=True,
                           episode_len=4000, eval_protocol="pose-v1")
        recorder = object.__new__(VideoRecorder)
        recorder.active = True
        recorder.n = 0
        recorder.frame_every = 2
        recorder.cam = mujoco.MjvCamera()
        recorder.renderer = FakeRenderer()
        recorder.proc = FakeProcess()

        def observed_step(env):
            with patch("mujoco.mj_forward", side_effect=AssertionError(
                    "recorder must not recompute physics")):
                recorder(env)

        replay = run_episode(recorded, runner, max_steps=4000,
                             recorder=observed_step)
        self.assertEqual(replay["reason"], baseline["reason"])
        self.assertEqual(replay["steps"], baseline["steps"])
        self.assertAlmostEqual(replay["max_x"], baseline["max_x"], places=6)
        self.assertEqual(replay["settled_initial_pose"],
                         baseline["settled_initial_pose"])

    def test_mismatched_replay_deletes_diagnostic_video(self):
        # Catches an ffmpeg file surviving a failed verification as a fake success.
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "invalid.mp4")
            with open(path, "wb") as stream:
                stream.write(b"not a verified result")
            recorder = object.__new__(VideoRecorder)
            recorder.active = True
            recorder.path = path
            recorder.proc = FakeProcess()
            with self.assertRaisesRegex(RuntimeError, "did not reproduce"):
                recorder.finish(False)
            self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
