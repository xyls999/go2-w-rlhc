import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from stage3_rl.ppo_train import (
    build_arg_parser,
    compute_chunk_count,
    ensure_empty_output_dir,
    Learner,
    write_probe_record,
)


class PPOProbeConfigTests(unittest.TestCase):
    def test_chunk_count_uses_ceiling_and_validates_inputs(self):
        self.assertEqual(compute_chunk_count(4096, 4, 1), 21)
        self.assertEqual(compute_chunk_count(65536, 4, 1), 328)
        for values in ((0, 4, 1), (4096, 0, 1), (4096, 4, 0),
                       (4096, 4, 1, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                compute_chunk_count(*values)

    def test_parser_requires_track_and_run_directory_and_sets_batch_default(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--probe-track", "track.npz", "--out", "new-run",
            "--batch-target", "4096",
        ])
        self.assertEqual(args.probe_track, "track.npz")
        self.assertEqual(args.out, "new-run")
        self.assertEqual(args.batch_target, 4096)
        self.assertEqual(parser.parse_args([
            "--probe-track", "track.npz", "--out", "new-run",
        ]).batch_target, 65536)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--out", "new-run"])

    def test_probe_bounds_are_configurable_and_positive(self):
        parser=build_arg_parser()
        args=parser.parse_args(['--probe-track','track.xml','--out','unused',
                                '--probe-episodes','3','--probe-max-steps','2500'])
        self.assertEqual(args.probe_episodes,3)
        self.assertEqual(args.probe_max_steps,2500)
        with self.assertRaises(SystemExit):
            parser.parse_args(['--probe-track','track.xml','--out','unused','--probe-episodes','0'])

    def test_probe_records_append_one_json_object_per_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "probe_metrics.jsonl")
            first = {"iter": 25, "best_x": 3.0}
            second = {"iter": 50, "best_x": 5.0}
            write_probe_record(path, first)
            write_probe_record(path, second)
            with open(path, encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(rows, [first, second])

    def test_track_probe_uses_explicit_track_and_reports_progress_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            track_path = os.path.join(temp_dir, "track.npz")
            with open(track_path, "wb") as stream:
                stream.write(b"track")

            class DummyNorm:
                def normalize(self, observations):
                    return np.asarray(observations, dtype=np.float32)

            class DummyEncoder(torch.nn.Module):
                def forward(self, observations):
                    return observations

            class DummyActor:
                def dist(self, observations):
                    return SimpleNamespace(mean=torch.zeros((len(observations), 29)))

            class DummyEnv:
                tracks = []
                protocols = []
                decoder_paths = []

                def __init__(self, seed, fixed_track, eval_mode, episode_len,
                             eval_protocol="legacy", decoder_path=None):
                    self.seed = seed
                    self.tracks.append(fixed_track)
                    self.protocols.append(eval_protocol)
                    self.decoder_paths.append(decoder_path)
                    self.steps_until_done = seed - 76
                    self.data = SimpleNamespace(qpos=np.array([0.5, 4.0, 1.0]))
                    self.requested_initial_pose = [0.5 + 0.01 * (seed - 77), 4.0, 0.0]
                    self.settled_initial_pose = self.requested_initial_pose[:]

                def reset(self):
                    self.steps = 0
                    self.data.qpos[0] = 0.5
                    return np.zeros(426, dtype=np.float32)

                def step(self, action):
                    self.steps += 1
                    self.data.qpos[0] += 8.0
                    done = self.steps >= self.steps_until_done
                    reason = {77: "body_contact", 78: "timeout", 79: "success"}[self.seed]
                    info = {"reason": reason if done else None,
                            "success": done and reason == "success",
                            "terminal_contacts": []}
                    return np.zeros(426, dtype=np.float32), 1.0, done, info

            learner = object.__new__(Learner)
            learner.norm = DummyNorm()
            learner.enc = DummyEncoder()
            learner.actor = DummyActor()
            learner.device = "cpu"
            with patch("stage3_rl.env.Go2WEnv", DummyEnv):
                probe = learner.track_probe(track_path, n_ep=3, max_steps=4,
                                            seed_base=77, eval_protocol="pose-v1")

            self.assertEqual(DummyEnv.tracks, [track_path] * 3)
            self.assertEqual(DummyEnv.protocols, ["pose-v1"] * 3)
            self.assertTrue(all(path is not None for path in DummyEnv.decoder_paths))
            self.assertEqual(probe["episodes"][1]["requested_initial_pose"],
                             [0.51, 4.0, 0.0])
            self.assertEqual(probe["episode_seeds"], [77, 78, 79])
            self.assertEqual(probe["best_x"], 24.5)
            self.assertEqual(probe["median_x"], 16.5)
            self.assertEqual(probe["success_count"], 1)
            self.assertEqual(probe["failure_counts"],
                             {"body_contact": 1, "timeout": 1})
            self.assertEqual(probe["completed_segments"], [1, 2, 3])

    def test_training_output_directory_must_be_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = os.path.join(temp_dir, "new-run")
            self.assertEqual(ensure_empty_output_dir(out), os.path.abspath(out))
            with open(os.path.join(out, "existing.txt"), "w") as stream:
                stream.write("preserve")
            with self.assertRaises(FileExistsError):
                ensure_empty_output_dir(out)


if __name__ == "__main__":
    unittest.main()
