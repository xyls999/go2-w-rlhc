"""Seed-distinct evaluation starts without relaxing the fixed race-track rules."""
import os
import unittest

from stage3_rl.env import Go2WEnv, track_success, within_track_lane
from stage3_rl import evaluate, evaluate_par


TRACK = os.environ.get("APTRL_TRACK_NPZ")


class EvaluationProtocolTests(unittest.TestCase):
    def test_seed_pose_mapping_is_repeatable_distinct_and_bounded(self):
        # Catches an ignored seed, shared mutable RNG, or oversized perturbation.
        from common.eval_protocol import initial_pose

        self.assertEqual(initial_pose(9000, "legacy"), (0.5, 4.0, 0.0))
        poses = [initial_pose(seed, "pose-v1") for seed in range(9000, 9100)]
        self.assertEqual(len(set(poses)), 100)
        self.assertEqual(poses[0], initial_pose(9000, "pose-v1"))
        for x, y, yaw in poses:
            self.assertTrue(0.45 <= x <= 0.55)
            self.assertTrue(3.9 <= y <= 4.1)
            self.assertTrue(-0.02 <= yaw <= 0.02)
        with self.assertRaises(ValueError):
            initial_pose(9000, "unknown")

    @unittest.skipUnless(TRACK, "APTRL_TRACK_NPZ is required for real-track protocol tests")
    def test_real_fixed_track_starts_differ_but_success_rule_does_not(self):
        # Catches jitter being ignored by the real MuJoCo reset or changing acceptance.
        first = Go2WEnv(seed=9000, fixed_track=TRACK, eval_mode=True,
                        eval_protocol="pose-v1")
        second = Go2WEnv(seed=9001, fixed_track=TRACK, eval_mode=True,
                         eval_protocol="pose-v1")
        self.assertNotEqual(first.requested_initial_pose,
                            second.requested_initial_pose)
        self.assertNotEqual(first.settled_initial_pose,
                            second.settled_initial_pose)
        first_again = Go2WEnv(seed=9000, fixed_track=TRACK, eval_mode=True,
                              eval_protocol="pose-v1")
        self.assertEqual(first.requested_initial_pose,
                         first_again.requested_initial_pose)
        self.assertEqual(first.settled_initial_pose,
                         first_again.settled_initial_pose)
        self.assertFalse(track_success(99.99, 4.0, 100.0))
        self.assertFalse(within_track_lane(5.9))
        with self.assertRaises(ValueError):
            Go2WEnv(seed=9000, eval_mode=False, eval_protocol="pose-v1")

    def test_evaluators_manifest_the_same_explicit_protocol(self):
        # Catches serial and parallel diverging while reporting the same seed policy.
        for module, name in ((evaluate, "serial"), (evaluate_par, "parallel")):
            args = module.build_arg_parser().parse_args([
                "--track", "track.npz", "--out", "new-run",
                "--eval-protocol", "pose-v1",
            ])
            config = evaluate.evaluation_config(args, name)
            self.assertEqual(config["eval_protocol"], "pose-v1")
            self.assertEqual(config["seed_policy"], "seed_base + episode_index")
            self.assertIn("decoder_path", config)
            self.assertIn("decoder_sha256", config)


if __name__ == "__main__":
    unittest.main()
