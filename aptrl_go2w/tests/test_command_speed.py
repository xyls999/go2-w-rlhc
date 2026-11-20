import unittest

import numpy as np

from stage3_rl.env import Go2WEnv, command_speed_at_step
from stage3_rl.ppo_train import build_arg_parser, validate_training_args, worker_env_config


class FixedTrainingCommandTests(unittest.TestCase):
    def test_cli_configures_a_fixed_speed_only_for_flat_training(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--out", "new-run", "--probe-track", "track.npz",
            "--train-stage", "flat", "--command-speed", "1.5",
        ])

        validate_training_args(args)

        self.assertEqual(args.command_speed, 1.5)
        self.assertEqual(worker_env_config(args)["command_speed"], 1.5)

    def test_fixed_speed_is_rejected_outside_the_approved_grid(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--out", "new-run", "--probe-track", "track.npz",
            "--train-stage", "flat", "--command-speed", "1.4",
        ])
        with self.assertRaisesRegex(ValueError, "command speed"):
            validate_training_args(args)

    def test_fixed_speed_cannot_be_combined_with_mixed_training(self):
        args = build_arg_parser().parse_args([
            "--out", "new-run", "--probe-track", "track.npz",
            "--train-stage", "mixed", "--command-speed", "1.5",
        ])
        with self.assertRaisesRegex(ValueError, "flat"):
            validate_training_args(args)

    def test_fixed_speed_is_visible_in_the_first_environment_observation(self):
        env = Go2WEnv(seed=37, eval_mode=False, train_stage="flat",
                      episode_len=100, command_speed=1.5,
                      decoder_path="stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz")
        observation = env.reset()

        self.assertEqual(env.vx_cmd, 1.5)
        self.assertEqual(float(observation[15]), 1.5)
        np.testing.assert_allclose(observation.shape, (426,))

    def test_linear_ramp_has_exact_start_midpoint_and_end(self):
        self.assertEqual(command_speed_at_step(1.0, 1.5, 500, 0), 1.0)
        self.assertEqual(command_speed_at_step(1.0, 1.5, 500, 250), 1.25)
        self.assertEqual(command_speed_at_step(1.0, 1.5, 500, 500), 1.5)
        self.assertEqual(command_speed_at_step(1.0, 1.5, 500, 900), 1.5)

    def test_linear_ramp_rejects_invalid_speeds_and_steps(self):
        with self.assertRaises(ValueError):
            command_speed_at_step(1.0, 1.5, 0, 0)
        with self.assertRaises(ValueError):
            command_speed_at_step(1.4, 1.5, 500, 0)
        with self.assertRaises(ValueError):
            command_speed_at_step(1.5, 1.0, 500, 0)

    def test_cli_configures_ramp_only_for_flat_training(self):
        args = build_arg_parser().parse_args([
            "--out", "new-run", "--probe-track", "track.npz",
            "--train-stage", "flat", "--command-ramp-start", "1.0",
            "--command-ramp-end", "1.5", "--command-ramp-steps", "500",
        ])
        validate_training_args(args)
        self.assertEqual(worker_env_config(args)["command_ramp"], (1.0, 1.5, 500))

    def test_ramp_requires_all_parameters_and_excludes_fixed_command(self):
        parser = build_arg_parser()
        base = ["--out", "new-run", "--probe-track", "track.npz",
                "--train-stage", "flat"]
        incomplete = parser.parse_args(base + ["--command-ramp-start", "1.0"])
        with self.assertRaisesRegex(ValueError, "all three"):
            validate_training_args(incomplete)
        both = parser.parse_args(base + ["--command-speed", "1.5",
                                       "--command-ramp-start", "1.0",
                                       "--command-ramp-end", "1.5",
                                       "--command-ramp-steps", "500"])
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_training_args(both)

    def test_ramp_starts_in_initial_observation_and_updates_after_each_step(self):
        env = Go2WEnv(seed=37, eval_mode=False, train_stage="flat",
                      episode_len=100, command_ramp=(1.0, 1.5, 500),
                      decoder_path="stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz")
        observation = env.reset()
        self.assertEqual(float(observation[15]), 1.0)
        action = np.zeros(29, np.float32)
        observation, _, _, _ = env.step(action)
        self.assertAlmostEqual(env.vx_cmd, 1.001)
        self.assertAlmostEqual(float(observation[15]), 1.001)


if __name__ == "__main__":
    unittest.main()
