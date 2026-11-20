import unittest
from types import SimpleNamespace

import numpy as np

from stage3_rl.env import (Go2WEnv, apply_base_velocity_kick,
                           failed_termination_adjustment, joint_acceleration_reward,
                           track_success, within_track_lane)


def make_fixed_env(x, y, steps=0, max_x=None):
    env = object.__new__(Go2WEnv)
    env.model = object()
    env.data = SimpleNamespace(
        xmat=np.eye(3).reshape(1, 9),
        xpos=np.array([[x, y, 1.0]]),
    )
    env.body_root = 0
    env._contacts = lambda: ([], 0.0)
    env.hfield = SimpleNamespace(height=lambda *_: 0.0)
    env.success_x = 100.0
    env.steps = steps
    env.env_episode_len = 10
    env.episode_len = 10
    env.max_x = float(x if max_x is None else max_x)
    env.fixed = object()
    return env


class TrackAcceptanceTests(unittest.TestCase):
    def test_failed_termination_has_no_absolute_position_progress_bonus(self):
        # A local episode starts past the old global progress cap; failure must not pay.
        self.assertEqual(failed_termination_adjustment(26.7), -1.5)
        self.assertEqual(failed_termination_adjustment(10.0), -1.5)

    def test_joint_acceleration_uses_consecutive_policy_step_velocities(self):
        # For a 1 rad/s change over 10ms, qddot=100 rad/s^2, not /10 smaller.
        dq = np.array([1.0, 0.0])
        previous = np.array([0.0, 0.0])
        self.assertAlmostEqual(joint_acceleration_reward(dq, previous, 0.01),
                               -1e-5 * (100.0**2))

    def test_environment_refreshes_previous_joint_velocity_each_policy_step(self):
        # The acceleration reward must compare consecutive actions, not episode start.
        env = Go2WEnv(seed=31, eval_mode=True, episode_len=20)
        previous = env.prev_dq.copy()
        env.step(np.zeros(29, dtype=np.float32))
        current = env.data.qvel[env.leg_vadr].copy()
        np.testing.assert_array_equal(env.prev_dq, current)
        self.assertFalse(np.array_equal(previous, env.prev_dq))

    def test_training_push_changes_world_linear_velocity_only(self):
        # MuJoCo free-joint qvel is world linear xyz then body angular xyz.
        qvel = np.zeros(6)
        apply_base_velocity_kick(qvel, np.array([0.2, -0.1]))
        np.testing.assert_array_equal(qvel, [0.2, -0.1, 0.0, 0.0, 0.0, 0.0])

    def test_lane_uses_strict_reported_half_width_everywhere(self):
        self.assertTrue(within_track_lane(4.0))
        self.assertTrue(within_track_lane(5.89))
        self.assertFalse(within_track_lane(5.9))
        self.assertFalse(within_track_lane(2.1))

    def test_success_requires_finish_and_global_lane_membership(self):
        self.assertTrue(track_success(100.0, 4.0, 100.0))
        self.assertTrue(track_success(100.0, 5.89, 100.0))
        self.assertFalse(track_success(99.99, 4.0, 100.0))
        self.assertFalse(track_success(100.0, 5.9, 100.0))

    def test_narrow_obstacle_segments_use_the_same_full_lane(self):
        self.assertEqual(make_fixed_env(62.0, 5.5)._check_done(),
                         (False, None, False))

    def test_fixed_track_timeout_does_not_award_partial_progress_success(self):
        self.assertEqual(make_fixed_env(98.0, 4.0, steps=10)._check_done(),
                         (True, "timeout", False))

    def test_fixed_track_exit_is_off_lane_and_finish_is_exact(self):
        self.assertEqual(make_fixed_env(100.0, 5.9, steps=9)._check_done(),
                         (True, "off_lane", False))
        self.assertEqual(make_fixed_env(100.0, 4.0, steps=10)._check_done(),
                         (True, "success", True))


if __name__ == "__main__":
    unittest.main()
