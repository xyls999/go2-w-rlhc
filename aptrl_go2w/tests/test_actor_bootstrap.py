import unittest

import numpy as np
import torch
from torch import nn

from stage3_rl.actor_bootstrap import (
    make_actor_input,
    make_teacher_action,
    split_episode_indices,
)
from stage3_rl.env import EXTERO, GAIT_DIM, PROPRIO
from stage3_rl.ppo_train import RunningNorm


class First32(nn.Module):
    def forward(self, value):
        return value[:, :32]


class ActorBootstrapUtilityTests(unittest.TestCase):
    def test_teacher_action_places_latent_aux_and_trot_logit(self):
        latent = np.arange(16, dtype=np.float32)

        action = make_teacher_action(latent, "trot")

        self.assertEqual(action.shape, (29,))
        np.testing.assert_array_equal(action[:16], latent)
        np.testing.assert_array_equal(action[16:28], np.zeros(12, np.float32))
        self.assertEqual(action[28], -8.0)

    def test_teacher_action_rejects_unknown_gait(self):
        with self.assertRaises(ValueError):
            make_teacher_action(np.zeros(16, np.float32), "pace")

    def test_episode_split_is_deterministic_disjoint_and_complete(self):
        train_a, val_a = split_episode_indices(10, 0.2, 19)
        train_b, val_b = split_episode_indices(10, 0.2, 19)

        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(val_a, val_b)
        self.assertEqual(len(set(train_a) & set(val_a)), 0)
        np.testing.assert_array_equal(np.sort(np.r_[train_a, val_a]), np.arange(10))
        self.assertGreater(len(train_a), 0)
        self.assertGreater(len(val_a), 0)

    def test_episode_split_requires_at_least_two_episodes(self):
        with self.assertRaises(ValueError):
            split_episode_indices(1, 0.2, 3)

    def test_actor_input_preserves_feature_order_and_expected_dimension(self):
        obs = np.arange(PROPRIO + EXTERO + GAIT_DIM, dtype=np.float32)
        norm = RunningNorm(obs.size, clip=1000.0)
        norm.mean[:] = 0
        norm.var[:] = 1

        result = make_actor_input(obs, norm, First32(), "cpu")

        self.assertEqual(tuple(result.shape), (1, PROPRIO + 32 + GAIT_DIM))
        torch.testing.assert_close(result[0, :PROPRIO], torch.from_numpy(obs[:PROPRIO]))
        torch.testing.assert_close(
            result[0, PROPRIO:PROPRIO + 32],
            torch.from_numpy(obs[PROPRIO:PROPRIO + 32]),
        )
        torch.testing.assert_close(
            result[0, PROPRIO + 32:],
            torch.from_numpy(obs[PROPRIO + EXTERO:]),
        )


if __name__ == "__main__":
    unittest.main()
