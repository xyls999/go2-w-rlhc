"""Train-only profiles keep official geometry, action bounds, and finishes intact."""
import os
import unittest

import mujoco
import numpy as np

from stage3_rl.env import Go2WEnv


TRACK = os.environ.get("APTRL_TRACK_NPZ")


@unittest.skipUnless(TRACK, "APTRL_TRACK_NPZ required for real training profiles")
class TrainingProfileTests(unittest.TestCase):
    def test_single_and_double_gap_are_local_only(self):
        # Catches accidental editing of the official full-track acceptance env.
        official = Go2WEnv(seed=9000, fixed_track=TRACK, eval_mode=True)
        first = Go2WEnv(seed=9000, train_stage="single_gap",
                        gap_width_m=0.1, train_track=TRACK, eval_mode=True)
        original_width = Go2WEnv(seed=9000, train_stage="single_gap",
                                 gap_width_m=0.5, train_track=TRACK,
                                 eval_mode=True)
        both = Go2WEnv(seed=9000, train_stage="double_gap", train_track=TRACK,
                       eval_mode=True)
        self.assertEqual(official.success_x, 100.0)
        self.assertEqual(first.success_x, 28.3)
        self.assertEqual(original_width.success_x, 28.3)
        self.assertEqual(both.success_x, 32.0)
        self.assertAlmostEqual(first.H[265, 40], official.H[265, 40])
        self.assertAlmostEqual(first.H[266, 40], official.H[270, 40])
        np.testing.assert_array_equal(original_width.H, official.H)
        np.testing.assert_array_equal(both.H, official.H)
        np.testing.assert_array_equal(first.H[300:304], official.H[300:304])
        np.testing.assert_array_equal(first.leg_effort, official.leg_effort)
        np.testing.assert_array_equal(first.model.actuator_ctrlrange,
                                      official.model.actuator_ctrlrange)
        self.assertAlmostEqual(official.requested_initial_pose[0], 0.5)
        self.assertTrue(24.45 <= first.requested_initial_pose[0] <= 24.55)
        self.assertGreater(first.settled_initial_pose[0], 24.3)

    def test_local_gate_seeds_differ_reproducibly_and_report_success(self):
        # Catches repeated, non-independent local gates and fixed-env stats loss.
        def fresh(seed):
            return Go2WEnv(seed=seed, train_stage="double_gap",
                           train_track=TRACK, eval_mode=True)
        a, b, repeat = fresh(9000), fresh(9001), fresh(9000)
        self.assertNotEqual(a.requested_initial_pose,
                            b.requested_initial_pose)
        self.assertEqual(a.requested_initial_pose,
                         repeat.requested_initial_pose)
        self.assertEqual(a.settled_initial_pose,
                         repeat.settled_initial_pose)
        self.assertGreater(a.settled_initial_pose[0], 24.3)
        before = a.stats["episodes"]
        a.report(True)
        self.assertEqual(a.stats["episodes"], before + 1)
        self.assertEqual(a.stats["success"], 1)

    def test_original_rough_slope_retention_profile_is_local_and_seeded(self):
        # Catches altered original terrain, a seam start, or accidental full finish.
        official = Go2WEnv(seed=9000, fixed_track=TRACK, eval_mode=True)
        first = Go2WEnv(seed=9000, train_stage="rough_slope_retain",
                        train_track=TRACK, eval_mode=True)
        second = Go2WEnv(seed=9001, train_stage="rough_slope_retain",
                         train_track=TRACK, eval_mode=True)
        repeat = Go2WEnv(seed=9000, train_stage="rough_slope_retain",
                         train_track=TRACK, eval_mode=True)
        np.testing.assert_array_equal(first.H, official.H)
        np.testing.assert_array_equal(first.model.actuator_ctrlrange,
                                      official.model.actuator_ctrlrange)
        self.assertEqual(first.success_x, 23.5)
        self.assertEqual(official.success_x, 100.0)
        self.assertTrue(14.95 <= first.requested_initial_pose[0] <= 15.05)
        self.assertNotEqual(first.requested_initial_pose,
                            second.requested_initial_pose)
        self.assertEqual(first.settled_initial_pose,
                         repeat.settled_initial_pose)
        training = Go2WEnv(seed=9000, train_stage="rough_slope_retain",
                            train_track=TRACK, eval_mode=False)
        geom_id = mujoco.mj_name2id(training.model, mujoco.mjtObj.mjOBJ_GEOM,
                                    "track_geom")
        for _ in range(4):
            training.reset()
            self.assertAlmostEqual(training.model.geom_friction[geom_id, 0],
                                   training.friction_gain, places=7)

    def test_local_training_friction_is_not_compounded_across_resets(self):
        # Catches cached hfield models accumulating multiplicative friction.
        env = Go2WEnv(seed=9000, train_stage="single_gap", gap_width_m=0.2,
                       train_track=TRACK, eval_mode=False)
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM,
                                    "track_geom")
        self.assertAlmostEqual(env.model.geom_friction[geom_id, 0],
                               env.friction_gain, places=7)
        for _ in range(10):
            env.reset()
            self.assertAlmostEqual(env.model.geom_friction[geom_id, 0],
                                   env.friction_gain, places=7)

    def test_flat_and_rough_starts_have_ground_contacts_at_requested_height(self):
        # Catches MuJoCo renormalizing a nearly constant hfield below z=0.
        for stage in ("flat", "rough_flat"):
            with self.subTest(stage=stage):
                env = Go2WEnv(seed=9000, train_stage=stage, eval_mode=True)
                self.assertGreater(env.data.qpos[2], 0.2)
                contacts, bad = env._contacts()
                self.assertGreaterEqual(contacts.sum(), 2)
                self.assertEqual(bad, 0.0)

    def test_flat_stage_has_only_flat_segments_and_original_default_stays_mixed(self):
        # Catches stage selection leaking into the existing default curriculum.
        flat = Go2WEnv(seed=9000, train_stage="flat", eval_mode=False)
        self.assertTrue(np.allclose(flat.H[:, 40], 0.0))
        self.assertIsNone(flat.fixed)
        flat_a = Go2WEnv(seed=9000, train_stage="flat", eval_mode=True)
        flat_b = Go2WEnv(seed=9001, train_stage="flat", eval_mode=True)
        self.assertNotEqual(flat_a.requested_initial_pose,
                            flat_b.requested_initial_pose)
        default = Go2WEnv(seed=9000, eval_mode=False)
        self.assertEqual(default.train_stage, "mixed")
        self.assertIsNone(default.fixed)
        with self.assertRaises(ValueError):
            Go2WEnv(seed=9000, train_stage="single_gap", train_track=TRACK,
                    gap_width_m=0.15)


    def test_surface_heightmap_matches_actual_mujoco_collision_surface_everywhere(self):
        # Catches 10cm hfield renormalization drift between flat training and fixed track.
        official = Go2WEnv(seed=9043, fixed_track=TRACK, eval_mode=True,
                           eval_protocol="pose-v1")
        flat = Go2WEnv(seed=9043, train_stage="flat", eval_mode=True)
        mixed = Go2WEnv(seed=9043, train_stage="mixed", eval_mode=False)
        for env in (official, flat, mixed):
            obs = env._get_obs()
            self.assertAlmostEqual(float(obs[6]), 0.347, delta=0.015)
            np.testing.assert_allclose(obs[7:11], 0.0, atol=0.005)
        np.testing.assert_allclose(official._get_obs()[6:11],
                                   flat._get_obs()[6:11], atol=0.005)


if __name__ == "__main__":
    unittest.main()
