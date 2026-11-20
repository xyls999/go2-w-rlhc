"""Physical admissibility contracts for the flat-ground TO prior."""
import unittest

import numpy as np

from stage1_to.trajectory_opt import (ForceSchedule, FootPlan, GaitSpec,
                                     assemble_traj, ensure_empty_output_dir,
                                     gen_traj, sample_trajectory_params,
                                     speeds_for_gait, simulate_cycle, solve_periodic,
                                     trajectory_quality, trot_foot_targets)


class TrajectoryOptimizationTests(unittest.TestCase):
    def test_bound_prior_sampling_covers_task_speed_with_short_feasible_cycle(self):
        # Keep task-speed bound samples while avoiding unreachable long strides.
        self.assertEqual(speeds_for_gait("bound"), [-0.5, 0.5, 1.0, 1.5])
        self.assertEqual(speeds_for_gait("trot"), [-0.5, 0.5, 1.0, 1.5])
        params = sample_trajectory_params("bound", 1.5, np.random.default_rng(4))
        self.assertTrue(0.13 <= params[0] <= 0.15)
        self.assertTrue(0.15 <= params[1] <= 0.17)
        self.assertTrue(0.6 <= params[2] <= 1.0)

    def test_existing_dataset_directory_is_never_overwritten(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            output.mkdir()
            protected = output / "dataset_trot.npz"
            protected.write_bytes(b"prior immutable dataset")
            with self.assertRaises(FileExistsError):
                ensure_empty_output_dir(output)
            self.assertEqual(protected.read_bytes(), b"prior immutable dataset")

    def bound_fixture(self):
        spec = GaitSpec("bound", 0.16, 0.18, vz_scale=0.8)
        schedule = ForceSchedule(spec, fx_amp=0.0,
                                 fz_jitter=0.0)
        feet = FootPlan(schedule, 1.5)
        return schedule, feet

    def test_trot_keeps_same_side_wheels_from_overlapping(self):
        spec = GaitSpec("trot", 0.30, 0.0)
        schedule = ForceSchedule(spec, fx_amp=0.0, fz_jitter=0.0)
        feet = FootPlan(schedule, 1.5)
        min_separation = float("inf")
        for t in np.arange(0.0, spec.T, 0.01):
            _, x, z, _, _ = trot_foot_targets(t, feet)
            min_separation = min(min_separation,
                                 float(np.hypot(x[0] - x[2], z[0] - z[2])),
                                 float(np.hypot(x[1] - x[3], z[1] - z[3])))
        self.assertGreaterEqual(min_separation, 2 * 0.086)

    def test_trot_dataset_has_alternating_diagonal_contacts(self):
        # A trot prior must not label every timestep as four-foot standing.
        spec = GaitSpec("trot", 0.30, 0.0)
        schedule = ForceSchedule(spec, fx_amp=0.0, fz_jitter=0.0)
        feet = FootPlan(schedule, 1.0)
        q0 = solve_periodic(schedule, feet, lam2=0.0, z0=0.30)
        traj = assemble_traj(schedule, feet, q0, "trot")
        contacts = np.unique(traj["state"][:, 14:18], axis=0)
        expected = np.array([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(contacts, expected)

    def test_bound_cycle_closes_in_height_without_net_vertical_drift(self):
        # Rejects a nominally periodic skill that loses 8cm altitude per cycle.
        schedule, feet = self.bound_fixture()
        q0 = solve_periodic(schedule, feet, lam2=1.0, z0=0.28,
                            th0=0.0, thd0=0.0)
        z_end = simulate_cycle(schedule, feet, q0)[0]
        self.assertLess(abs(z_end - q0[0]), 0.005)

    def test_bound_inverse_kinematics_uses_wheel_axle_height(self):
        # Contact means the wheel axle is 0.086m above ground, not at ground.
        _, feet = self.bound_fixture()
        q0 = solve_periodic(feet.spec, feet, lam2=1.0, z0=0.28,
                            th0=0.0, thd0=0.0)
        traj = assemble_traj(feet.spec, feet, q0, "bound")
        self.assertTrue(traj["diagnostics"]["all_targets_reachable"])
        self.assertAlmostEqual(traj["state"][0, 10], 0.0, places=6)
        self.assertFalse(trajectory_quality(feet.spec, feet, q0, traj)["admissible"])

    def test_generator_keeps_a_periodic_reachable_bound_skill(self):
        # A seeded physically admissible sample survives the new TO quality gate.
        seed = 20
        rng = np.random.default_rng(seed)
        vx = float(rng.uniform(0.5, 3.0))
        t_st = float(rng.uniform(0.13, 0.20))
        t_sw = float(rng.uniform(0.15, 0.21))
        vz_scale = float(rng.uniform(0.6, 1.0))
        traj = gen_traj("bound", vx, t_st, t_sw, vz_scale, seed)
        self.assertTrue(traj["diagnostics"]["cycle_quality"]["admissible"])
        self.assertTrue(traj["diagnostics"]["all_targets_reachable"])
        self.assertLess(abs(traj["diagnostics"]["cycle_quality"]["pitch_rate_drift_rad_s"]), 0.2)


if __name__ == "__main__":
    unittest.main()
