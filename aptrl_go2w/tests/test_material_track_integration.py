"""Integration contract: approved material XML must be usable for training."""
from pathlib import Path
import unittest

import mujoco
from stage3_rl.env import Go2WEnv

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'results/diagnostics/robocon-material-preview-20261113-v4/scene.xml'
DECODER = ROOT / 'stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz'


class MaterialTrainingIntegrationTests(unittest.TestCase):
    def test_training_env_loads_preview_collision_geometry(self):
        env = Go2WEnv(seed=7, train_stage='full_track', train_track=str(SCENE),
                      episode_len=9000, decoder_path=str(DECODER))
        for name in ('tire_0_0_0', 'stair_up_0', 'balance_beam',
                     'bridge_barrier_left', 'clearance_top'):
            self.assertGreaterEqual(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name), 0, name)
        self.assertGreater(env.hfield.height(25.22, 3.725),
                           env.hfield.height(25.0, 3.725) + 0.05)
        self.assertAlmostEqual(env.hfield.height(31.16, 4), 0.12, places=3)
        self.assertAlmostEqual(env.hfield.height(32.44, 4), 0.60, places=3)
        self.assertAlmostEqual(env.hfield.height(36.7, 4), 0.10, places=3)
        self.assertAlmostEqual(env.hfield.height(45, 4), 0.0, places=3)
        self.assertGreater(env.material_track.height(45, 4, include_overhead=True), 0.25)

    def test_material_training_resets_only_on_safe_runups(self):
        env = Go2WEnv(seed=12, train_stage='full_track', train_track=str(SCENE),
                      episode_len=9000, decoder_path=str(DECODER), reward_profile='paper_adapted')
        starts = []
        for _ in range(20):
            env.reset()
            x, y, _ = env.requested_initial_pose
            starts.append(x)
            self.assertTrue(0.4 <= x <= 0.6 or 21.4 <= x <= 21.6, (x, y))
            self.assertAlmostEqual(env.hfield.height(x, y), 0.0, places=4)
        self.assertTrue(any(x < 1 for x in starts))
        self.assertTrue(any(x > 21 for x in starts))

    def test_stair_mix_starts_on_safe_approaches(self):
        env = Go2WEnv(seed=21, train_stage='full_track', train_track=str(SCENE),
                      episode_len=9000, decoder_path=str(DECODER), reward_profile='stair_v1',
                      material_start_profile='stair_mix')
        starts=[]
        for _ in range(30):
            env.reset()
            x,y,_=env.requested_initial_pose
            starts.append(x)
            self.assertTrue(any(abs(x-z)<0.051 for z in (0.5,21.5,30.2)))
            self.assertAlmostEqual(env.hfield.height(x,y),0.0,places=4)
        self.assertTrue(any(abs(x-30.2)<0.051 for x in starts))
        self.assertTrue(any(x<1 for x in starts))

    def test_dedicated_workers_reset_at_assigned_safe_start(self):
        for profile, expected in [('flat_only',0.5),('stair_only',30.2),('tire_only',21.5)]:
            env=Go2WEnv(seed=22,train_stage='full_track',train_track=str(SCENE),
                        episode_len=9000,decoder_path=str(DECODER),reward_profile='integrated_v1',
                        material_start_profile=profile)
            for _ in range(5):
                env.reset()
                self.assertAlmostEqual(env.requested_initial_pose[0],expected,delta=.051)

    def test_eval_can_start_at_stair_runup_without_training_randomization(self):
        env = Go2WEnv(seed=9143, train_stage='full_track', train_track=str(SCENE),
                      eval_mode=True, episode_len=1000, decoder_path=str(DECODER),
                      material_eval_start=30.2)
        env.reset()
        self.assertTrue(30.14 <= env.requested_initial_pose[0] <= 30.26)
        self.assertEqual(env.motor_gain, 1.0)

    def test_eval_can_start_at_tire_runup_without_training_randomization(self):
        env = Go2WEnv(seed=9043, train_stage='full_track', train_track=str(SCENE),
                      eval_mode=True, episode_len=1200, decoder_path=str(DECODER),
                      material_eval_start=21.5)
        env.reset()
        self.assertTrue(21.4 <= env.requested_initial_pose[0] <= 21.6)
        self.assertEqual(env.motor_gain, 1.0)
        self.assertTrue(env.eval_mode)

    def test_spatial_filter_matches_bruteforce_physics_trajectory(self):
        import numpy as np
        def make():
            return Go2WEnv(seed=9143,train_stage='full_track',train_track=str(SCENE),
                           material_eval_start=21.5,eval_mode=True,episode_len=1000,decoder_path=str(DECODER))
        fast,reference=make(),make()
        terrain=reference.material_track
        terrain._candidate_indices=lambda x,y,overhead: np.flatnonzero(np.ones(len(terrain.boxes),dtype=bool) if overhead else ~terrain._overhead)
        obs_fast,obs_reference=fast.reset(),reference.reset()
        np.testing.assert_array_equal(obs_fast,obs_reference)
        completed=0
        for step in range(40):
            action=np.zeros(29);action[:16]=np.sin(step*.1)*.1
            obs_fast,r_fast,d_fast,_=fast.step(action)
            obs_reference,r_reference,d_reference,_=reference.step(action)
            np.testing.assert_array_equal(obs_fast,obs_reference)
            np.testing.assert_array_equal(fast.data.qpos,reference.data.qpos)
            self.assertEqual(r_fast,r_reference)
            self.assertEqual(d_fast,d_reference)
            completed+=1
            if d_fast:break
        self.assertGreaterEqual(completed,20)

    def test_material_height_spatial_filter_retains_collision_accuracy(self):
        import numpy as np
        from common.material_track import MaterialTrack
        terrain=MaterialTrack(str(SCENE))
        self.assertEqual(len(terrain._candidate_indices(np.array([.5]),np.array([4.]),True)),0)
        self.assertGreater(len(terrain._candidate_indices(np.array([25.22]),np.array([3.725]),True)),0)
        model=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        rng=np.random.default_rng(1543);x=rng.uniform(24,46,128);y=rng.uniform(3.5,4.5,128)
        calculated=terrain.height(x,y,include_overhead=True)
        for i in range(len(x)):
            distance=mujoco.mj_ray(model,data,np.array([x[i],y[i],2.]),np.array([0.,0.,-1.]),None,True,-1,None)
            self.assertGreaterEqual(distance,0)
            self.assertAlmostEqual(calculated[i],2-distance,places=5)
            self.assertAlmostEqual(terrain.height(x[i],y[i],include_overhead=True),calculated[i],places=8)

    def test_height_query_matches_actual_mujoco_rays(self):
        import numpy as np
        from common.material_track import MaterialTrack
        terrain = MaterialTrack(str(SCENE))
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for x, y in [(25, 3.725), (25.22, 3.725), (31.16, 4),
                     (32.44, 4), (33.8, 4), (34.6, 4), (36.7, 4),
                     (39.7, 4), (45, 4)]:
            distance = mujoco.mj_ray(model, data, np.array([x, y, 2.0]),
                                     np.array([0., 0., -1.]), None, True, -1, None)
            self.assertGreaterEqual(distance, 0)
            self.assertAlmostEqual(terrain.height(x, y, include_overhead=True),
                                   2 - distance, places=5)

    def test_material_auxiliary_limit_can_be_disabled_for_diagnostic(self):
        import numpy as np
        env = Go2WEnv(seed=7, train_stage='full_track', train_track=str(SCENE),
                      episode_len=9000, decoder_path=str(DECODER), material_aux_limit=None)
        action = np.zeros(29)
        action[16:28] = 0.5
        env.step(action)
        self.assertAlmostEqual(float(env.a_aux[0]), 0.5)

    def test_material_auxiliary_actions_are_limited(self):
        import numpy as np
        env = Go2WEnv(seed=7, train_stage='full_track', train_track=str(SCENE),
                      episode_len=9000, decoder_path=str(DECODER))
        action = np.zeros(29)
        action[16:28] = 5.0
        env.step(action)
        self.assertLessEqual(float(np.max(np.abs(env.a_aux))), 0.200001)


if __name__ == '__main__':
    unittest.main()
