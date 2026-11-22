import tempfile
from pathlib import Path
import unittest
import numpy as np
from common.material_acceptance import (verify_target_scene, trajectory_digest,
                                        qualify_summary, reserve_attempt, compare_replay,
                                        generate_holdout_seeds, policy_material_aux_limit,
                                        policy_material_command_profile)
from common.eval_protocol import material_command_speed

ROOT=Path(__file__).resolve().parents[1]
SCENE=ROOT/'results/diagnostics/robocon-material-preview-20261119-v6/scene.xml'

class MaterialAcceptanceTests(unittest.TestCase):
    def test_only_exact_approved_scene_bytes_are_accepted(self):
        self.assertEqual(verify_target_scene(SCENE),'3c01e89a8a79871b87f9723051d1747562771d1a495597279e1da44598a163b2')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'modified.xml';p.write_bytes(SCENE.read_bytes()+b' ')
            with self.assertRaises(ValueError):verify_target_scene(p)

    def test_trajectory_hash_changes_when_any_state_changes(self):
        first=[np.array([1.,2.]),np.array([3.,4.])]
        same=[x.copy() for x in first]
        changed=[np.array([1.,2.]),np.array([3.,4.001])]
        self.assertEqual(trajectory_digest(first),trajectory_digest(same))
        self.assertNotEqual(trajectory_digest(first),trajectory_digest(changed))

    def test_acceptance_requires_100_unique_seeds_and_91_valid(self):
        rows=[{'seed':i,'success':i<91,'physics_success':i<91,
               'route':{'success':i<91, 'violations':[]}} for i in range(100)]
        self.assertTrue(qualify_summary(rows)['accepted'])
        self.assertFalse(qualify_summary(rows[:91])['accepted'])
        duplicate=[dict(x) for x in rows];duplicate[-1]['seed']=duplicate[0]['seed']
        self.assertFalse(qualify_summary(duplicate)['accepted'])
        insufficient=[dict(x) for x in rows];insufficient[90]['success']=False
        self.assertFalse(qualify_summary(insufficient)['accepted'])
        fake=[dict(x) for x in rows];fake[0]['route']={'success':False,'violations':['bridge_outside']}
        self.assertFalse(qualify_summary(fake)['accepted'])

    def test_real_material_short_rollout_cannot_be_accepted_and_replays(self):
        from stage3_rl.env import Go2WEnv
        from stage3_rl.material_acceptance import run_material_episode
        decoder=ROOT/'stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz'
        class ZeroRunner:
            def act(self, obs, deterministic=True):return np.zeros(29)
        def make():
            return Go2WEnv(seed=123, fixed_track=str(SCENE), eval_mode=True,
                           eval_protocol='pose-v1', episode_len=5, decoder_path=str(decoder))
        first=run_material_episode(make(),ZeroRunner(),max_steps=5)
        second=run_material_episode(make(),ZeroRunner(),max_steps=5)
        self.assertFalse(first['success'])
        self.assertFalse(first['route']['success'])
        self.assertEqual(first['trajectory_sha256'],second['trajectory_sha256'])
        self.assertGreater(first['steps'],0)

    def test_strict_rollout_accepts_only_matching_30_action_wheel_mode(self):
        from stage3_rl.env import Go2WEnv
        from stage3_rl.material_acceptance import run_material_episode
        decoder=ROOT/'stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz'
        class Runner:
            def __init__(self,n):self.n=n
            def act(self,obs,deterministic=True):return np.zeros(self.n)
        def make():
            return Go2WEnv(seed=123,fixed_track=str(SCENE),eval_mode=True,
                           eval_protocol='pose-v1',episode_len=5,decoder_path=str(decoder),
                           material_aux_limit=4.5,material_wheel_control='residual_v1')
        result=run_material_episode(make(),Runner(30),max_steps=5)
        self.assertFalse(result['success'])
        with self.assertRaises(ValueError):run_material_episode(make(),Runner(29),max_steps=5)

    def test_strict_rollout_uses_checkpoint_start_ramp(self):
        from stage3_rl.env import Go2WEnv
        from stage3_rl.material_acceptance import run_material_episode
        decoder=ROOT/'stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz'
        env=Go2WEnv(seed=123,fixed_track=str(SCENE),eval_mode=True,
                    eval_protocol='pose-v1',episode_len=5,decoder_path=str(decoder))
        class Runner:
            material_command_profile='start_ramp_100_v1'
            def __init__(self):self.commands=[]
            def act(self,obs,deterministic=True):self.commands.append(env.vx_cmd);return np.zeros(29)
        runner=Runner();run_material_episode(env,runner,max_steps=5)
        np.testing.assert_allclose(runner.commands,[0.5,0.51,0.52,0.53,0.54])

    def test_versioned_start_ramp_does_not_change_obstacle_command(self):
        self.assertEqual(policy_material_command_profile({}),'instant_v1')
        self.assertEqual(policy_material_command_profile({'material_command_profile':'start_ramp_100_v1'}),'start_ramp_100_v1')
        with self.assertRaises(ValueError):policy_material_command_profile({'material_command_profile':'unknown'})
        self.assertEqual(material_command_speed(0,0.5,'start_ramp_100_v1'),0.5)
        self.assertAlmostEqual(material_command_speed(99,1.,'start_ramp_100_v1'),1.49)
        self.assertEqual(material_command_speed(100,2.,'start_ramp_100_v1'),1.5)
        self.assertEqual(material_command_speed(1000,24.,'start_ramp_100_v1'),0.8)
        self.assertEqual(material_command_speed(0,0.5,'instant_v1'),1.5)

    def test_evaluation_requires_recorded_valid_material_action_limit(self):
        self.assertEqual(policy_material_aux_limit({'material_aux_limit':3.0}),3.0)
        self.assertIsNone(policy_material_aux_limit({'material_aux_limit':None}))
        for checkpoint in ({}, {'material_aux_limit':9.0}, {'material_aux_limit':0},
                           {'material_aux_limit':float('nan')}, {'material_aux_limit':True}):
            with self.assertRaises(ValueError):policy_material_aux_limit(checkpoint)

    def test_holdout_has_100_unique_high_entropy_nonnegative_seeds(self):
        seeds=generate_holdout_seeds()
        self.assertEqual(len(seeds),100)
        self.assertEqual(len(set(seeds)),100)
        self.assertTrue(all(isinstance(s,int) and 0 <= s < 2**128 for s in seeds))

    def test_replay_must_match_exact_successful_trajectory(self):
        original={'seed':7,'success':True,'physics_success':True,
                  'trajectory_sha256':'a'*64,'steps':50,'route':{'success':True,'violations':[]}}
        same=dict(original)
        self.assertTrue(compare_replay(original,same))
        changed={**same,'trajectory_sha256':'b'*64}
        self.assertFalse(compare_replay(original,changed))
        self.assertFalse(compare_replay(original,{**same,'success':False}))
        self.assertFalse(compare_replay(original,{**same,'seed':8}))

    def test_same_checkpoint_and_scene_cannot_be_retried_silently(self):
        with tempfile.TemporaryDirectory() as td:
            reserve_attempt(td,'a'*64,'b'*64)
            with self.assertRaises(FileExistsError):reserve_attempt(td,'a'*64,'b'*64)
            reserve_attempt(td,'c'*64,'b'*64)

if __name__=='__main__':unittest.main()
