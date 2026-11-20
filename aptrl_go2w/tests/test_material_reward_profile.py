import unittest
from pathlib import Path
import numpy as np
from stage3_rl.env import Go2WEnv
from stage3_rl.ppo_train import build_arg_parser, worker_env_config, summarize_reward_infos, validate_training_args, material_start_for_worker

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'results/diagnostics/robocon-material-preview-20261113-v4/scene.xml'
DECODER = ROOT / 'stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz'

class MaterialRewardProfileTests(unittest.TestCase):
    def test_explicit_profile_removes_legacy_progress_and_scales_velocity(self):
        env = Go2WEnv(seed=3, train_stage='full_track', train_track=str(SCENE),
                      episode_len=9000, decoder_path=str(DECODER), reward_profile='paper_adapted')
        _, terms = env._reward()
        for key in ('forward_progress', 'alive', 'new_max_x', 'milestone'):
            self.assertNotIn(key, terms)
        self.assertGreater(terms['lin_vel'], 0)
        self.assertLessEqual(terms['lin_vel'], 1.5)

    def test_profile_rejects_non_material_map(self):
        with self.assertRaises(ValueError):
            Go2WEnv(seed=3, train_stage='flat', reward_profile='paper_adapted')

    def test_summary_records_reward_components_and_terminations(self):
        infos = [{'terms': {'lin_vel': 1.0, 'aux_smooth': -0.2}, 'reason': None, '_diagnostic_x': 21.5, '_diagnostic_start_x': 21.5},
                 {'terms': {'lin_vel': 0.5, 'termination': -1.5}, 'reason': 'body_contact', '_diagnostic_x': 24.4, '_diagnostic_start_x': 21.5}]
        summary = summarize_reward_infos(infos)
        self.assertEqual(summary['steps'], 2)
        self.assertAlmostEqual(summary['terms_sum']['lin_vel'], 1.5)
        self.assertAlmostEqual(summary['terms_sum']['termination'], -1.5)
        self.assertEqual(summary['terminations'], {'body_contact': 1})
        self.assertAlmostEqual(summary['max_x'], 24.4)
        self.assertEqual(summary['tire_runup_steps'], 2)
        self.assertEqual(summary['stair_runup_steps'], 0)
        stairs = summarize_reward_infos([{'_diagnostic_x': 30.3, '_diagnostic_start_x': 30.2}])
        self.assertEqual(stairs['stair_runup_steps'], 1)

    def test_tire_profile_has_dense_terms_and_explicit_shaper(self):
        env=Go2WEnv(seed=3,train_stage='full_track',train_track=str(SCENE),
                    episode_len=9000,decoder_path=str(DECODER),reward_profile='tire_v1')
        _,terms=env._reward()
        self.assertEqual(terms['alive'],0.5)
        self.assertIn('forward_progress',terms)
        self.assertTrue(hasattr(env,'tire_shaper'))
        self.assertNotIn('milestone',terms)

    def test_tire_profile_calls_shaper_after_physics_and_termination(self):
        class Stub:
            def __init__(self):self.called=None;self.kwargs=None
            def step(self,*args,**kwargs):
                self.called=args;self.kwargs=kwargs
                return {'tire_progress':0.01} if not args[2] else {}
        env=Go2WEnv(seed=3,train_stage='full_track',train_track=str(SCENE),
                    episode_len=9000,decoder_path=str(DECODER),reward_profile='tire_v1')
        stub=Stub();env.tire_shaper=stub
        _,_,done,info=env.step(np.zeros(29))
        self.assertIsNotNone(stub.called)
        self.assertEqual(stub.called[2],done)
        self.assertEqual(stub.kwargs['wheel_xy'].shape,(4,2))
        if not done:self.assertEqual(info['terms']['tire_progress'],0.01)

    def test_tire_l1_profile_requires_exact_l1_and_exercises_shaper(self):
        l1=ROOT/'stage3_rl/experiments/20261113T160601Z/actor-bootstrap/material-tire-levels-v1/tire-level-1.xml'
        env=Go2WEnv(seed=3,train_stage='full_track',train_track=str(l1),
                    episode_len=9000,decoder_path=str(DECODER),reward_profile='tire_l1_v1')
        self.assertTrue(hasattr(env,'tire_shaper'))
        self.assertEqual(env._reward()[1]['alive'],0.5)
        with self.assertRaises(ValueError):
            Go2WEnv(seed=3,train_stage='full_track',train_track=str(SCENE),
                    episode_len=9000,decoder_path=str(DECODER),reward_profile='tire_l1_v1')
        with self.assertRaises(ValueError):
            Go2WEnv(seed=3,train_stage='full_track',train_track=str(l1),
                    episode_len=9000,decoder_path=str(DECODER),reward_profile='tire_v1')

    def test_tire_profile_rejects_changed_scene(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'modified.xml';p.write_bytes(SCENE.read_bytes()+b' ')
            with self.assertRaises(ValueError):
                Go2WEnv(seed=3,train_stage='full_track',train_track=str(p),
                        episode_len=9000,decoder_path=str(DECODER),reward_profile='tire_v1')

    def test_integrated_profile_adds_dense_terms_without_old_milestones(self):
        env=Go2WEnv(seed=3,train_stage='full_track',train_track=str(SCENE),
                    episode_len=9000,decoder_path=str(DECODER),reward_profile='integrated_v1')
        _,terms=env._reward()
        self.assertEqual(terms['alive'],0.5)
        self.assertIn('forward_progress',terms)
        self.assertGreaterEqual(terms['forward_progress'],-0.5)
        self.assertLessEqual(terms['forward_progress'],1.5)
        self.assertNotIn('milestone',terms)
        self.assertNotIn('new_max_x',terms)
        self.assertLessEqual(terms['lin_vel'],1.5)

    def test_integrated_profile_has_no_dense_training_bonus_in_eval(self):
        env=Go2WEnv(seed=3,fixed_track=str(SCENE),eval_mode=True,episode_len=1000,
                    decoder_path=str(DECODER),reward_profile='integrated_v1')
        _,terms=env._reward()
        self.assertNotIn('alive',terms)
        self.assertNotIn('forward_progress',terms)
        self.assertNotIn('milestone',terms)

    def test_stair_profile_rejects_changed_stair_layout(self):
        import tempfile, xml.etree.ElementTree as ET
        tree = ET.parse(SCENE)
        step = next(g for g in tree.getroot().iter('geom') if g.get('name') == 'stair_up_0')
        step.set('pos', '10 4 0.06')
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'modified.xml'
            tree.write(p, encoding='unicode')
            with self.assertRaises(ValueError):
                Go2WEnv(seed=3, train_stage='full_track', train_track=str(p),
                        episode_len=9000, decoder_path=str(DECODER), reward_profile='stair_v1')

    def test_stair_profile_calls_shaper_after_terminal_check(self):
        class Stub:
            def __init__(self): self.called = None
            def step(self, *args, **kwargs):
                self.called = args
                return {'stair_progress': 0.01} if not args[3] else {}
        env = Go2WEnv(seed=3, train_stage='full_track', train_track=str(SCENE),
                      episode_len=9000, decoder_path=str(DECODER), reward_profile='stair_v1')
        stub = Stub()
        env.stair_shaper = stub
        _, _, done, info = env.step(np.zeros(29))
        self.assertIsNotNone(stub.called)
        self.assertEqual(stub.called[3], done)
        if not done: self.assertEqual(info['terms']['stair_progress'], 0.01)

    def test_stair_profile_rejects_non_material_track_before_worker_launch(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.npz') as f:
            args = build_arg_parser().parse_args(['--out', 'unused', '--probe-track', str(SCENE),
                     '--train-stage', 'full_track', '--train-track', f.name,
                     '--episode-len', '9000', '--reward-profile', 'stair_v1'])
            with self.assertRaises(ValueError): validate_training_args(args)

    def test_tire_retention_allocates_two_flat_two_tire_workers(self):
        self.assertEqual([material_start_for_worker('tire_retain',i,4) for i in range(4)],
                         ['flat_only','tire_only','flat_only','tire_only'])
        with self.assertRaises(ValueError):material_start_for_worker('tire_retain',0,2)

    def test_full_course_starts_every_worker_at_start_line(self):
        self.assertEqual([material_start_for_worker('full_start', i, 4) for i in range(4)],
                         ['flat_only'] * 4)
        args = build_arg_parser().parse_args(['--out','unused','--probe-track',str(SCENE),
                   '--train-stage','full_track','--train-track',str(SCENE),
                   '--material-start-profile','full_start'])
        self.assertEqual(worker_env_config(args)['material_start_profile'],'full_start')

    def test_stair_retention_allocates_two_flat_two_stair_workers(self):
        self.assertEqual([material_start_for_worker('stair_retain', i, 4) for i in range(4)],
                         ['flat_only','stair_only','flat_only','stair_only'])
        with self.assertRaises(ValueError): material_start_for_worker('stair_retain', 0, 2)

    def test_stair_start_profile_passes_to_workers(self):
        args=build_arg_parser().parse_args(['--out','unused','--probe-track',str(SCENE),
             '--train-stage','full_track','--train-track',str(SCENE),
             '--reward-profile','stair_v1','--material-start-profile','stair_mix'])
        self.assertEqual(worker_env_config(args)['material_start_profile'],'stair_mix')

    def test_cli_passes_profile_to_workers(self):
        args = build_arg_parser().parse_args(['--out', 'unused', '--probe-track', str(SCENE),
                 '--train-stage', 'full_track', '--train-track', str(SCENE),
                 '--reward-profile', 'paper_adapted'])
        self.assertEqual(worker_env_config(args)['reward_profile'], 'paper_adapted')

if __name__ == '__main__': unittest.main()
