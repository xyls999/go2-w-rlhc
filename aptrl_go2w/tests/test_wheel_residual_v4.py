import unittest
import tempfile
import hashlib
import numpy as np
import torch
from pathlib import Path
from common.material_acceptance import policy_material_wheel_control
from stage3_rl.env import material_wheel_scale, Go2WEnv
from stage3_rl.ppo_train import (BoundedBaseResidualActor, WheelResidualActor,
                                 ExteroEncoder, build_arg_parser, clamp_wheel_log_std, PROPRIO, GAIT)
from stage3_rl.evaluate import PolicyRunner

class WheelResidualV4Tests(unittest.TestCase):
    def test_v4_cli_and_checkpoint_loader_preserve_provenance(self):
        args=build_arg_parser().parse_args(['--out','unused','--probe-track','track.xml','--residual-v4'])
        self.assertTrue(args.residual_v4)
        root=Path(__file__).resolve().parents[1]
        dec=root/'stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz'
        ck={'enc':ExteroEncoder(split_frozen=True).state_dict(),
            'actor':WheelResidualActor().state_dict(),'actor_arch':'residual_v4',
            'material_wheel_control':'residual_v1','material_aux_limit':4.5,
            'decoder_sha256':hashlib.sha256(dec.read_bytes()).hexdigest(),
            'norm':{'mean':np.zeros(426),'var':np.ones(426)}}
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'model.pt';torch.save(ck,p)
            runner=PolicyRunner(str(p),'cpu',str(dec))
            self.assertIs(type(runner.actor),WheelResidualActor)
            self.assertEqual(runner.material_wheel_control,'residual_v1')
            self.assertEqual(runner.material_aux_limit,4.5)
            self.assertEqual(runner.act(np.zeros((1,426),dtype=np.float32)).shape,(30,))
            del ck['material_wheel_control'];torch.save(ck,p)
            with self.assertRaises(ValueError):PolicyRunner(str(p),'cpu',str(dec))

    def test_wheel_exploration_std_is_bounded(self):
        actor=WheelResidualActor()
        with torch.no_grad():actor.wheel_log_std.fill_(np.log(10.))
        clamp_wheel_log_std(actor)
        self.assertAlmostEqual(float(actor.wheel_log_std.exp()),2.,places=5)
        with torch.no_grad():actor.wheel_log_std.fill_(np.log(.001))
        clamp_wheel_log_std(actor)
        self.assertAlmostEqual(float(actor.wheel_log_std.exp()),.14,places=5)

    def test_wheel_scaling_is_bounded_and_zero_preserves_base(self):
        self.assertEqual(material_wheel_scale(0.),1.)
        self.assertEqual(material_wheel_scale(-1.),0.)
        self.assertEqual(material_wheel_scale(-3.),0.)
        self.assertEqual(material_wheel_scale(.5),1.5)
        self.assertEqual(material_wheel_scale(3.),1.5)
        with self.assertRaises(ValueError):material_wheel_scale(float('nan'))

    def test_checkpoint_requires_matching_control_provenance(self):
        self.assertEqual(policy_material_wheel_control({}),'none')
        self.assertEqual(policy_material_wheel_control({'actor_arch':'residual_v4','material_wheel_control':'residual_v1'}),'residual_v1')
        for ck in ({'actor_arch':'residual_v4'}, {'material_wheel_control':'residual_v1'},
                   {'actor_arch':'residual_v4','material_wheel_control':'none'}):
            with self.assertRaises(ValueError):policy_material_wheel_control(ck)

    def test_zero_wheel_residual_has_identical_simulation_to_legacy_control(self):
        root=Path(__file__).resolve().parents[1]
        track=root/'results/diagnostics/robocon-material-preview-20261113-v4/scene.xml'
        dec=root/'stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz'
        envs=[Go2WEnv(seed=17,fixed_track=str(track),eval_mode=True,episode_len=1000,
                      decoder_path=str(dec),material_wheel_control=m) for m in ('none','residual_v1')]
        for e in envs:e.reset()
        for _ in range(20):
            for e,a in zip(envs,(np.zeros(29),np.zeros(30))):e.step(a)
            np.testing.assert_array_equal(envs[0].data.qpos,envs[1].data.qpos)
            np.testing.assert_array_equal(envs[0].data.qvel,envs[1].data.qvel)
        with self.assertRaises(ValueError):envs[0].step(np.zeros(30))
        with self.assertRaises(ValueError):envs[1].step(np.zeros(29))

    def test_initial_actor_retains_v3_actions_and_has_trainable_wheel_head(self):
        torch.manual_seed(23)
        base=BoundedBaseResidualActor();v4=WheelResidualActor()
        missing,unexpected=v4.load_state_dict(base.state_dict(),strict=False)
        self.assertEqual(set(missing),{'wheel_head.weight','wheel_head.bias','wheel_log_std'})
        self.assertFalse(unexpected)
        x=torch.randn(8,PROPRIO+96+GAIT)
        a=base.dist(x).mean;b=v4.dist(x).mean
        self.assertEqual(b.shape[1],30)
        torch.testing.assert_close(a,b[:,:29],rtol=0,atol=0)
        torch.testing.assert_close(b[:,29],torch.zeros(8),rtol=0,atol=0)
        b[:,29].sum().backward()
        self.assertGreater(float(v4.wheel_head.weight.grad.abs().max()),0.)

if __name__=='__main__':unittest.main()
