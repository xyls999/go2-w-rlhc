import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from stage3_rl.ppo_train import (Actor, ExteroEncoder, TrainableResidualActor,
                                 BoundedBaseResidualActor, PROPRIO, EXTERO, GAIT,
                                 build_arg_parser)
from stage3_rl.evaluate import load_policy
from stage3_rl.env import transform_material_aux

ROOT=Path(__file__).resolve().parents[1]
DEC=ROOT/'stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz'

class TrainableResidualTests(unittest.TestCase):
    def test_float32_boundary_matches_historical_float64_clamp_exactly(self):
        raw=np.array([-.7,-.2,-.1,0,.1,.2,.8],dtype=np.float32)
        historical=transform_material_aux(raw.astype(np.float64),.2)
        v3=transform_material_aux(np.clip(raw,-.2,.2).astype(np.float64),4.5)
        np.testing.assert_array_equal(historical,v3)

    def test_v3_preserves_initial_effective_aux_but_allows_residual_motion(self):
        torch.manual_seed(12)
        model=BoundedBaseResidualActor()
        model.base_head.bias.data[16:28]=torch.linspace(-1.,1.,12)
        x=torch.randn(4,PROPRIO+96+GAIT)
        base,gate,_,_=model.forward_parts(x)
        mean=model.dist(x).mean
        torch.testing.assert_close(mean[:,:16],base[:,:16])
        torch.testing.assert_close(mean[:,16:28],base[:,16:28].clamp(-.2,.2))
        torch.testing.assert_close(mean[:,28:],base[:,28:])
        mean.sum().backward()
        self.assertGreater(float(model.res_head.weight.grad.abs().max()),0.)
        self.assertTrue(torch.all(gate>0))

    def test_new_architectures_can_be_selected_explicitly(self):
        for variant in ('v2','v3'):
            args=build_arg_parser().parse_args(['--out','unused','--probe-track','track.xml',f'--residual-{variant}'])
            self.assertTrue(getattr(args,f'residual_{variant}'))

    def test_initial_mean_matches_base_and_head_receives_gradient(self):
        torch.manual_seed(7)
        base=Actor();enc=ExteroEncoder();split=ExteroEncoder(split_frozen=True)
        split.net.load_state_dict(enc.net.state_dict())
        v2=TrainableResidualActor();v2.base_body.load_state_dict(base.body.state_dict());v2.base_head.load_state_dict(base.head.state_dict())
        obs=torch.randn(8,PROPRIO+EXTERO+GAIT)
        p=obs[:,:PROPRIO];e=obs[:,PROPRIO:PROPRIO+EXTERO];g=obs[:,PROPRIO+EXTERO:]
        expected=base.dist(torch.cat([p,enc(e),g],1)).mean
        actual=v2.dist(torch.cat([p,split(e),g],1)).mean
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        actual.sum().backward()
        self.assertGreater(float(v2.res_head.weight.grad.abs().max()),0.)

    def test_saved_arch_loads_the_same_version_not_legacy_residual(self):
        import hashlib
        for arch,cls in [('residual_v2',TrainableResidualActor),('residual_v3',BoundedBaseResidualActor)]:
            with self.subTest(arch=arch),tempfile.TemporaryDirectory() as td:
                enc=ExteroEncoder(split_frozen=True);actor=cls()
                actor.res_head.bias.data.fill_(0.5)
                ck={'enc':enc.state_dict(),'actor':actor.state_dict(),
                    'norm':{'mean':np.zeros(426),'var':np.ones(426)},'actor_arch':arch,
                    'actor_hidden':512,'decoder_sha256':hashlib.sha256(DEC.read_bytes()).hexdigest()}
                p=Path(td)/'model.pt';torch.save(ck,p)
                _,restored,_,_,_=load_policy(p,'cpu',str(DEC))
                self.assertIs(type(restored),cls)
                x=torch.randn(3,PROPRIO+96+GAIT)
                torch.testing.assert_close(restored.dist(x).mean,actor.dist(x).mean)

if __name__=='__main__':unittest.main()
