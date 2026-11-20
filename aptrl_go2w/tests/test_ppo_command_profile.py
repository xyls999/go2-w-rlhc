import unittest
from types import SimpleNamespace
from stage3_rl.ppo_train import apply_material_start_profile

class PPOCommandProfileTests(unittest.TestCase):
    def test_ramp_applies_only_to_from_start_material_episodes(self):
        e=SimpleNamespace(material_track=object(),requested_initial_pose=[.5,4,0],steps=0,
                          data=SimpleNamespace(qpos=[.5]),vx_cmd=1.5)
        apply_material_start_profile(e,'start_ramp_100_v1');self.assertEqual(e.vx_cmd,.5)
        e.steps=40;apply_material_start_profile(e,'start_ramp_100_v1');self.assertEqual(e.vx_cmd,.9)
        e.steps=100;apply_material_start_profile(e,'start_ramp_100_v1');self.assertEqual(e.vx_cmd,1.5)
        e.requested_initial_pose=[21.5,4,0];e.steps=0;e.vx_cmd=.8
        apply_material_start_profile(e,'start_ramp_100_v1');self.assertEqual(e.vx_cmd,.8)

if __name__=='__main__':unittest.main()
