import unittest
from stage3_rl.tire_reward import TireReward

class TireRewardTests(unittest.TestCase):
    def test_forward_progress_is_bounded_not_refarmed_and_never_paid_on_failure(self):
        r=TireReward();r.step(23.3,4,False)
        a=r.step(23.6,4,False);self.assertGreater(a['tire_progress'],0)
        self.assertLessEqual(a['tire_progress'],0.12)
        r.step(23.3,4,False)
        self.assertNotIn('tire_progress',r.step(23.6,4,False))
        self.assertNotIn('tire_progress',r.step(23.7,5,False))
        self.assertNotIn('tire_progress',r.step(23.8,4,True))

    def test_completion_requires_contact_and_all_wheels_and_is_one_shot(self):
        r=TireReward();r.step(23.4,4,False)
        self.assertNotIn('tire_clear',r.step(24.,4,False,tire_wheel_contact=False))
        r.step(24.5,4,False,tire_wheel_contact=True)
        self.assertNotIn('tire_clear',r.step(27.5,4,False,wheel_xy=[[27,4],[27,4],[26.9,4],[27,4]]))
        wheels=[[27,3.8],[27,4.2],[27,3.8],[27,4.2]]
        self.assertEqual(r.step(27.5,4,False,wheel_xy=wheels)['tire_clear'],2.0)
        self.assertNotIn('tire_clear',r.step(27.6,4,False,wheel_xy=wheels))

    def test_different_episode_resets_state(self):
        a=TireReward();a.step(23.4,4,False,tire_wheel_contact=True)
        b=TireReward();w=[[27,4]]*4
        self.assertNotIn('tire_clear',b.step(27.5,4,False,wheel_xy=w))

if __name__=='__main__':unittest.main()
