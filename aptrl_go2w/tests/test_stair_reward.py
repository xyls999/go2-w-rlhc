import unittest
from stage3_rl.stair_reward import StairReward

class StairRewardTests(unittest.TestCase):
    def test_progress_is_bounded_and_cannot_be_farmed_by_backtracking(self):
        r=StairReward();r.step(30.9,4,1.0,False)
        first=r.step(31.1,4,1.0,False)
        self.assertGreater(first.get('stair_progress',0),0)
        self.assertLessEqual(first['stair_progress'],0.03)
        r.step(30.9,4,1.0,False)
        self.assertNotIn('stair_progress',r.step(31.1,4,1.0,False))
        self.assertNotIn('stair_progress',r.step(31.2,5,1.0,False))
        self.assertNotIn('stair_progress',r.step(31.3,4,1.0,True))

    def test_clear_requires_entry_all_wheels_and_is_one_shot(self):
        r=StairReward();r.step(30.9,4,1.,False)
        r.step(31.1,4,1.,False,wheel_x=[31,31,30.8,30.8])
        self.assertNotIn('stair_clear',r.step(33.5,4,1.,False,wheel_x=[33.4,33.4,32.4,32.4],wheel_y=[4]*4))
        self.assertNotIn('stair_clear',r.step(33.6,4,1.,False,wheel_x=[33.4,33.4,33,33],wheel_y=[4,4,4,4.45]))
        self.assertEqual(r.step(33.6,4,1.,False,wheel_x=[33.4,33.4,33,33],wheel_y=[4]*4)['stair_clear'],1.)
        self.assertNotIn('stair_clear',r.step(33.7,4,1.,False,wheel_x=[33.5]*4,wheel_y=[4]*4))

    def test_stuck_is_time_bounded_and_only_when_commanded_in_region(self):
        r=StairReward()
        for _ in range(199):self.assertNotIn('stair_stuck',r.step(31.1,4,1.,False))
        self.assertEqual(r.step(31.1,4,1.,False)['stair_stuck'],-0.2)
        self.assertNotIn('stair_stuck',r.step(31.1,4,0.,False))

    def test_lift_requires_actual_contact_forward_motion_and_is_one_shot(self):
        r=StairReward();r.step(31.1,4,1.,False,wheel_z=[.09]*4)
        self.assertNotIn('stair_recovery_lift',r.step(31.2,4,1.,False,wheel_z=[.2]*4))
        r.step(31.2,4,1.,False,wheel_z=[.1]*4,stair_wheel_contacts=[0])
        self.assertNotIn('stair_recovery_lift',r.step(31.21,4,1.,False,wheel_z=[.2]*4))
        self.assertEqual(r.step(31.26,4,1.,False,wheel_z=[.2]*4)['stair_recovery_lift'],0.05)
        r.step(31.3,4,1.,False,wheel_z=[.1]*4,stair_wheel_contacts=[1])
        self.assertNotIn('stair_recovery_lift',r.step(31.4,4,1.,False,wheel_z=[.2]*4))

if __name__=='__main__':unittest.main()
