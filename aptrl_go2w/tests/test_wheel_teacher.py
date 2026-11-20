import unittest
import numpy as np
from stage3_rl.wheel_teacher import supported_lift_teacher_action

class WheelTeacherTests(unittest.TestCase):
    def test_action_is_bounded_slewed_and_separates_wheel_stop(self):
        raw=np.linspace(-.4,.4,29);prev=np.zeros(12);p=np.array([1.,2.,-4.,-1.,-.5,2.,-1.,-3.,0.,2.])
        action,next_prev=supported_lift_teacher_action(raw,prev,p,pulse=True,stop=True)
        self.assertEqual(action.shape,(30,))
        np.testing.assert_array_equal(action[:16],raw[:16])
        self.assertEqual(action[28],raw[28]);self.assertEqual(action[29],-1.)
        self.assertTrue(np.all(np.abs(next_prev-prev)<=.10000001))
        self.assertTrue(np.all(np.abs(next_prev)<=4.5))
        action2,_=supported_lift_teacher_action(raw,next_prev,p,pulse=False,stop=False)
        self.assertEqual(action2[29],0.)

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):supported_lift_teacher_action(np.zeros(28),np.zeros(12),np.zeros(10),pulse=True,stop=True)
        p=np.zeros(10);p[0]=np.nan
        with self.assertRaises(ValueError):supported_lift_teacher_action(np.zeros(29),np.zeros(12),p,pulse=True,stop=True)

if __name__=='__main__':unittest.main()
