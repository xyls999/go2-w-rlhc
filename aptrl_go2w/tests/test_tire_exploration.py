import unittest
import numpy as np
from stage3_rl.tire_exploration import bounded_group_action

class TireExplorationTests(unittest.TestCase):
    def test_offsets_respect_absolute_and_per_step_bounds(self):
        action=np.zeros(29);action[16:28]=8
        previous=np.zeros(12)
        out,applied=bounded_group_action(action,[3,-3,3,-3],previous)
        self.assertTrue(np.all(np.abs(applied)<=3))
        self.assertTrue(np.all(np.abs(applied-previous)<=0.1000001))
        np.testing.assert_array_equal(out[:16],action[:16])
        self.assertEqual(out[28],action[28])
        np.testing.assert_allclose(out[16:28],applied)

    def test_wider_diagnostic_limit_is_explicit_and_preserves_joint_targets(self):
        action=np.zeros(29);previous=np.full(12,-4.4)
        out,applied=bounded_group_action(action,[-4.5]*4,previous,absolute_limit=4.5)
        self.assertTrue(np.all(np.abs(applied)<=4.5))
        self.assertTrue(np.all(np.abs(applied-previous)<=0.1000001))
        calf_targets=-1.8+0.2*applied[[2,5,8,11]]
        self.assertTrue(np.all(calf_targets>=-2.7227))
        self.assertTrue(np.all(calf_targets<=-0.83776))

    def test_rejects_invalid_shapes_and_nonfinite_inputs(self):
        with self.assertRaises(ValueError):bounded_group_action(np.zeros(28),[0]*4,np.zeros(12))
        with self.assertRaises(ValueError):bounded_group_action(np.zeros(29),[float('nan')]*4,np.zeros(12))
        with self.assertRaises(ValueError):bounded_group_action(np.zeros(29),[0]*4,np.zeros(11))

if __name__=='__main__':unittest.main()
