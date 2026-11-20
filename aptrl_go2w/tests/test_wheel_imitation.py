import unittest
import torch
from stage3_rl.train_wheel_imitation import weighted_imitation_loss

class WheelImitationTests(unittest.TestCase):
    def test_loss_is_zero_for_identical_actions_and_positive_for_errors(self):
        target=torch.zeros(3,30);weight=torch.tensor([1.,2.,6.]);wheel_weight=torch.tensor([1.,3.,3.])
        self.assertEqual(float(weighted_imitation_loss(target,target,weight,wheel_weight)),0.)
        pred=target.clone();pred[2,18]=2.;pred[1,29]=-1.
        self.assertGreater(float(weighted_imitation_loss(pred,target,weight,wheel_weight)),0.)
    def test_invalid_shapes_are_rejected(self):
        with self.assertRaises(ValueError):weighted_imitation_loss(torch.zeros(2,29),torch.zeros(2,30),torch.ones(2),torch.ones(2))

if __name__=='__main__':unittest.main()
