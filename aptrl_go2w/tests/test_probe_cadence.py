"""Short PPO pilots must probe on their final (25th) update."""
import unittest

from stage3_rl import ppo_train


class ProbeCadenceTests(unittest.TestCase):
    def test_first_25_update_pilot_probes_exactly_once(self):
        # Catches the previous zero-based off-by-one that made no best checkpoint.
        self.assertEqual([i for i in range(25) if ppo_train.should_probe(i)], [24])
        self.assertTrue(ppo_train.should_probe(49))
        self.assertFalse(ppo_train.should_probe(0))
        self.assertFalse(ppo_train.should_probe(23))

    def test_invalid_interval_fails_closed(self):
        with self.assertRaises(ValueError):
            ppo_train.should_probe(24, 0)

    def test_training_probe_protocol_is_explicit(self):
        # Catches an apparently varied evaluator paired with legacy training probes.
        args = ppo_train.build_arg_parser().parse_args([
            "--out", "fresh", "--probe-track", "track.npz",
            "--eval-protocol", "pose-v1",
        ])
        self.assertEqual(args.eval_protocol, "pose-v1")


if __name__ == "__main__":
    unittest.main()
