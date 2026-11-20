import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stage3_rl.actor_bootstrap import (
    save_bootstrap_checkpoint,
    train_bootstrap_actor,
)
from stage3_rl.evaluate import PolicyRunner
from stage3_rl.ppo_train import Actor, ExteroEncoder


class ActorBootstrapTrainingTests(unittest.TestCase):
    def make_demo(self):
        rng = np.random.default_rng(8)
        observations = rng.normal(size=(6, 8, 426)).astype(np.float32)
        actions = np.zeros((6, 8, 29), dtype=np.float32)
        for episode in range(6):
            actions[episode, :, :16] = episode / 5.0
            actions[episode, :, 28] = -8.0
        return {"observation": observations, "action": actions}

    def test_training_splits_by_episode_and_reports_each_action_head(self):
        demos = self.make_demo()

        encoder, actor, norm, metrics = train_bootstrap_actor(
            demos, seed=42, epochs=1, batch_size=16, lr=1e-3, device="cpu"
        )

        self.assertIsInstance(encoder, ExteroEncoder)
        self.assertIsInstance(actor, Actor)
        self.assertEqual(norm.mean.shape, (426,))
        train_ids = set(metrics["train_episode_indices"])
        val_ids = set(metrics["validation_episode_indices"])
        self.assertFalse(train_ids & val_ids)
        self.assertEqual(train_ids | val_ids, set(range(6)))
        self.assertIn("latent_mse", metrics["validation"])
        self.assertIn("auxiliary_mse", metrics["validation"])
        self.assertIn("gait_bce", metrics["validation"])

    def test_checkpoint_is_ppo_loadable_and_rejects_other_decoder(self):
        encoder = ExteroEncoder()
        actor = Actor()
        from stage3_rl.ppo_train import RunningNorm
        norm = RunningNorm(426)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            decoder = root / "decoder.npz"
            other_decoder = root / "other.npz"
            np.savez(decoder, arbitrary=np.array([1]))
            np.savez(other_decoder, arbitrary=np.array([2]))
            checkpoint = root / "bootstrap.pt"

            save_bootstrap_checkpoint(
                checkpoint, encoder=encoder, actor=actor, norm=norm,
                decoder_path=decoder,
                metrics={"validation": {"latent_mse": 0.1}},
                provenance={"demonstrations_sha256": "a" * 64},
            )

            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(saved["iter"], 0)
            self.assertEqual(saved["decoder_sha256"], hashlib.sha256(decoder.read_bytes()).hexdigest())
            self.assertEqual(saved["bootstrap_provenance"]["demonstrations_sha256"], "a" * 64)
            self.assertIn("critic", saved)
            self.assertIn("norm", saved)
            self.assertEqual(PolicyRunner(str(checkpoint), device="cpu",
                                          decoder_path=str(decoder)).act(np.zeros((1, 426), np.float32)).shape,
                             (29,))
            with self.assertRaisesRegex(ValueError, "hashes do not match"):
                PolicyRunner(str(checkpoint), device="cpu", decoder_path=str(other_decoder))


if __name__ == "__main__":
    unittest.main()
