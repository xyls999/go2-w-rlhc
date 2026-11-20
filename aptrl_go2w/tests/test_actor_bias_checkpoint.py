import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stage3_rl.actor_bootstrap import offset_aux_bias_checkpoint


class AuxiliaryBiasCheckpointTests(unittest.TestCase):
    def test_bias_checkpoint_changes_only_requested_action_biases_and_records_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            decoder = root / "decoder.npz"
            np.savez(decoder, x=np.array([1]))
            source = root / "source.pt"
            output = root / "biased.pt"
            actor_bias = torch.arange(29, dtype=torch.float32)
            ck = {
                "enc": {"dummy": torch.tensor([1.0])},
                "actor": {"head.bias": actor_bias.clone(), "other": torch.tensor([2.0])},
                "critic": {"dummy": torch.tensor([3.0])},
                "norm": {"mean": np.zeros(426), "var": np.ones(426), "count": 1.0},
                "iter": 24,
                "decoder_sha256": hashlib.sha256(decoder.read_bytes()).hexdigest(),
            }
            torch.save(ck, source)

            offset_aux_bias_checkpoint(
                source, output, decoder_path=decoder,
                action_indices=(18, 21, 24, 27), delta=-1.0,
            )

            saved = torch.load(output, map_location="cpu", weights_only=False)
            expected = actor_bias.clone()
            expected[[18, 21, 24, 27]] -= 1.0
            torch.testing.assert_close(saved["actor"]["head.bias"], expected)
            torch.testing.assert_close(saved["actor"]["other"], ck["actor"]["other"])
            self.assertEqual(saved["iter"], 24)
            self.assertEqual(saved["decoder_sha256"], ck["decoder_sha256"])
            provenance = saved["action_bias_provenance"]
            self.assertEqual(provenance["source_checkpoint_sha256"],
                             hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(provenance["action_indices"], [18, 21, 24, 27])
            self.assertEqual(provenance["delta"], -1.0)

    def test_bias_checkpoint_rejects_decoder_mismatch_and_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, decoder, other = root / "source.pt", root / "decoder.npz", root / "other.npz"
            output = root / "biased.pt"
            np.savez(decoder, x=np.array([1]))
            np.savez(other, x=np.array([2]))
            torch.save({"actor": {"head.bias": torch.zeros(29)},
                        "decoder_sha256": hashlib.sha256(decoder.read_bytes()).hexdigest()}, source)
            with self.assertRaisesRegex(ValueError, "decoder hash"):
                offset_aux_bias_checkpoint(source, root / "mismatch.pt",
                                          decoder_path=other,
                                          action_indices=(18, 21, 24, 27), delta=-1.0)
            offset_aux_bias_checkpoint(source, output, decoder_path=decoder,
                                      action_indices=(18, 21, 24, 27), delta=-1.0)
            with self.assertRaises(FileExistsError):
                offset_aux_bias_checkpoint(source, output, decoder_path=decoder,
                                          action_indices=(18, 21, 24, 27), delta=-1.0)


if __name__ == "__main__":
    unittest.main()
