import hashlib
import tempfile
import unittest
from pathlib import Path

from stage3_rl.evaluate import validate_decoder_pairing


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DecoderProvenanceTests(unittest.TestCase):
    def test_checkpoint_requires_exact_decoder_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "decoder-original.npz"
            candidate = Path(directory) / "decoder-candidate.npz"
            original.write_bytes(b"original")
            candidate.write_bytes(b"candidate")
            ckpt = {"decoder_sha256": sha256(candidate)}
            self.assertEqual(validate_decoder_pairing(ckpt, candidate),
                             sha256(candidate))
            with self.assertRaisesRegex(ValueError, "hashes do not match"):
                validate_decoder_pairing(ckpt, original)

    def test_legacy_checkpoint_is_bound_to_its_original_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "legacy.npz"
            alternate = Path(directory) / "alternate.npz"
            original.write_bytes(b"legacy decoder")
            alternate.write_bytes(b"new decoder")
            self.assertEqual(validate_decoder_pairing({}, original, original),
                             sha256(original))
            with self.assertRaisesRegex(ValueError, "lacks decoder provenance"):
                validate_decoder_pairing({}, alternate, original)


if __name__ == "__main__":
    unittest.main()
