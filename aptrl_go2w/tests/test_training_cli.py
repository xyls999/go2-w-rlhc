"""PPO stage settings are validated before workers and recorded immutably."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from stage3_rl import ppo_train


class TrainingCLITests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("APTRL_TRACK_NPZ"),
                         "APTRL_TRACK_NPZ required for real worker initialization")
    def test_real_workers_receive_v21_before_first_collection(self):
        # Catches a first PPO batch collected by random worker policies despite --init.
        track = os.environ["APTRL_TRACK_NPZ"]
        checkpoint = Path(__file__).resolve().parents[1] / "stage3_rl/runs_v21/model_latest.pt"
        expected = torch.load(checkpoint, map_location="cpu", weights_only=False)
        case = self
        events = []

        class CheckedLearner(ppo_train.Learner):
            def _broadcast_weights(self, verbose=False):
                super()._broadcast_weights(verbose)
                events.append("all_workers_acknowledged_weights")

            def collect(self):
                case.assertTrue(events, "first collection precedes initial weight sync")
                if "collected" not in events:
                    case.assertEqual(events, ["all_workers_acknowledged_weights"])
                    for name, value in self.actor.state_dict().items():
                        case.assertTrue(torch.equal(value.cpu(), expected["actor"][name]))
                batch = super().collect()
                events.append("collected")
                return batch

        with tempfile.TemporaryDirectory() as directory, patch.object(
                ppo_train, "Learner", CheckedLearner):
            ppo_train.main([
                "--init", str(checkpoint), "--out", str(Path(directory) / "smoke"),
                "--probe-track", track, "--train-track", track,
                "--train-stage", "gap_rough_retain", "--gap-width", "0.2",
                "--workers", "4", "--epw", "1", "--iters", "1",
                "--batch-target", "200", "--epochs", "1", "--episode-len", "900",
            ])
        self.assertEqual(events, ["all_workers_acknowledged_weights", "collected",
                                  "all_workers_acknowledged_weights"])

    def test_gap_rough_retention_has_fixed_two_plus_two_worker_mapping(self):
        # Catches silent ratio drift or an invalid composite reaching worker spawn.
        self.assertEqual([
            ppo_train.worker_stage_for_index("gap_rough_retain", index, 4)
            for index in range(4)
        ], ["single_gap", "rough_slope_retain",
            "single_gap", "rough_slope_retain"])
        for workers in (1, 2, 3, 5):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                ppo_train.worker_stage_for_index("gap_rough_retain", 0, workers)

    def test_composite_validation_and_manifest_record_actual_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track = root / "track.npz"
            init = root / "v21.pt"
            track.write_bytes(b"fixed track fixture")
            init.write_bytes(b"checkpoint fixture")
            parser = ppo_train.build_arg_parser()
            base = ["--out", str(root / "run"), "--probe-track", str(track),
                    "--init", str(init), "--train-stage", "gap_rough_retain",
                    "--train-track", str(track), "--gap-width", "0.2",
                    "--workers", "4", "--epw", "1", "--episode-len", "900"]
            args = parser.parse_args(base)
            ppo_train.validate_training_args(args)
            for replacement in (("--workers", "2"), ("--epw", "2"),
                                ("--gap-width", "0.1"),
                                ("--train-track", str(track) + ".missing")):
                changed = base.copy()
                index = changed.index(replacement[0])
                changed[index+1] = replacement[1]
                with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                    ppo_train.validate_training_args(parser.parse_args(changed))
            out = root / "run"
            out.mkdir()
            config_path = ppo_train.write_training_run_config(args, str(out))
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))
            self.assertEqual(config["worker_stages"], [
                "single_gap", "rough_slope_retain",
                "single_gap", "rough_slope_retain",
            ])
            self.assertEqual(config["allocation"],
                             "2 gap : 2 rough_slope_retain")
            self.assertEqual(config["init_sha256"],
                             hashlib.sha256(init.read_bytes()).hexdigest())
            self.assertEqual(config["train_track_sha256"],
                             hashlib.sha256(track.read_bytes()).hexdigest())

    def test_stage_validation_requires_real_track_and_approved_width(self):
        # Catches accidental training on missing/eased geometry or short full episodes.
        with tempfile.TemporaryDirectory() as directory:
            track = Path(directory) / "track.npz"
            track.write_bytes(b"training fixture")
            parser = ppo_train.build_arg_parser()
            def args(*extra):
                return parser.parse_args(["--out", "new-run", "--probe-track",
                                          str(track), *extra])
            for extras in (("--train-stage", "single_gap"),
                           ("--train-stage", "double_gap"),
                           ("--train-stage", "single_gap", "--train-track",
                            str(track), "--gap-width", "0.15"),
                           ("--train-stage", "full_track", "--train-track",
                            str(track), "--episode-len", "8000"),
                           ("--train-stage", "double_gap", "--train-track",
                            str(track) + ".missing")):
                with self.subTest(extras=extras), self.assertRaises(ValueError):
                    ppo_train.validate_training_args(args(*extras))
            valid = args("--train-stage", "single_gap", "--train-track",
                         str(track), "--gap-width", "0.1", "--episode-len", "900")
            ppo_train.validate_training_args(valid)
            self.assertEqual(ppo_train.worker_env_config(valid), {
                "train_stage": "single_gap", "train_track": str(track),
                "gap_width_m": 0.1, "episode_len": 900,
                "decoder_path": ppo_train.build_arg_parser().get_default("decoder_path"),
                "command_speed": None, "command_ramp": None,
            })

    def test_material_aux_limit_is_explicit_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);track=root/'track.xml';track.write_text('<mujoco/>')
            out=root/'run';out.mkdir()
            args=ppo_train.build_arg_parser().parse_args([
                '--out',str(out),'--probe-track',str(track),'--train-stage','full_track',
                '--train-track',str(track),'--episode-len','9000','--material-aux-limit','3.0'])
            ppo_train.validate_training_args(args)
            self.assertEqual(ppo_train.worker_env_config(args)['material_aux_limit'],3.0)
            config=json.loads(Path(ppo_train.write_training_run_config(args,str(out))).read_text())
            self.assertEqual(config['material_aux_limit'],3.0)
            self.assertIsNone(ppo_train.parse_material_aux_limit('none'))
            with self.assertRaises(Exception):ppo_train.parse_material_aux_limit('9')
            with self.assertRaises(Exception):ppo_train.parse_material_aux_limit('0')

    def test_run_config_records_optimizer_and_stability_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);track=root/'track.npz';track.write_bytes(b'track')
            out=root/'run';out.mkdir()
            args=ppo_train.build_arg_parser().parse_args([
                '--out',str(out),'--probe-track',str(track),'--freeze-norm',
                '--bc-coef','0.05','--lr','0.0001','--critic-warmup','2',
                '--latent-std','0.3','--aux-std','0.2','--gait-std','0.4'])
            config=json.loads(Path(ppo_train.write_training_run_config(args,str(out))).read_text())
            self.assertTrue(config['freeze_norm'])
            self.assertEqual(config['bc_coef'],0.05)
            self.assertEqual(config['learning_rate'],0.0001)
            self.assertEqual(config['critic_warmup'],2)
            self.assertEqual(config['latent_std'],0.3)
            self.assertEqual(config['aux_std'],0.2)
            self.assertEqual(config['gait_std'],0.4)

    def test_new_run_config_records_stage_and_source_hashes(self):
        # Catches an untraceable checkpoint whose training geometry was forgotten.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track = root / "track.npz"
            init = root / "v21.pt"
            track.write_bytes(b"track-bytes")
            init.write_bytes(b"v21-bytes")
            out = root / "new-run"
            args = ppo_train.build_arg_parser().parse_args([
                "--out", str(out), "--probe-track", str(track),
                "--init", str(init), "--train-track", str(track),
                "--train-stage", "single_gap", "--gap-width", "0.1",
                "--eval-protocol", "pose-v1", "--episode-len", "900",
                "--iters", "25",
            ])
            ppo_train.validate_training_args(args)
            out.mkdir()
            path = ppo_train.write_training_run_config(args, str(out))
            config = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(config["train_stage"], "single_gap")
            self.assertEqual(config["gap_width_m"], 0.1)
            self.assertEqual(config["eval_protocol"], "pose-v1")
            self.assertEqual(config["iters"], 25)
            self.assertEqual(config["init_sha256"],
                             hashlib.sha256(init.read_bytes()).hexdigest())
            self.assertEqual(config["train_track_sha256"],
                             hashlib.sha256(track.read_bytes()).hexdigest())
            self.assertTrue(config["source_sha256"]["stage3_rl/env.py"])
            with self.assertRaises(FileExistsError):
                ppo_train.write_training_run_config(args, str(out))
            with self.assertRaises(FileExistsError):
                ppo_train.ensure_empty_output_dir(out)


if __name__ == "__main__":
    unittest.main()
