"""Local promotion must be earned and must not mask full-track regression."""
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from stage3_rl import curriculum_gate


class CurriculumGateTests(unittest.TestCase):
    def test_script_is_runnable_from_project_root(self):
        # Catches missing root import setup when invoked as a script on Windows.
        root = Path(__file__).resolve().parents[1]
        process = subprocess.run([sys.executable,
                                  str(root / "stage3_rl/curriculum_gate.py"),
                                  "--help"], cwd=root, capture_output=True,
                                 text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--train-stage", process.stdout)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manifest = Path(self.temp.name) / "manifest.json"
        self.episodes_path = self.manifest.with_name("episodes.jsonl")

    def rows(self, passed=8):
        return [dict(seed=9000+i, success=i < passed,
                     reason="success" if i < passed else "body_contact",
                     max_x=32.1 if i < passed else 30.2,
                     landed_after_lip=i < passed)
                for i in range(10)]

    def gate(self, passed=8, median=27.0):
        self.manifest.write_text(json.dumps({"evaluation": {
            "eval_protocol": "pose-v1", "episodes": 10,
        }}), encoding="utf-8")
        self.episodes_path.write_text("".join(json.dumps({
            "seed": 9000+i, "max_x": median,
        }) + "\n" for i in range(10)), encoding="utf-8")
        return dict(train_stage="double_gap", finish_x=32.0,
                    episodes=self.rows(passed),
                    full_track_median_x=median,
                    full_track_manifest=str(self.manifest))

    def test_requires_eight_real_local_successes_and_full_track_nonregression(self):
        # Catches treating a local gain as promotion despite a worse full track.
        self.assertTrue(curriculum_gate.should_promote(self.gate(), 26.7))
        self.assertFalse(curriculum_gate.should_promote(self.gate(7), 26.7))
        self.assertFalse(curriculum_gate.should_promote(self.gate(median=26.6), 26.7))

    def test_landing_finish_and_failure_reason_are_all_required(self):
        # Catches a base crossing the lip while falling or without wheel contact.
        gate = self.gate()
        gate["episodes"][0]["landed_after_lip"] = False
        self.assertFalse(curriculum_gate.should_promote(gate, 26.7))
        gate = self.gate()
        gate["episodes"][0]["reason"] = "off_lane"
        self.assertFalse(curriculum_gate.should_promote(gate, 26.7))
        gate = self.gate()
        gate["episodes"][0]["max_x"] = 31.9
        self.assertFalse(curriculum_gate.should_promote(gate, 26.7))

    def test_rejects_repeated_seeds_or_missing_full_track_manifest(self):
        # Catches ten repeats of one deterministic scenario presented as 10 trials.
        gate = self.gate()
        gate["episodes"][1]["seed"] = 9000
        self.assertFalse(curriculum_gate.should_promote(gate, 26.7))
        gate = self.gate()
        gate["full_track_manifest"] = ""
        self.assertFalse(curriculum_gate.should_promote(gate, 26.7))
        gate = self.gate()
        gate["full_track_median_x"] = 99.0  # cannot invent the metric
        self.assertFalse(curriculum_gate.should_promote(gate, 26.7))

    def test_original_rough_slope_gate_needs_finish_not_gap_contact(self):
        # Catches erroneously requiring a gap landing for the retention slice.
        row = {"seed": 9000, "success": True, "reason": "success",
               "max_x": 23.5, "landed_after_lip": None}
        self.assertTrue(curriculum_gate._local_passed(
            row, "rough_slope_retain", 23.5))
        row["max_x"] = 23.49
        self.assertFalse(curriculum_gate._local_passed(
            row, "rough_slope_retain", 23.5))
        gate = self.gate(passed=9)
        gate["train_stage"] = "rough_slope_retain"
        gate["finish_x"] = 23.5
        for episode in gate["episodes"]:
            episode["landed_after_lip"] = None
        self.assertTrue(curriculum_gate.should_promote(gate, 26.7))

    def test_flat_gate_does_not_require_gap_contact(self):
        gate = self.gate()
        gate["train_stage"] = "flat"
        gate["finish_x"] = 19.6
        for row in gate["episodes"]:
            row["landed_after_lip"] = False
        self.assertTrue(curriculum_gate.should_promote(gate, 26.7))


class RetentionDecisionTests(unittest.TestCase):
    baseline_success_seeds = {9001, 9003, 9004, 9005, 9008, 9009}

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.track = self.root / "original.npz"
        self.track.write_bytes(b"unchanged official track fixture")
        self.track_hash = hashlib.sha256(self.track.read_bytes()).hexdigest()
        self.v21 = self.root / "v21.pt"
        self.candidate = self.root / "candidate.pt"
        self.v21.write_bytes(b"v21 fixture")
        self.candidate.write_bytes(b"candidate fixture")
        self.baseline_hash = hashlib.sha256(self.v21.read_bytes()).hexdigest()
        self.candidate_hash = hashlib.sha256(self.candidate.read_bytes()).hexdigest()
        self.gap = self.local_gate("single_gap", set(range(9000, 9008)),
                                   self.candidate_hash, 28.3, 0.2)
        self.rough = self.local_gate("rough_slope_retain",
                                     self.baseline_success_seeds,
                                     self.candidate_hash, 23.5)
        self.baseline_rough = self.local_gate("rough_slope_retain",
                                              self.baseline_success_seeds,
                                              self.baseline_hash, 23.5)
        self.full_candidate = self.full_manifest("candidate", self.candidate,
                                                  [20.0 + i for i in range(10)])
        self.full_baseline = self.full_manifest("baseline", self.v21,
                                                 [19.9 + i for i in range(10)])

    def local_gate(self, stage, successes, checkpoint_hash, finish, width=None):
        rows = [dict(seed=seed, success=seed in successes,
                     reason="success" if seed in successes else "body_contact",
                     max_x=finish + 0.01 if seed in successes else finish - 0.01,
                     landed_after_lip=True if stage == "single_gap" else None,
                     passed=seed in successes,
                     training_heightmap_sha256="same-original-heightmap"
                     if stage == "rough_slope_retain" else "training-gap-heightmap")
                for seed in range(9000, 9010)]
        return dict(train_stage=stage, finish_x=finish, gap_width_m=width,
                    n_pass=len(successes), checkpoint_sha256=checkpoint_hash,
                    track_sha256=self.track_hash, episodes=rows)

    def full_manifest(self, label, checkpoint, xs):
        folder = self.root / label
        folder.mkdir()
        rows = [dict(seed=9000+i, max_x=x,
                     requested_initial_pose=[0.45+i*0.001, 4.0, 0.0])
                for i, x in enumerate(xs)]
        (folder / "episodes.jsonl").write_text(
            "".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
        path = folder / "manifest.json"
        path.write_text(json.dumps({
            "checkpoint": {"path": str(checkpoint),
                           "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
            "track": {"path": str(self.track), "sha256": self.track_hash},
            "evaluation": {"eval_protocol": "pose-v1", "seed_base": 9000,
                           "episodes": 10, "seed_policy": "seed_base + episode_index",
                           "deterministic": True, "episode_len": 16000,
                           "success_x_m": 100.0, "lane_center_y_m": 4.0,
                           "lane_half_width_m": 1.9,
                           "normal_speed_mps": 1.5, "slow_speed_mps": 0.8,
                           "slow_segments": [5, 6, 7, 9, 10, 11]}
        }), encoding="utf-8")
        return path

    def decide(self):
        return curriculum_gate.retention_decision(
            self.gap, self.rough, self.baseline_rough,
            str(self.full_candidate), str(self.full_baseline))

    def test_passes_only_when_six_baseline_successes_are_retained(self):
        result = self.decide()
        self.assertTrue(result["promote"], result)
        self.assertEqual(result["retained_baseline_seeds"],
                         sorted(self.baseline_success_seeds))
        self.assertEqual(len(result["paired_deltas_m"]), 10)
        self.assertEqual(result["candidate_checkpoint_sha256"],
                         self.candidate_hash)

    def test_fails_for_local_threshold_and_swapped_success_seed(self):
        self.gap["episodes"][7]["success"] = False
        self.gap["episodes"][7]["reason"] = "body_contact"
        self.gap["episodes"][7]["passed"] = False
        self.gap["n_pass"] = 7
        self.assertFalse(self.decide()["conditions"]["gap_8_of_10"])
        self.gap = self.local_gate("single_gap", set(range(9000, 9008)),
                                   self.candidate_hash, 28.3, 0.2)
        self.rough["episodes"][1].update(success=False, reason="body_contact",
                                          passed=False, max_x=23.49)
        self.rough["episodes"][0].update(success=True, reason="success",
                                          passed=True, max_x=23.51)
        result = self.decide()
        self.assertEqual(self.rough["n_pass"], 6)
        self.assertFalse(result["conditions"]["retained_baseline_seeds"])
        self.assertFalse(result["promote"])
        self.rough["episodes"][0].update(success=False, reason="body_contact",
                                          passed=False, max_x=23.49)
        self.rough["n_pass"] = 5
        result = self.decide()
        self.assertFalse(result["conditions"]["rough_6_of_10"])

    def test_fails_for_one_paired_regression_or_lower_median(self):
        rows_path = self.full_candidate.with_name("episodes.jsonl")
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        rows[0]["max_x"] -= 5.11  # starts +0.10 ahead; becomes -5.01
        rows_path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        result = self.decide()
        self.assertFalse(result["conditions"]["paired_regression"])
        self.assertFalse(result["promote"])
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        for i, row in enumerate(rows):
            row["max_x"] = 19.8 + i  # every seed -0.10: median declines
        rows_path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        result = self.decide()
        self.assertTrue(result["conditions"]["paired_regression"])
        self.assertFalse(result["conditions"]["median_nonregression"])

    def test_rejects_mismatched_hash_protocol_and_duplicate_seed(self):
        self.rough["checkpoint_sha256"] = "bad-hash"
        self.assertFalse(self.decide()["conditions"]["artifact_integrity"])
        self.rough["checkpoint_sha256"] = self.candidate_hash
        manifest = json.loads(self.full_candidate.read_text())
        manifest["evaluation"]["eval_protocol"] = "legacy"
        self.full_candidate.write_text(json.dumps(manifest))
        self.assertFalse(self.decide()["conditions"]["artifact_integrity"])
        manifest["evaluation"]["eval_protocol"] = "pose-v1"
        self.full_candidate.write_text(json.dumps(manifest))
        self.gap["episodes"][1]["seed"] = 9000
        self.assertFalse(self.decide()["conditions"]["artifact_integrity"])


if __name__ == "__main__":
    unittest.main()
