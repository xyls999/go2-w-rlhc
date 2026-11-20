import hashlib
import json
import os
import tempfile
import unittest

import numpy as np

from common.evaluation import (
    build_run_manifest,
    sha256_file,
    summarize_track,
    write_run_manifest,
)
from stage3_rl.evaluate import build_arg_parser as serial_arg_parser
from stage3_rl.evaluate_par import (build_arg_parser as parallel_arg_parser,
                                    _episode_summary)


class EvaluationManifestTests(unittest.TestCase):
    def test_sha256_file_matches_standard_digest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "payload.bin")
            with open(path, "wb") as stream:
                stream.write(b"track-checkpoint-payload")
            expected = hashlib.sha256(b"track-checkpoint-payload").hexdigest()
            self.assertEqual(sha256_file(path), expected)

    def test_manifest_records_hashes_runtime_config_fixes_and_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = os.path.join(temp_dir, "policy.pt")
            track = os.path.join(temp_dir, "track.npz")
            source = os.path.join(temp_dir, "env.py")
            for path, payload in ((checkpoint, b"checkpoint"),
                                  (track, b"track"), (source, b"source")):
                with open(path, "wb") as stream:
                    stream.write(payload)
            config = {"episodes": 3, "workers": 1, "deterministic": True}
            fixes = ["stairs_cliff peak 1.65 -> 0.80"]
            track_summary = {"sequence": ["start"], "segments": []}

            manifest = build_run_manifest(
                checkpoint=checkpoint,
                track=track,
                config=config,
                fixes=fixes,
                source_files=[source],
                track_summary=track_summary,
            )

            self.assertEqual(manifest["checkpoint"]["sha256"], sha256_file(checkpoint))
            self.assertEqual(manifest["track"]["sha256"], sha256_file(track))
            self.assertEqual(manifest["evaluation"], config)
            self.assertEqual(manifest["fixes"], fixes)
            self.assertEqual(manifest["track_summary"], track_summary)
            self.assertEqual(manifest["source_files"][0]["sha256"], sha256_file(source))
            self.assertEqual(manifest["source_files"][0]["path"], os.path.realpath(source))
            for package in ("python", "numpy", "torch", "mujoco"):
                self.assertTrue(manifest["runtime"][package])

    def test_summarize_track_reports_per_segment_ranges_and_boxes(self):
        heightmap = np.array([[0.0, 0.2], [0.1, -0.1],
                              [0.3, 0.4], [0.2, 0.5]], dtype=np.float32)
        boxes = [dict(x0=1.0, x1=2.0, y0=3.0, y1=4.0, z0=0.0, z1=0.5)]
        summary = summarize_track(heightmap, boxes, ["start", "gap"])
        self.assertEqual([s["name"] for s in summary["segments"]], ["start", "gap"])
        self.assertAlmostEqual(summary["segments"][0]["min_m"], -0.1)
        self.assertAlmostEqual(summary["segments"][1]["max_m"], 0.5)
        self.assertEqual(summary["obstacle_boxes"], boxes)

    def test_both_evaluator_clis_require_explicit_track_and_output(self):
        serial = serial_arg_parser().parse_args([
            "--track", "track.npz", "--out", "serial-run", "--episodes", "1",
        ])
        parallel = parallel_arg_parser().parse_args([
            "--track", "track.npz", "--out", "parallel-run", "--episodes", "1",
            "--workers", "1",
        ])
        self.assertEqual(serial.track, "track.npz")
        self.assertEqual(serial.out, "serial-run")
        self.assertEqual(parallel.track, "track.npz")
        self.assertEqual(parallel.out, "parallel-run")
        with self.assertRaises(SystemExit):
            serial_arg_parser().parse_args(["--out", "run"])
        with self.assertRaises(SystemExit):
            parallel_arg_parser().parse_args(["--track", "track.npz"])

    def test_parallel_summary_keeps_legacy_track_fixes_field_and_adds_details(self):
        result = {
            "success": False,
            "reason": "body_contact",
            "furthest_segment": "gap",
            "max_x": 25.0,
            "seed": 9000,
        }
        fixes = ["stairs_cliff peak 1.65 -> 0.80"]
        summary = _episode_summary([result], "checkpoint.pt", 12, fixes)
        self.assertIsInstance(summary["track_fixes"], str)
        self.assertEqual(summary["fixes"], fixes)
        self.assertEqual(summary["fails"], {"body_contact": 1})
        self.assertEqual(summary["best_success_seed"], None)

    def test_manifest_writer_refuses_to_overwrite_nonempty_run_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = os.path.join(temp_dir, "policy.pt")
            track = os.path.join(temp_dir, "track.npz")
            for path in (checkpoint, track):
                with open(path, "wb") as stream:
                    stream.write(b"x")
            manifest = build_run_manifest(
                checkpoint=checkpoint,
                track=track,
                config={"episodes": 1},
                fixes=[],
                source_files=[],
                track_summary={},
            )
            out_dir = os.path.join(temp_dir, "run")
            manifest_path = write_run_manifest(out_dir, manifest)
            with open(manifest_path, encoding="utf-8") as stream:
                saved = json.load(stream)
            self.assertEqual(saved, manifest)
            with self.assertRaises(FileExistsError):
                write_run_manifest(out_dir, manifest)


if __name__ == "__main__":
    unittest.main()
