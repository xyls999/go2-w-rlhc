import os
import unittest

import numpy as np

from common.terrain import load_npz_track


EXPECTED_SEQUENCE = [
    "start", "rough_flat", "rough_slope", "gap", "continuous_gap",
    "stairs_up", "stairs_down", "stepping_one_bridge", "stool", "crawl",
    "stairs_cliff", "slope_cliff", "start",
]
SEGMENT_ROWS = 80


def boxes_matching(boxes, predicate):
    return [box for box in boxes if predicate(box)]


def is_crawl_bar(box):
    return box["x1"] - box["x0"] < 0.6 and box["y1"] - box["y0"] > 6.0


def is_stool_component(box):
    return box["x1"] - box["x0"] < 0.8 and box["y1"] - box["y0"] < 6.0


class TrackSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("APTRL_TRACK_NPZ")
        if not path:
            raise RuntimeError("Set APTRL_TRACK_NPZ to the fixed race-track NPZ")
        cls.path = path
        cls.source = np.load(path, allow_pickle=True)
        cls.expected_h = (
            cls.source["height_field_raw"].astype(np.float32)
            * float(cls.source["vertical_scale"])
        )
        (cls.fixed_h, cls.fixed_boxes, cls.fixed_seq,
         cls.fixed_meta) = load_npz_track(path, apply_fixes=True)
        (cls.raw_h, cls.raw_boxes, cls.raw_seq,
         cls.raw_meta) = load_npz_track(path, apply_fixes=False)

    def test_track_sequence_matches_the_declared_thirteen_segments(self):
        self.assertEqual(self.fixed_seq, EXPECTED_SEQUENCE)
        self.assertEqual(self.raw_seq, EXPECTED_SEQUENCE)

    def test_fixed_loader_preserves_all_undocumented_heightmap_segments(self):
        for index, name in enumerate(EXPECTED_SEQUENCE):
            if name not in ("stairs_cliff", "slope_cliff"):
                section = slice(index * SEGMENT_ROWS, (index + 1) * SEGMENT_ROWS)
                np.testing.assert_array_equal(
                    self.fixed_h[section], self.expected_h[section],
                    err_msg=f"unexpected geometry change in {name} segment {index}",
                )

    def test_stairs_cliff_is_capped_and_drops_at_documented_grid_row(self):
        segment = self.fixed_h[10 * SEGMENT_ROWS:11 * SEGMENT_ROWS]
        cliff_row = int(round((85.8 - 10 * 8.0) / 0.1))
        self.assertAlmostEqual(float(segment.max()), 0.80, places=6)
        np.testing.assert_allclose(segment[cliff_row - 1], 0.80, atol=1e-6)
        np.testing.assert_array_equal(segment[cliff_row:], 0.0)

    def test_slope_cliff_is_capped_and_drops_at_documented_grid_row(self):
        segment = self.fixed_h[11 * SEGMENT_ROWS:12 * SEGMENT_ROWS]
        cliff_row = int(round((95.8 - 11 * 8.0) / 0.1))
        self.assertAlmostEqual(float(segment.max()), 0.62, places=6)
        np.testing.assert_allclose(segment[cliff_row - 1], 0.62, atol=1e-6)
        np.testing.assert_array_equal(segment[cliff_row:], 0.0)

    def test_crawl_clearance_is_040m_only_when_fixes_are_enabled(self):
        fixed_bars = boxes_matching(self.fixed_boxes, is_crawl_bar)
        raw_bars = boxes_matching(self.raw_boxes, is_crawl_bar)
        self.assertEqual(len(fixed_bars), 1)
        self.assertEqual(len(raw_bars), 1)
        self.assertAlmostEqual(fixed_bars[0]["z0"], 0.40, places=6)
        self.assertAlmostEqual(raw_bars[0]["z0"], 0.345, places=6)

    def test_stool_components_are_merged_only_when_fixes_are_enabled(self):
        fixed_stools = boxes_matching(self.fixed_boxes, is_stool_component)
        raw_stools = boxes_matching(self.raw_boxes, is_stool_component)
        self.assertEqual(len(fixed_stools), 3)
        self.assertEqual(len(raw_stools), 9)
        for box in fixed_stools:
            self.assertAlmostEqual(box["z0"], 0.0, places=6)
            self.assertGreater(box["z1"], 0.38)
            self.assertLess(box["z1"], 0.40)

    def test_disabling_fixes_preserves_source_heightmap_and_reports_no_fixes(self):
        np.testing.assert_array_equal(self.raw_h, self.expected_h)
        self.assertEqual(self.raw_meta["fixes"], [])
        self.assertEqual(len(self.raw_boxes), 10)

    def test_fix_metadata_lists_only_reported_transformations(self):
        fixes = self.fixed_meta["fixes"]
        self.assertTrue(any("stairs_cliff" in fix for fix in fixes))
        self.assertTrue(any("slope_cliff" in fix for fix in fixes))
        self.assertTrue(any("crawl" in fix for fix in fixes))
        self.assertTrue(any("stool" in fix for fix in fixes))
        self.assertFalse(any(term in fix for fix in fixes for term in (
            "gap:", "continuous_gap:", "stairs_up:", "stairs_down:",
            "stepping_one_bridge:", "feasibility recalibration",
        )))


if __name__ == "__main__":
    unittest.main()
