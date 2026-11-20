import unittest
import xml.etree.ElementTree as ET

from common.material_track_preview import build_scene


class MaterialTrackPreviewTests(unittest.TestCase):
    def test_scene_has_measured_obstacles_and_hollow_tires(self):
        root = ET.fromstring(build_scene())
        geoms = {g.attrib['name']: g for g in root.iter('geom') if 'name' in g.attrib}
        self.assertEqual(geoms['track_floor'].attrib['size'], '52 4 0.1')
        self.assertIn('stair_up_0', geoms)
        self.assertIn('stair_down_0', geoms)
        self.assertIn('brick_pit_base', geoms)
        self.assertIn('balance_beam', geoms)
        self.assertIn('clearance_top', geoms)
        self.assertEqual(len([n for n in geoms if n.startswith('tire_')]), 8 * 32)
        self.assertNotIn('high_wall', geoms)
        self.assertNotIn('stepping_stones', geoms)
        self.assertAlmostEqual(float(geoms['stair_up_0'].attrib['size'].split()[2]) * 2, 0.12)
        self.assertEqual(len({n.rsplit('_', 1)[0] for n in geoms if n.startswith('tire_')}), 8)

    def test_compact_course_more_steps_and_bridge_cannot_be_bypassed(self):
        root = ET.fromstring(build_scene())
        g = {v.attrib['name']: v for v in root.iter('geom') if 'name' in v.attrib}
        for prefix in ('stair_up_', 'stair_down_'):
            self.assertEqual(len([n for n in g if n.startswith(prefix) and n[len(prefix):].isdigit()]), 5)
        self.assertIn('bridge_barrier_left', g)
        self.assertIn('bridge_barrier_right', g)
        bridge_x = float(g['balance_beam'].attrib['pos'].split()[0])
        for name in ('bridge_barrier_left', 'bridge_barrier_right'):
            self.assertAlmostEqual(float(g[name].attrib['pos'].split()[0]), bridge_x - 1.5)
            self.assertGreaterEqual(float(g[name].attrib['size'].split()[1]), 1.9)
        self.assertLess(float(g['clearance_top'].attrib['pos'].split()[0]) - 25.0, 25.0)

    def test_scene_compiles(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string(build_scene())
        self.assertGreater(model.ngeom, 260)

if __name__ == '__main__':
    unittest.main()
