import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from stage3_rl.env import Go2WEnv, is_bad_contact_pair


class TerminalContactTests(unittest.TestCase):
    def test_bad_contact_predicate_preserves_the_body_id_xor_rule(self):
        bad_bodies = {2, 3}
        self.assertTrue(is_bad_contact_pair(2, 9, bad_bodies))
        self.assertTrue(is_bad_contact_pair(9, 3, bad_bodies))
        self.assertFalse(is_bad_contact_pair(2, 3, bad_bodies))
        self.assertFalse(is_bad_contact_pair(8, 9, bad_bodies))

    def test_terminal_contact_diagnostic_contains_names_distance_and_position(self):
        xml = """<mujoco><worldbody>
          <geom name="ground" type="plane" size="2 2 0.1"/>
          <body name="trunk" pos="0 0 1">
            <geom name="trunk_geom" type="sphere" size="0.2"/>
          </body>
        </worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        trunk_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                       "trunk_geom")
        trunk_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        env = object.__new__(Go2WEnv)
        env.model = model
        env.data = SimpleNamespace(ncon=1, contact=[SimpleNamespace(
            geom1=ground,
            geom2=trunk_geom,
            dist=-0.025,
            pos=np.array([1.0, 2.0, 3.0]),
        )])
        env.bad_bodies = {trunk_body}
        env.foot_bodies = []

        _, bad = env._contacts()
        details = env._terminal_contact_details()

        self.assertEqual(bad, 1.0)
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["geom1"], "ground")
        self.assertEqual(details[0]["geom2"], "trunk_geom")
        self.assertEqual(details[0]["body1"], "world")
        self.assertEqual(details[0]["body2"], "trunk")
        self.assertAlmostEqual(details[0]["distance"], -0.025)
        self.assertEqual(details[0]["position"], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
