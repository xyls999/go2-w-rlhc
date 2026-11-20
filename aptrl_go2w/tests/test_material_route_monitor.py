import unittest
import numpy as np
from common.material_route_monitor import MaterialRouteMonitor, frame_overlap_top


def wheels(x, y=4.0):
    return np.array([[x+.2,y+.15],[x+.2,y-.15],[x-.2,y+.15],[x-.2,y-.15]])

class MaterialRouteMonitorTests(unittest.TestCase):
    def test_frame_top_uses_actual_collidable_geometry_extents(self):
        import mujoco
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body pos="45 4 0.20"><freejoint/><geom name="robot" type="box" size="0.1 0.05 0.04"/></body></worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        self.assertAlmostEqual(frame_overlap_top(model,data,44.98,45.02),0.24)
        self.assertIsNone(frame_overlap_top(model,data,40,41))

    def test_direct_finish_without_course_is_rejected(self):
        m=MaterialRouteMonitor()
        m.update(.5,4.,wheels(.5),[],None)
        m.update(100.,4.,wheels(100.),[],None)
        self.assertFalse(m.success())

    def test_valid_sequence_requires_real_contacts_and_bridge_wheels(self):
        m=MaterialRouteMonitor()
        for x in np.linspace(.5, 100., 1991):
            contacts=[]
            if 24.725 <= x <= 26.925: contacts.append(('FR_foot_0','tire_0_0_0'))
            if 27.5 <= x <= 29.5: contacts.append(('FR_foot_0','brick_pit_base'))
            if 31 <= x <= 33.4: contacts.append(('FR_foot_0','stair_up_0'))
            if 35.2 <= x <= 38.2: contacts.extend([(f'{leg}_foot_0','balance_beam') for leg in ('FL','FR','RL','RR')])
            if 41.05 <= x <= 42.65: contacts.append(('FR_foot_0','stair_down_0'))
            m.update(float(x),4.,wheels(x),contacts,.24 if 44.5<=x<=45.5 else None)
        self.assertTrue(m.success(),m.report())

    def test_tire_corridor_without_any_tire_contact_does_not_complete_tires(self):
        m=MaterialRouteMonitor()
        for x in np.linspace(.5,28.,551):
            m.update(float(x),4.,wheels(x),[],None)
        self.assertFalse(m.report()['completed']['tire'])

    def test_bridge_side_step_invalidates_even_if_later_returned(self):
        m=MaterialRouteMonitor()
        m.update(.5,4.,wheels(.5),[],None)
        m.update(36.,4.,wheels(36.,4.35),[],None)
        m.update(100.,4.,wheels(100.),[],None)
        self.assertFalse(m.success())
        self.assertIn('bridge_outside',m.report()['violations'])

    def test_frame_cannot_be_jumped_over(self):
        m=MaterialRouteMonitor()
        m.update(.5,4.,wheels(.5),[],None)
        m.update(45.,4.,wheels(45.),[],.4)
        self.assertIn('frame_overhead',m.report()['violations'])

    def test_lane_is_strict_and_failure_is_sticky(self):
        m=MaterialRouteMonitor()
        m.update(.5,4.,wheels(.5),[],None)
        m.update(10.,5.9,wheels(10.,5.9),[],None)
        m.update(11.,4.,wheels(11.),[],None)
        self.assertIn('off_lane',m.report()['violations'])

if __name__=='__main__':unittest.main()
