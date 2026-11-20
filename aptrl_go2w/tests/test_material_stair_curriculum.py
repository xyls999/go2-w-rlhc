import json
from pathlib import Path
import tempfile
import unittest
import mujoco
import numpy as np
from common.material_track import MaterialTrack
from stage3_rl.material_stair_curriculum import generate_levels

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/diagnostics/robocon-material-preview-20261113-v4/scene.xml'

class MaterialStairCurriculumTests(unittest.TestCase):
    def test_versions_are_distinct_continuous_and_preserve_tires(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'levels';manifest=generate_levels(SOURCE,out)
            self.assertEqual([x['step_height_m'] for x in manifest['levels']], [.05,.075,.10,.12])
            original=MaterialTrack(SOURCE)
            for entry in manifest['levels']:
                path=out/entry['file'];t=MaterialTrack(path);model=mujoco.MjModel.from_xml_path(str(path))
                data=mujoco.MjData(model);mujoco.mj_forward(model,data)
                for x in (31.16, 32.44, 33.0, 33.8, 34.6, 39.0, 39.7, 40.24):
                    distance=mujoco.mj_ray(model,data,np.array([x,4.,2.]),np.array([0.,0.,-1.]),None,True,-1,None)
                    self.assertGreaterEqual(distance,0)
                    self.assertAlmostEqual(t.height(x,4,include_overhead=True),2-distance,places=5)
                h=entry['step_height_m'];top=5*h
                self.assertAlmostEqual(t.height(31.16,4),h,places=5)
                self.assertAlmostEqual(t.height(32.44,4),top,places=5)
                self.assertAlmostEqual(t.height(33.0,4),top,places=5)
                self.assertLess(abs(t.height(33.41,4)-top),.04)
                self.assertLess(abs(t.height(35.04,4)),.04)
                self.assertLess(abs(t.height(38.61,4)),.04)
                self.assertLess(abs(t.height(40.24,4)-top),.04)
                self.assertAlmostEqual(t.height(25.22,3.725),original.height(25.22,3.725))
            self.assertEqual(json.loads((out/'manifest.json').read_text()),manifest)
            with self.assertRaises(FileExistsError):generate_levels(SOURCE,out)

if __name__=='__main__': unittest.main()
