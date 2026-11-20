import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from stage3_rl.material_tire_curriculum import generate_levels

ROOT=Path(__file__).resolve().parents[1]
SCENE=ROOT/'results/diagnostics/robocon-material-preview-20261113-v4/scene.xml'

class MaterialTireCurriculumTests(unittest.TestCase):
    def test_levels_scale_only_tire_height_and_record_source(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'levels'
            manifest=generate_levels(SCENE,out)
            self.assertEqual(len(manifest['levels']),3)
            original=ET.parse(SCENE).getroot()
            source={e.get('name'):e for e in original.iter('geom') if e.get('name')}
            for entry in manifest['levels']:
                root=ET.parse(out/entry['file']).getroot()
                geoms={e.get('name'):e for e in root.iter('geom') if e.get('name')}
                self.assertEqual(sum(n.startswith('tire_') for n in geoms),256)
                size=[float(v) for v in geoms['tire_0_0_0'].get('size').split()]
                pos=[float(v) for v in geoms['tire_0_0_0'].get('pos').split()]
                self.assertAlmostEqual(2*size[2],entry['height_m'])
                self.assertAlmostEqual(pos[2],size[2])
                self.assertEqual(geoms['balance_beam'].attrib,source['balance_beam'].attrib)
            self.assertEqual(json.loads((out/'manifest.json').read_text()),manifest)
            with self.assertRaises(FileExistsError):generate_levels(SCENE,out)

if __name__=='__main__':unittest.main()
