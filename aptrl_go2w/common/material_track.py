"""Load the explicitly marked, versioned material preview as training geometry."""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


class MaterialTrack:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        if root.tag != 'mujoco' or root.get('model') != 'material_track_preview':
            raise ValueError('not a supported material-track scene')
        self.geoms = []
        self.boxes = []
        world = root.find('worldbody')
        for g in world.findall('geom'):
            name = g.get('name', '')
            if name == 'track_floor' or name.startswith('lane_mark_'):
                continue
            if g.get('type') != 'box':
                raise ValueError('unsupported material-track geometry: ' + name)
            pos = np.fromstring(g.get('pos', ''), sep=' ')
            size = np.fromstring(g.get('size', ''), sep=' ')
            euler = np.fromstring(g.get('euler', '0 0 0'), sep=' ')
            if len(pos) != 3 or len(size) != 3 or len(euler) != 3 or np.any(size <= 0):
                raise ValueError('invalid box geometry: ' + name)
            quat = np.empty(4)
            mujoco.mju_euler2Quat(quat, euler, 'xyz')
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, quat)
            rot = mat.reshape(3, 3)
            self.boxes.append((name, pos, size, rot))
            self.geoms.append(ET.tostring(g, encoding='unicode'))
        bounds = []
        overhead = []
        for name, pos, size, rot in self.boxes:
            ex, ey = np.abs(rot[0]) @ size, np.abs(rot[1]) @ size
            bounds.append((pos[0]-ex, pos[0]+ex, pos[1]-ey, pos[1]+ey))
            overhead.append(name.startswith('frame_') or name == 'clearance_top' or
                            name.startswith('bridge_barrier_'))
        self._bounds = np.asarray(bounds, dtype=float).reshape(-1, 4)
        self._overhead = np.asarray(overhead, dtype=bool)
        self._x_order = np.argsort(self._bounds[:, 0], kind='stable')
        self._x_sorted = self._bounds[self._x_order, 0]

    def _candidate_indices(self, x, y, include_overhead):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return np.empty(0, dtype=np.int64)
        xmin, xmax = float(np.min(x[finite])), float(np.max(x[finite]))
        ymin, ymax = float(np.min(y[finite])), float(np.max(y[finite]))
        count = int(np.searchsorted(self._x_sorted, xmax + 1e-9, side='right'))
        indices = self._x_order[:count]
        b = self._bounds[indices]
        mask = ((b[:, 1] >= xmin - 1e-9) & (b[:, 2] <= ymax + 1e-9) &
                (b[:, 3] >= ymin - 1e-9))
        if not include_overhead:
            mask &= ~self._overhead[indices]
        return indices[mask]

    def validate_stair_reward_layout(self):
        """Fail closed if the coordinate-specific shaper no longer matches."""
        by_name = {name: (pos, size, rot) for name, pos, size, rot in self.boxes}
        expected = {}
        for prefix, start in (('stair_up', 31.0), ('stair_down', 41.05)):
            for i in range(5):
                h = 0.12 * (i + 1 if prefix == 'stair_up' else 5 - i)
                expected[f'{prefix}_{i}'] = ([start + 0.16 + 0.32*i, 4, h/2], [0.16, 0.4, h/2])
        expected['up_platform'] = ([33.0, 4, 0.3], [0.4, 0.4, 0.3])
        expected['down_platform'] = ([40.65, 4, 0.3], [0.4, 0.4, 0.3])
        for name, (pos, size) in expected.items():
            if (name not in by_name or not np.allclose(by_name[name][0], pos, atol=1e-6) or
                    not np.allclose(by_name[name][1], size, atol=1e-6) or
                    not np.allclose(by_name[name][2], np.eye(3), atol=1e-6)):
                raise ValueError('stair reward layout mismatch: ' + name)

    def height(self, x, y, *, include_overhead=False):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        result = np.zeros(x.shape, dtype=float)
        for i in self._candidate_indices(x, y, include_overhead):
            name, pos, size, rot = self.boxes[i]
            # Intersect the world-vertical line through (x,y) with the oriented box.
            dx, dy = x - pos[0], y - pos[1]
            extent_x = np.abs(rot[0]) @ size
            extent_y = np.abs(rot[1]) @ size
            mask = (np.abs(dx) <= extent_x + 1e-9) & (np.abs(dy) <= extent_y + 1e-9)
            if not np.any(mask):
                continue
            lo = np.full(x.shape, -np.inf)
            hi = np.full(x.shape, np.inf)
            valid = mask.copy()
            for axis in range(3):
                a = dx * rot[0, axis] + dy * rot[1, axis] - pos[2] * rot[2, axis]
                b = rot[2, axis]
                if abs(b) < 1e-12:
                    valid &= np.abs(a) <= size[axis] + 1e-9
                else:
                    z1 = (-size[axis] - a) / b
                    z2 = (size[axis] - a) / b
                    lo = np.maximum(lo, np.minimum(z1, z2))
                    hi = np.minimum(hi, np.maximum(z1, z2))
            valid &= lo <= hi + 1e-9
            result = np.where(valid, np.maximum(result, hi), result)
        return float(result) if result.ndim == 0 else result
