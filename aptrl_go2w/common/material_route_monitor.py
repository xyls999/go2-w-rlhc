"""Fail-closed route monitor for the frozen v4 material-course geometry.

Call once per physics policy step with actual wheel coordinates and contacts.
This checks traversal evidence, not whether a policy was trained fairly.
"""
import numpy as np
import mujoco


def frame_overlap_top(model, data, x_min, x_max):
    """Return the upper z extent of collidable robot geoms crossing the bar.

    Use oriented box/cylinder/sphere/capsule support functions rather than
    bounding spheres, which can falsely reject a feasible crouch. Unknown
    collision geometry fails closed instead of being silently ignored.
    """
    top = None
    for i in range(model.ngeom):
        if int(model.geom_bodyid[i]) == 0 or not (model.geom_contype[i] or model.geom_conaffinity[i]):
            continue
        kind = int(model.geom_type[i])
        size = np.asarray(model.geom_size[i], dtype=float)
        R = np.asarray(data.geom_xmat[i], dtype=float).reshape(3, 3)
        def extent(axis):
            row = R[axis]
            if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                return float(np.abs(row) @ size[:3])
            if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                return float(size[0])
            if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                c = min(1.0, abs(float(row[2])))
                return float(size[1] * c + size[0] * np.sqrt(max(0.0, 1.0 - c*c)))
            if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                return float(size[1] * abs(row[2]) + size[0])
            if kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                return float(np.linalg.norm(row * size[:3]))
            raise ValueError(f'unsupported robot collision geometry type {kind}')
        center = data.geom_xpos[i]
        ex = extent(0)
        if center[0] + ex < x_min or center[0] - ex > x_max:
            continue
        z = float(center[2]) + extent(2)
        top = z if top is None else max(top, z)
    return top


class MaterialRouteMonitor:
    TIRE_X = (24.725, 26.925)
    BRICK_X = (27.5, 29.5)
    UP_X = (31.0, 33.4)
    BRIDGE_X = (35.2, 38.2)
    DOWN_X = (41.05, 42.65)
    FRAME_X = (44.98, 45.02)
    WHEELS = ('FL_foot_0', 'FR_foot_0', 'RL_foot_0', 'RR_foot_0')

    def __init__(self):
        self.started = False
        self.last_x = None
        self.max_x = -float('inf')
        self.violations = set()
        self.tire_seen = set()
        self.tire_contact = False
        self.brick_entered = self.brick_contact = False
        self.up_entered = self.up_contact = False
        self.bridge_seen = set()
        self.down_entered = self.down_contact = False
        self.frame_checked = False
        self.completed = {name: False for name in ('tire','brick','up','bridge','down','frame')}

    def update(self, x, y, wheel_xy, contacts, frame_top_z):
        x, y = float(x), float(y)
        wheel_xy = np.asarray(wheel_xy, dtype=float)
        if wheel_xy.shape != (4, 2) or not np.isfinite(wheel_xy).all() or not np.isfinite([x, y]).all():
            self.violations.add('invalid_state')
            return
        if self.last_x is None:
            self.started = 0.0 <= x <= 1.0 and abs(y - 4.0) < 0.2
            if not self.started:
                self.violations.add('invalid_start')
        elif abs(x - self.last_x) > 0.15:
            self.violations.add('position_jump')
        self.last_x = x
        self.max_x = max(self.max_x, x)
        if not abs(y - 4.0) < 1.9:
            self.violations.add('off_lane')
        normalized = {(str(a), str(b)) for a,b in contacts}
        def touching(wheel, predicate):
            return any((a == wheel and predicate(b)) or (b == wheel and predicate(a))
                       for a,b in normalized)
        for i,(wx,wy) in enumerate(wheel_xy):
            if self.TIRE_X[0] <= wx <= self.TIRE_X[1]:
                if not 3.4 <= wy <= 4.6:
                    self.violations.add('tire_bypass')
                else:
                    self.tire_seen.add(i)
                    if touching(self.WHEELS[i], lambda n: n.startswith('tire_')):
                        self.tire_contact = True
            if self.BRIDGE_X[0] <= wx <= self.BRIDGE_X[1]:
                if not 2.1 <= wy <= 5.9:
                    self.violations.add('bridge_outside')
                if touching(self.WHEELS[i], lambda n: n == 'balance_beam'):
                    self.bridge_seen.add(i)
        if self.tire_contact and len(self.tire_seen) == 4 and np.min(wheel_xy[:,0]) > self.TIRE_X[1]:
            self.completed['tire'] = True
        if self.BRICK_X[0] <= x <= self.BRICK_X[1]:
            self.brick_entered = True
            if not 3.0 <= y <= 5.0:
                self.violations.add('brick_bypass')
        if any(touching(w, lambda n: n == 'brick_pit_base' or n.startswith('brick_bark_')) for w in self.WHEELS):
            self.brick_contact = True
        if self.brick_entered and self.brick_contact and np.min(wheel_xy[:,0]) > self.BRICK_X[1]:
            self.completed['brick'] = True
        for key, limits in (('up', self.UP_X), ('down', self.DOWN_X)):
            if limits[0] <= x <= limits[1]:
                setattr(self, key + '_entered', True)
                if not 3.6 <= y <= 4.4:
                    self.violations.add(key + '_bypass')
            if any(touching(w, lambda n: n.startswith('stair_' + key + '_') or n == key + '_platform') for w in self.WHEELS):
                setattr(self, key + '_contact', True)
            if (getattr(self, key + '_entered') and getattr(self, key + '_contact') and
                    np.min(wheel_xy[:,0]) > limits[1]):
                self.completed[key] = True
        if len(self.bridge_seen) == 4 and np.min(wheel_xy[:,0]) > self.BRIDGE_X[1]:
            self.completed['bridge'] = True
        if frame_top_z is not None:
            frame_top_z = float(frame_top_z)
            if not np.isfinite(frame_top_z) or frame_top_z > 0.25 + 1e-6:
                self.violations.add('frame_overhead')
            if not 44.5 <= x <= 45.5:
                self.violations.add('frame_measurement_outside')
            if not np.all(np.abs(wheel_xy[:,1] - 4.0) <= 0.4):
                self.violations.add('frame_bypass')
            self.frame_checked = True
        if self.frame_checked and np.min(wheel_xy[:,0]) > self.FRAME_X[1]:
            self.completed['frame'] = True

    def success(self):
        return self.started and self.max_x >= 100.0 and not self.violations and all(self.completed.values())

    def report(self):
        return {'started':self.started,'max_x':self.max_x,'completed':dict(self.completed),
                'violations':sorted(self.violations),'success':self.success()}
