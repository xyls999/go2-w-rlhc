"""Terrain generation for APT-RL Go2-W training and evaluation.

Two APIs:
1. Curriculum segment generators (paper S3): 12 obstacle types x 10 difficulty
   levels (+ 2 flat pre-curriculum levels). Each returns a heightmap (0.1 m
   grid) plus a list of box obstacles (stool / crawl bars).
2. Fixed npz race track loader (go2w_race_track_seed50001.npz) with documented
   design-logic fixes:
   - stairs_cliff / slope_cliff: the npz generator never produced the "cliff"
     (symmetric pyramid instead). We add a true vertical cliff drop and cap
     drop height to 0.8 m / 0.62 m (1.65 m falls exceed quadruped limits).
   - crawl bar clearance raised 0.345 -> 0.40 m (Go2-W trunk 0.114 m tall +
     crouched legs leave <5 cm margin at 0.345 m).
   All other segments (gap, continuous_gap, stairs, bridge, stool, rough) are
   kept exactly as authored in the npz.

Heightmap convention: H[i, j] = height at x = i*0.1, y = j*0.1 (meters).
"""
import os
import tempfile
import numpy as np

HS = 0.1           # horizontal scale (m)
LANE_Y0, LANE_Y1 = 2.2, 5.8   # navigable lane half-width 1.8 m around y=4
TRACK_Y0, TRACK_Y1 = 0.0, 8.0
TRACK_W = int((TRACK_Y1 - TRACK_Y0) / HS)      # 80 cols

TERRAIN_TYPES = [
    "rough_flat", "rough_slope", "gap", "continuous_gap",
    "stairs_up", "stairs_down", "stepping_one_bridge", "stool",
    "crawl", "stairs_cliff", "slope_cliff",
]

# ---------------------------------------------------------------- helpers
def _grid(length_m, width_m=TRACK_W):
    n = int(round(length_m / HS))
    return np.zeros((n, width_m), dtype=np.float32)


def _smooth(H, k=1):
    """Simple box smoothing along both axes (k passes of 3-tap)."""
    for _ in range(k):
        H = 0.25 * np.roll(H, 1, 0) + 0.5 * H + 0.25 * np.roll(H, -1, 0)
        H = 0.25 * np.roll(H, 1, 1) + 0.5 * H + 0.25 * np.roll(H, -1, 1)
        H[0] = H[1]; H[-1] = H[-2]
    return H


def _rough(H, hmin, hmax, rng, lane_only=True):
    """Rough noise like Isaac Gym's rough terrain: per-cell uniform heights,
    2x downsampled (0.2 m blocks), lane-constrained."""
    n, w = H.shape
    lo = np.zeros((n, w), dtype=np.float32)
    blk = 2  # 0.2 m blocks
    vals = rng.uniform(hmin, hmax, size=(n // blk + 1, w // blk + 1)).astype(np.float32)
    for i in range(n):
        for j in range(w):
            lo[i, j] = vals[i // blk, j // blk]
    lo = _smooth(lo, 1)
    if lane_only:
        mask = np.zeros((n, w), dtype=np.float32)
        mask[:, int(LANE_Y0 / HS):int(LANE_Y1 / HS)] = 1.0
        lo *= mask
    return H + lo


# ------------------------------------------------------- curriculum segments
def seg_flat(length=8.0, rng=None, level=0):
    return _grid(length), []


def seg_rough_flat(length=8.0, rng=None, level=1):
    lv = (level - 1) / 9.0
    H = _grid(length)
    hmin, hmax = -0.02 - 0.04 * lv, 0.02 + 0.04 * lv
    return _rough(H, hmin, hmax, rng), []


def seg_rough_slope(length=8.0, rng=None, level=1):
    lv = (level - 1) / 9.0
    H = _grid(length)
    slope = (0.02 + 0.08 * lv) * (1.0 if rng.uniform() < 0.5 else -1.0)
    xs = np.arange(H.shape[0]) * HS
    H += (slope * xs)[:, None].astype(np.float32)
    hmin, hmax = -0.02 - 0.04 * lv, 0.02 + 0.04 * lv
    return _rough(H, hmin, hmax, rng), []


def seg_gap(length=8.0, rng=None, level=1):
    """Discrete chasm gaps (fall = fail)."""
    lv = (level - 1) / 9.0
    H = _grid(length)
    n_gaps = 1 if lv < 0.5 else 2
    w = 0.10 + 0.55 * lv
    for _ in range(n_gaps):
        c = rng.uniform(2.5, length - 2.5)
        i0, i1 = int((c - w / 2) / HS), int((c + w / 2) / HS)
        H[i0:i1, :] = -2.0
    return H, []


def seg_continuous_gap(length=8.0, rng=None, level=1):
    """Washboard grooves every 0.7 m (npz seed uses 0.3 m x 0.3 m)."""
    lv = (level - 1) / 9.0
    H = _grid(length)
    w = 0.08 + 0.27 * lv
    d = 0.06 + 0.24 * lv
    for c in np.arange(1.2, length - 1.0, 0.7):
        i0, i1 = int((c - w / 2) / HS), int((c + w / 2) / HS)
        H[i0:i1, :] = -d
    return H, []


def _stairs_valley(length, step_h, tread, rng, direction):
    """Valley: descend 9 steps (tread 0.3 m), plateau, ascend mirrored.
    stairs_down uses the inverted hill profile."""
    H = _grid(length)
    n = H.shape[0]
    n_steps = 9
    rows_per = max(1, int(round(tread / HS)))
    descent_rows = n_steps * rows_per
    mid = n // 2
    for i in range(n):
        if i < descent_rows:
            k = (i // rows_per) + 1
        elif i < n - descent_rows:
            k = n_steps
        else:
            k = ((n - 1 - i) // rows_per) + 1
        H[i, :] = -step_h * k
    return H, []


def seg_stairs_up(length=8.0, rng=None, level=1):
    """Valley: 0.04-0.14 m steps down then up (tread 0.3 m)."""
    lv = (level - 1) / 9.0
    return _stairs_valley(length, 0.02 + 0.12 * lv, 0.3, rng, "valley")


def seg_stairs_down(length=8.0, rng=None, level=1):
    """Plateau: steps up to a mid plateau then down (mirrored valley)."""
    lv = (level - 1) / 9.0
    H, _ = _stairs_valley(length, 0.02 + 0.12 * lv, 0.3, rng, "valley")
    return -H, []   # inverted: hill instead of valley


def seg_stepping_one_bridge(length=8.0, rng=None, level=1):
    """Single elevated bridge over a -1 m abyss."""
    lv = (level - 1) / 9.0
    H = _grid(length)
    H[:, :] = -0.2 if level <= 2 else -1.0
    bw = 2.4 - 0.6 * lv  # bridge width narrows with difficulty
    y0 = 4.0 - bw / 2
    j0, j1 = int(y0 / HS), int((y0 + bw) / HS)
    H[:, j0:j1] = 0.0
    return H, []


def seg_stool(length=8.0, rng=None, level=1):
    """Three box obstacles (npz: 0.62 x 1.76 x 0.39 m, 2 m apart)."""
    lv = (level - 1) / 9.0
    H = _grid(length)
    h = 0.05 + 0.34 * lv
    boxes = []
    for cx in (2.5, 4.5, 6.5):
        boxes.append(dict(x0=cx - 0.31, x1=cx + 0.31,
                          y0=4.0 - 0.88, y1=4.0 + 0.88,
                          z0=0.0, z1=round(h, 3)))
    return H, boxes


def seg_crawl(length=8.0, rng=None, level=1):
    """Overhead bar: crawl/walk under it. Clearance shrinks with level."""
    lv = (level - 1) / 9.0
    clearance = 0.55 - 0.20 * lv  # 0.55 m -> 0.35 m
    bar_z1 = clearance + 1.0
    boxes = [dict(x0=length / 2 - 0.2, x1=length / 2 + 0.2,
                  y0=0.0, y1=8.0, z0=clearance, z1=bar_z1)]
    return _grid(length), boxes


def seg_stairs_cliff(length=8.0, rng=None, level=1):
    """Stairs up to a plateau, then a true vertical cliff down (npz logic fix).
    Training caps cliff height at 0.8 m; eval fixed track uses 0.8 m."""
    lv = (level - 1) / 9.0
    H = _grid(length)
    Hc = 0.15 + 0.65 * lv          # cliff height 0.15..0.80
    n_steps = 8
    tread = 0.3
    step_h = Hc / n_steps
    for k in range(n_steps):
        H[int((0.5 + k * tread) / HS):int((0.5 + (k + 1) * tread) / HS), :] = step_h * (k + 1)
    top0 = int((0.5 + n_steps * tread) / HS)
    cliff_x = length - 2.0
    ci = int(cliff_x / HS)
    H[top0:ci, :] = Hc
    # vertical drop back to 0 at ci (2 cells to avoid aliasing)
    H[ci:, :] = 0.0
    return H, []


def seg_slope_cliff(length=8.0, rng=None, level=1):
    """Slope up to a plateau, then a vertical cliff down (npz logic fix).
    Training cliff height 0.35..0.62 m; eval fixed track uses 0.62 m."""
    lv = (level - 1) / 9.0
    H = _grid(length)
    Hc = 0.15 + 0.47 * lv
    ang = np.deg2rad(12 + 10 * lv)
    run = Hc / np.tan(ang)
    xs = np.arange(H.shape[0]) * HS
    ramp = np.clip(xs / run, 0, 1) * Hc
    H[:, :] = ramp[:, None]
    top_i = int(run / HS)
    cliff_x = length - 2.0
    ci = int(cliff_x / HS)
    H[top_i:ci, :] = Hc
    H[ci:, :] = 0.0
    return H, []


SEG_FNS = {
    "flat": seg_flat,
    "rough_flat": seg_rough_flat,
    "rough_slope": seg_rough_slope,
    "gap": seg_gap,
    "continuous_gap": seg_continuous_gap,
    "stairs_up": seg_stairs_up,
    "stairs_down": seg_stairs_down,
    "stepping_one_bridge": seg_stepping_one_bridge,
    "stool": seg_stool,
    "crawl": seg_crawl,
    "stairs_cliff": seg_stairs_cliff,
    "slope_cliff": seg_slope_cliff,
}


def build_segment(name, level, length=8.0, seed=0):
    rng = np.random.default_rng(seed)
    H, boxes = SEG_FNS[name](length, rng, level)
    return H.astype(np.float32), boxes


# ------------------------------------------------------------- npz track
def load_npz_track(npz_path, apply_fixes=True):
    """Load the fixed race track from npz. Returns (H, boxes, seq, meta).

    Fixes (documented deviations from the authored npz):
    - stairs_cliff: replace the trailing 2 m stair-descent with a 0.8 m
      vertical cliff at x=85.8 (original pyramid has no cliff).
    - slope_cliff: replace the trailing descent with a 0.62 m cliff at x=95.8.
    - crawl bar: bottom face 0.345 -> 0.40 m clearance.
    """
    d = np.load(npz_path, allow_pickle=True)
    H = d["height_field_raw"].astype(np.float32) * float(d["vertical_scale"])
    seq = [str(s) for s in d["sequence"]]
    hs = float(d["horizontal_scale"])
    assert abs(hs - HS) < 1e-9
    fixes = []

    # obstacle mesh vertices -> boxes via face-connected components
    v = d["obstacle_vertices"].astype(np.float64)
    f = d["obstacle_faces"].astype(np.int64)
    boxes = []
    if len(v) and len(f):
        parent = np.arange(len(v))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for tri in f:
            r = [find(t) for t in tri]
            m = min(r)
            for t in r:
                parent[t] = m
        labels = np.array([find(i) for i in range(len(v))])
        comps = []
        for lab in np.unique(labels):
            vv = v[labels == lab]
            if len(vv) < 4:
                continue
            x0, x1 = vv[:, 0].min(), vv[:, 0].max()
            if x1 - x0 < 0.05:      # degenerate plane fragment
                continue
            comps.append(dict(x0=x0, x1=x1,
                              y0=vv[:, 1].min(), y1=vv[:, 1].max(),
                              z0=vv[:, 2].min(), z1=vv[:, 2].max()))
        if apply_fixes:
            # Merge stool top plates and thin side walls into solid boxes.
            comps.sort(key=lambda b: (b["x0"], b["x1"]))
            merged = []
            for b in comps:
                if (merged and
                        abs(b["x0"] - merged[-1]["x0"]) < 0.1 and
                        abs(b["x1"] - merged[-1]["x1"]) < 0.1):
                    m = merged[-1]
                    m["y0"] = min(m["y0"], b["y0"])
                    m["y1"] = max(m["y1"], b["y1"])
                    m["z0"] = min(m["z0"], b["z0"])
                    m["z1"] = max(m["z1"], b["z1"])
                else:
                    merged.append(b)
            boxes = merged
            if len(merged) < len(comps):
                fixes.append("stool mesh components merged into solid boxes")
        else:
            boxes = comps
    if apply_fixes:
        # --- crawl bar clearance: raise bottom face to the report's 0.40 m ---
        for b in boxes:
            if (b["x1"] - b["x0"] < 0.6 and
                    b["y1"] - b["y0"] > 6.0 and b["z0"] < 0.40):
                fixes.append(f"crawl bar z0 {b['z0']:.3f} -> 0.400")
                b["z0"] = 0.40

        # --- stairs_cliff: vertical cliff at end of the pyramid ---
        si = seq.index("stairs_cliff")
        r0, r1 = si * 80, (si + 1) * 80
        seg = H[r0:r1]
        peak = seg.max()
        new_peak = min(peak, 0.80)
        if abs(peak - new_peak) > 1e-6:
            fixes.append(f"stairs_cliff peak {peak:.2f} -> {new_peak:.2f}")
            seg[seg > 1e-6] *= new_peak / peak
        # find the descending slope end and replace with vertical drop
        center = seg[:, int(4.0 / HS)]
        i_top = int(np.argmax(center))
        # The loader's documented global cliff position is x=85.8 m.
        i_cliff = int(round((85.8 - si * 8.0) / HS))
        fixes.append(f"stairs_cliff: plateau to i={i_cliff}, vertical cliff drop "
                     f"{new_peak:.2f} m")
        seg[i_top:i_cliff, :] = new_peak
        seg[i_cliff:, :] = 0.0
        H[r0:r1] = seg

        # --- slope_cliff: same treatment ---
        si = seq.index("slope_cliff")
        r0, r1 = si * 80, (si + 1) * 80
        seg = H[r0:r1]
        peak = seg.max()
        new_peak = min(peak, 0.62)
        if abs(peak - new_peak) > 1e-6:
            fixes.append(f"slope_cliff peak {peak:.2f} -> {new_peak:.2f}")
            seg[seg > 1e-6] *= new_peak / peak
        center = seg[:, int(4.0 / HS)]
        i_top = int(np.argmax(center))
        # The loader's documented global cliff position is x=95.8 m.
        i_cliff = int(round((95.8 - si * 8.0) / HS))
        fixes.append(f"slope_cliff: plateau to i={i_cliff}, vertical cliff drop "
                     f"{new_peak:.2f} m")
        seg[i_top:i_cliff, :] = new_peak
        seg[i_cliff:, :] = 0.0
        H[r0:r1] = seg

    meta = dict(seq=seq, length_m=H.shape[0] * HS, width_m=H.shape[1] * HS,
                fixes=fixes)
    return H, boxes, seq, meta


# -------------------------------------------------------------- heights API
class HeightField:
    """Bilinear height lookup + mesh building for a heightmap with origin."""

    def __init__(self, H, x0=0.0, y0=0.0, hs=HS):
        self.H = np.asarray(H, dtype=np.float64)
        self.x0, self.y0, self.hs = x0, y0, hs
        self.nx, self.ny = self.H.shape
        self.x1 = x0 + (self.nx - 1) * hs
        self.y1 = y0 + (self.ny - 1) * hs

    def height(self, x, y):
        """Bilinear interpolated terrain height (vectorized)."""
        u = np.clip((np.asarray(x) - self.x0) / self.hs, 0, self.nx - 1 - 1e-6)
        v = np.clip((np.asarray(y) - self.y0) / self.hs, 0, self.ny - 1 - 1e-6)
        i = u.astype(np.int64); j = v.astype(np.int64)
        fu = u - i; fv = v - j
        h = (self.H[i, j] * (1 - fu) * (1 - fv) + self.H[i + 1, j] * fu * (1 - fv)
             + self.H[i, j + 1] * (1 - fu) * fv + self.H[i + 1, j + 1] * fu * fv)
        return h

    def mesh(self, z_base=None):
        """Triangle mesh (verts, faces), vectorized. Includes side skirts down
        to z_base so the mesh is watertight at outer boundaries."""
        nx, ny = self.H.shape
        H = self.H
        xs = self.x0 + np.arange(nx) * self.hs
        ys = self.y0 + np.arange(ny) * self.hs
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        top = np.stack([X, Y, H], axis=-1).reshape(-1, 3)          # (nx*ny, 3)
        vid = np.arange(nx * ny).reshape(nx, ny)
        a = vid[:-1, :-1].reshape(-1); b = vid[1:, :-1].reshape(-1)
        c = vid[1:, 1:].reshape(-1); d = vid[:-1, 1:].reshape(-1)
        faces_top = np.stack([a, b, c, a, c, d], axis=1)
        zb = z_base if z_base is not None else (H.min() - 1.5)
        rings = []
        rings.append(np.stack([xs, np.full(nx, ys[0]), H[:, 0]], axis=1))
        rings.append(np.stack([xs, np.full(nx, ys[-1]), H[:, -1]], axis=1))
        rings.append(np.stack([np.full(ny, xs[0]), ys, H[0, :]], axis=1))
        rings.append(np.stack([np.full(ny, xs[-1]), ys, H[-1, :]], axis=1))
        verts = [top]
        faces = [faces_top]
        base0 = nx * ny
        for ring in rings:
            rb = ring.copy(); rb[:, 2] = zb
            m = len(ring)
            verts.append(ring); verts.append(rb)
            va = np.arange(base0, base0 + m)
            vb = np.arange(base0 + m, base0 + 2 * m)
            f = np.stack([va[:-1], va[1:], vb[1:],
                          va[:-1], vb[1:], vb[:-1]], axis=1)
            faces.append(f)
            base0 += 2 * m
        verts_np = np.concatenate(verts).astype(np.float32)
        faces_np = np.concatenate(faces).astype(np.int32)
        return verts_np, faces_np


_HF_DIR = os.path.join(tempfile.gettempdir(), "aptrl_hfields")
_HF_CACHE = {}


def _hfield_file(hdata_norm, nrow, ncol):
    """Write normalized hfield heights to a cached binary file (float32)."""
    import hashlib
    key = hashlib.md5(hdata_norm.tobytes()).hexdigest()[:16]
    if key in _HF_CACHE:
        return _HF_CACHE[key]
    os.makedirs(_HF_DIR, exist_ok=True)
    path = os.path.join(_HF_DIR, key + ".raw")
    if not os.path.exists(path):
        # write-then-rename so concurrent workers never read a torn file
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            # MuJoCo hfield file format: int32 (nrow, ncol) header + float32
            f.write(np.array([nrow, ncol], dtype=np.int32).tobytes())
            f.write(hdata_norm.astype(np.float32).tobytes())
        try:
            os.replace(tmp, path)
        except OSError:
            os.remove(tmp)  # another worker won the race; file is valid
    _HF_CACHE[key] = path
    return path


def build_env_terrain_xml(H, boxes, x0=0.0, y0=0.0, name="terrain", color=None,
                          exact_height=False):
    """Terrain as a MuJoCo HFIELD (exact non-convex height collision) + boxes.

    IMPORTANT: mesh terrain is unusable for collision -- MuJoCo collides the
    CONVEX HULL of a mesh, which spreads a phantom floor over every valley
    and chasm. The hfield primitive collides the true height grid.
    """
    nx, ny = H.shape
    if exact_height:
        # MuJoCo normalizes loaded hfield data to [0, 1] a second time.
        # Keep the actual MuJoCo collision surface aligned with H everywhere.
        z_base = float(H.min())
        z_span = float(H.max()) - z_base
        if z_span < 1e-6:
            z_span = 1.0
    else:
        # Preserve the legacy collision surface of the official fixed track.
        z_base = float(H.min()) - 0.5
        z_span = float(H.max()) - z_base
        if z_span < 1e-6:
            z_span = 1.0
    half_x = (nx - 1) * HS / 2
    half_y = (ny - 1) * HS / 2
    # MuJoCo maps hfield nrow -> local Y, ncol -> local X (verified
    # empirically), so the grid is transposed relative to our (x, y) layout
    hdata = ((H.T - z_base) / z_span).astype(np.float32).reshape(-1)
    hfile = _hfield_file(hdata, ny, nx)
    asset = (
        '<hfield name="%s" nrow="%d" ncol="%d" '
        'size="%.4f %.4f %.4f 0.05" '
        'file="%s"/>\n' % (name, ny, nx, half_x, half_y, z_span, hfile)
    )
    rgba = color or "0.55 0.5 0.45 1"
    cx = x0 + half_x
    cy = y0 + half_y
    geoms = ('<geom name="%s_geom" type="hfield" hfield="%s" '
             'pos="%.4f %.4f %.4f" '
             'friction="1.0 0.005 0.0001" rgba="%s" condim="3"/>\n'
             % (name, name, cx, cy, z_base, rgba))
    for k, b in enumerate(boxes):
        cx, cy = (b["x0"] + b["x1"]) / 2 + x0, (b["y0"] + b["y1"]) / 2 + y0
        sx, sy, sz = (b["x1"] - b["x0"]) / 2, (b["y1"] - b["y0"]) / 2, (b["z1"] - b["z0"]) / 2
        cz = (b["z0"] + b["z1"]) / 2
        geoms += ('<geom name="%s_box%d" type="box" pos="%.3f %.3f %.3f" '
                  'size="%.3f %.3f %.3f" friction="1.0 0.005 0.0001" '
                  'rgba="0.75 0.4 0.3 1" condim="3"/>\n'
                  % (name, k, cx, cy, cz, sx, sy, sz))
    return asset, geoms
