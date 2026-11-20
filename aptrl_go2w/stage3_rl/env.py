"""APT-RL Go2-W MuJoCo environment (paper Sec. 4.3, Table S1-S3).

Single-env class used by multiprocessing workers.

Observation (paper Table S1), worker returns flat 426-dim:
- proprio 117: base lin vel (3), ang vel (3), height (1), foot heights (4),
  contacts (4), cmd (3), proj gravity (3), q (12), dq (12), q hist (12),
  dq hist (12), tau (12), tau hist (12), a_aux (12), a_aux hist (12)
- extero 255: 210 height samples + 45 LiDAR bins (2.5D elevation ahead)
- gait info 54: gait sel hist (10), gait out hist (10), latent (16),
  latent hist (16), selected flag (1), unselected flag (1)
Learner: actor = enc(extero)[32] + proprio + gait = 203; critic = raw 426.

Action 29: a_latent 16 (frozen per-gait torque decoder), a_aux 12 (PD
position offset, scale 0.2), gait logit 1 (2 Hz switch, 0.5 s hold).

Control law (paper 4.3):
  tau = tau_dec + kp (q_default - q + 0.2 a_aux) - kd qdot   at 500 Hz
  tau_dec = D_{trot|bound}(a_latent)   switched at 2 Hz (numpy decoder)
Adaptation for Go2-W: wheel velocity governor so the wheeled-leg robot holds
traction; legs remain the primary locomotion learned via APT decoder + aux.

Training terrain: [flat 2 m][segment 8 m][flat 2 m] strip; success = cross
segment end. Eval: fixed npz race track (104 m, with documented fixes).
"""
import os
import sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.eval_protocol import initial_pose
from common.training_terrain import APPROVED_SINGLE_GAP_WIDTHS, training_heightmap
from common.terrain import (build_segment, load_npz_track, HeightField,
                            build_env_terrain_xml, TERRAIN_TYPES, TRACK_W)

BASE_XML = os.path.join(ROOT, "assets", "go2w.xml")
TVAE_DEC = os.path.join(ROOT, "stage2_tvae", "tvae_decoders.npz")

LEGS = ("FL", "FR", "RL", "RR")
LEG_JOINTS = [f"{l}_{p}_joint" for l in LEGS for p in ("hip", "thigh", "calf")]
WHEEL_JOINTS = [f"{l}_foot_joint" for l in LEGS]
EFFORT = {"hip": 23.7, "thigh": 23.7, "calf": 35.55}
KP = np.array([80.0, 100.0, 120.0] * 4)
KD = 3.5
AUX_SCALE = 0.2
WHEEL_R = 0.086
WHEEL_KW = 1.2
WHEEL_TMAX = 10.0
Q_DEFAULT = np.array([0.0, 0.9, -1.8] * 4)

P_Q, P_DQ = 21, 33
PROPRIO = 117
EXTERO = 255
GAIT_DIM = 54
OBS_WORKER = PROPRIO + EXTERO + GAIT_DIM   # 426
ACT_DIM = 29
TRAIN_COMMAND_SPEEDS = (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5)


def resolve_command_speed(sampled_speed, fixed_speed=None):
    """Use the sampled command unless a supported training override is set."""
    if fixed_speed is None:
        return float(sampled_speed)
    fixed_speed = float(fixed_speed)
    if not any(np.isclose(fixed_speed, value) for value in TRAIN_COMMAND_SPEEDS):
        raise ValueError(f"unsupported command speed: {fixed_speed}")
    return fixed_speed


def command_speed_at_step(start_speed, end_speed, ramp_steps, step):
    """Return the linearly ramped training command at a policy-step index."""
    start_speed = resolve_command_speed(0.0, start_speed)
    end_speed = resolve_command_speed(0.0, end_speed)
    if int(ramp_steps) != ramp_steps or ramp_steps <= 0:
        raise ValueError("ramp_steps must be a positive integer")
    if int(step) != step or step < 0:
        raise ValueError("step must be a non-negative integer")
    if end_speed <= start_speed:
        raise ValueError("command ramp end must be greater than start")
    fraction = min(float(step) / int(ramp_steps), 1.0)
    return start_speed + (end_speed - start_speed) * fraction


HS = 0.1
_HS_X = np.concatenate([np.arange(-0.8, -0.1, 0.1), np.arange(0.2, 1.51, 0.1)])
_HS_Y = np.concatenate([np.arange(-0.5, -0.09, 0.1), np.arange(0.1, 0.51, 0.1)])
_LIDAR_D = np.arange(0.6, 5.01, 0.1)

SEG_LEN = 8.0
APPROACH = 2.0
EXIT = 2.0
MID_GAP = 2.0


def _robot_xml_without_ground_path(path):
    with open(path, encoding="utf-8") as stream:
        txt = stream.read()
    txt = txt.replace(
        '  <geom name="ground" type="plane" size="0 0 0.05" pos="0 0 0" '
        'friction="1.0 0.005 0.0001" condim="3"/>\n', "")
    txt = txt.replace(
        '  <light directional="true" pos="0 0 3" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>\n',
        '  <light directional="true" pos="0 0 3" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>\n'
        '  <geom name="fail_floor" type="plane" pos="0 0 -30" size="40 40 0.05" '
        'friction="1.0" condim="3"/>\n')
    return txt


def load_decoders(path=TVAE_DEC):
    """Load the versioned numpy per-gait torque decoder artifact."""
    z = np.load(path)
    out = {}
    for g in ("trot", "bound"):
        out[g] = tuple(z[f"{g}_{k}"] for k in ("w1", "b1", "w2", "b2", "w3", "b3"))
    out["t_mean"] = z["t_mean"]
    out["t_std"] = z["t_std"]
    out["checkpoint_sha256"] = (str(z["checkpoint_sha256"].item())
                                 if "checkpoint_sha256" in z else None)
    return out


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) *
                                    (x + 0.044715 * x ** 3)))


def decode_torque_np(params, a_latent, gait):
    w1, b1, w2, b2, w3, b3 = params[gait]  # torch Linear layout (out, in)
    h = gelu(a_latent @ w1.T + b1)
    h = gelu(h @ w2.T + b2)
    t = h @ w3.T + b3
    return t * params["t_std"] + params["t_mean"]


LANE_CENTER = 4.0
LANE_HALF_WIDTH = 1.9


def within_track_lane(y: float) -> bool:
    """Return whether y stays strictly inside the report's full-track lane."""
    return abs(float(y) - LANE_CENTER) < LANE_HALF_WIDTH


def track_success(x: float, y: float, success_x: float) -> bool:
    """The track is complete only after the finish while still in lane."""
    return float(x) >= float(success_x) and within_track_lane(y)


def is_bad_contact_pair(body1, body2, bad_bodies) -> bool:
    """Preserve the environment's body-ID XOR rule for terminal contacts."""
    return (body1 in bad_bodies) ^ (body2 in bad_bodies)


def failed_termination_adjustment(max_x):
    """Failure remains a penalty; absolute track position cannot reward failure."""
    return -1.5


def joint_acceleration_reward(dq, previous_dq, dt):
    """Table S3 acceleration penalty from adjacent policy-step velocities."""
    ddq = (np.asarray(dq) - np.asarray(previous_dq)) / float(dt)
    return -1e-5 * float(ddq @ ddq)


def apply_base_velocity_kick(qvel, kick):
    """Apply a training push to free-joint world linear x/y velocity."""
    qvel[:2] += np.asarray(kick, dtype=qvel.dtype)


def transform_material_aux(aux, limit):
    """Canonicalize the float32 historical boundary, then apply the limit."""
    value = np.asarray(aux, dtype=np.float64)
    boundary = float(np.float32(0.2))
    value = np.where(value == boundary, 0.2, value)
    value = np.where(value == -boundary, -0.2, value)
    return np.clip(value, -limit, limit)


def material_wheel_scale(action):
    """Bound a policy-selected wheel-speed residual around the legacy target."""
    value = float(action)
    if not np.isfinite(value):
        raise ValueError('wheel action must be finite')
    return float(np.clip(1.0 + value, 0.0, 1.5))


class Go2WEnv:
    def __init__(self, seed=0, fixed_track=None, eval_mode=False,
                 episode_len=600, train_iter=0.0, base_xml=None,
                 eval_protocol="legacy", train_stage="mixed", gap_width_m=None,
                 train_track=None, decoder_path=None, command_speed=None,
                 command_ramp=None, reward_profile='legacy', material_aux_limit=0.2,
                 material_eval_start=None, material_start_profile='default',
                 material_wheel_control='none'):
        valid_stages = {"mixed", "flat", "rough_flat", "single_gap",
                        "double_gap", "full_track", "rough_slope_retain"}
        if train_stage not in valid_stages:
            raise ValueError(f"unknown training stage: {train_stage}")
        if fixed_track is not None and train_stage != "mixed":
            raise ValueError("training stage cannot change official fixed-track evaluation")
        if train_stage in ("single_gap", "double_gap", "full_track",
                           "rough_slope_retain") and not train_track:
            raise ValueError(f"{train_stage} requires an explicit train_track")
        if train_stage == "single_gap" and gap_width_m not in APPROVED_SINGLE_GAP_WIDTHS:
            raise ValueError("unsupported training gap width")
        if command_speed is not None and command_ramp is not None:
            raise ValueError("fixed command speed and command ramp are mutually exclusive")
        if (command_speed is not None or command_ramp is not None) and (
                train_stage != "flat" or eval_mode or fixed_track is not None):
            raise ValueError("command override is restricted to flat training")
        if command_ramp is not None:
            if len(command_ramp) != 3:
                raise ValueError("command ramp requires start, end, and step count")
            start_speed, end_speed, ramp_steps = command_ramp
            command_speed_at_step(start_speed, end_speed, ramp_steps, 0)
            command_ramp = (float(start_speed), float(end_speed), int(ramp_steps))
        if eval_protocol != "legacy" and (fixed_track is None or not eval_mode):
            raise ValueError("non-legacy eval protocol requires fixed-track evaluation")
        if reward_profile not in ('legacy', 'paper_adapted', 'stair_v1', 'integrated_v1', 'tire_v1', 'tire_l1_v1'):
            raise ValueError('unknown reward profile')
        profile_track = fixed_track or train_track
        if material_wheel_control not in ('none', 'residual_v1'):
            raise ValueError('unknown material wheel control')
        if material_wheel_control != 'none' and (not profile_track or not str(profile_track).lower().endswith('.xml')):
            raise ValueError('wheel residual requires a material-track XML')
        self.material_wheel_control = material_wheel_control
        if reward_profile != 'legacy' and (not profile_track or not str(profile_track).lower().endswith('.xml')):
            raise ValueError('experimental reward requires a material-track XML')
        if material_eval_start is not None and (not eval_mode or not profile_track or
                not str(profile_track).lower().endswith('.xml') or material_eval_start not in (21.5, 30.2)):
            raise ValueError('material eval start must be a known safe run-up')
        if material_start_profile not in ('default', 'stair_mix', 'flat_only', 'stair_only', 'tire_only'):
            raise ValueError('unknown material start profile')
        if material_start_profile != 'default' and (eval_mode or train_stage != 'full_track' or
                not train_track or not str(train_track).lower().endswith('.xml')):
            raise ValueError('stair mix requires material full-track training')
        self.material_start_profile = material_start_profile
        self.material_eval_start = material_eval_start
        self.reward_profile = reward_profile
        if material_aux_limit is not None and (not np.isfinite(material_aux_limit) or material_aux_limit <= 0 or material_aux_limit > 8):
            raise ValueError('material auxiliary limit must be in (0,8] or None')
        self.material_aux_limit = material_aux_limit
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.eval_mode = eval_mode
        self.eval_protocol = eval_protocol
        self.train_stage = train_stage
        self.train_track = train_track
        self.gap_width_m = gap_width_m
        self.command_speed = (None if command_speed is None else
                              resolve_command_speed(0.0, command_speed))
        self.command_ramp = command_ramp
        self.local_training = train_stage in ("single_gap", "double_gap",
                                              "rough_slope_retain")
        self.episode_len = episode_len
        self.train_iter = train_iter
        self.base_xml_path = base_xml or BASE_XML
        self.robot_xml = _robot_xml_without_ground_path(self.base_xml_path)
        self.decoder_path = os.path.abspath(decoder_path or TVAE_DEC)
        self.dec = load_decoders(self.decoder_path)

        track_path = fixed_track or (train_track if train_stage in
                                     ("single_gap", "double_gap", "full_track",
                                      "rough_slope_retain") else None)
        self.material_track = None
        if track_path is not None:
            if str(track_path).lower().endswith('.xml'):
                from common.material_track import MaterialTrack
                self.material_track = MaterialTrack(track_path)
                if self.reward_profile == 'tire_v1':
                    from common.material_acceptance import verify_target_scene
                    verify_target_scene(track_path)
                elif self.reward_profile == 'tire_l1_v1':
                    from common.material_acceptance import file_sha256
                    if file_sha256(track_path) != 'faca948af15ec3462b378cb95b22520b9294cd1e9142d17351670d85ec547d05':
                        raise ValueError('tire_l1_v1 requires the frozen 4cm curriculum scene')
                if self.reward_profile in ('stair_v1', 'integrated_v1', 'tire_v1', 'tire_l1_v1'):
                    self.material_track.validate_stair_reward_layout()
                H = np.zeros((1041, 81), dtype=np.float32)
                boxes, seq = [], ['material_track']
                meta = {'length_m': 104.0, 'width_m': 8.0, 'fixes': []}
            else:
                H, boxes, seq, meta = load_npz_track(track_path, apply_fixes=True)
            if train_stage == "single_gap":
                H = training_heightmap(track_path, gap_width_m)
            self.fixed = dict(H=H, boxes=boxes, seq=seq, meta=meta,
                              hfield=HeightField(H, 0.0, 0.0))
            self.track_len = H.shape[0] * HS
        else:
            self.fixed = None
        self.cur_type_idx = int(self.rng.integers(len(TERRAIN_TYPES)))
        self.cur_level = 1
        self._terrain_sig = None
        self._build_model()
        self.reset()

    # ------------------------------------------------------------ terrain
    def _build_model(self):
        if self.fixed is not None:
            sig = "fixed"
        else:
            segA = (self.train_stage if self.train_stage in ("flat", "rough_flat")
                    else TERRAIN_TYPES[self.cur_type_idx])
            segB = (self.train_stage if self.train_stage in ("flat", "rough_flat")
                    else TERRAIN_TYPES[(self.cur_type_idx + 5) % len(TERRAIN_TYPES)])
            sig = f"{segA}+{segB}|L{self.cur_level}"
        if sig == self._terrain_sig and hasattr(self, "model"):
            return  # cached terrain, nothing to rebuild
        self._terrain_sig = sig
        if self.fixed is not None:
            H = self.fixed["H"].copy()
            boxes = list(self.fixed["boxes"])
            self.boxes = boxes
            x0, y0 = 0.0, 0.0
            name = "track"
        else:
            # TWO-segment strip: success requires crossing both, so selection
            # pressure covers long-horizon postural endurance (the base-height
            # sag that kills long crossings at ~600 steps)
            segA = (self.train_stage if self.train_stage in ("flat", "rough_flat")
                    else TERRAIN_TYPES[self.cur_type_idx])
            segB = (self.train_stage if self.train_stage in ("flat", "rough_flat")
                    else TERRAIN_TYPES[(self.cur_type_idx + 5) % len(TERRAIN_TYPES)])
            level = self.cur_level
            HA, bA = build_segment(segA, level, SEG_LEN,
                                   seed=int(self.rng.integers(1 << 30)))
            HB, bB = build_segment(segB, level, SEG_LEN,
                                   seed=int(self.rng.integers(1 << 30)))
            n_seg = HA.shape[0]
            n_pre = int(APPROACH / HS)
            n_mid = int(MID_GAP / HS)
            n_post = int(EXIT / HS)
            H = np.zeros((n_pre + 2 * n_seg + n_mid + n_post, TRACK_W),
                         dtype=np.float32)
            H[n_pre:n_pre + n_seg] = HA
            H[n_pre + n_seg + n_mid:n_pre + 2 * n_seg + n_mid] = HB
            boxes = []
            for b in bA:
                boxes.append(dict(x0=b["x0"] + APPROACH, x1=b["x1"] + APPROACH,
                                  y0=b["y0"], y1=b["y1"], z0=b["z0"], z1=b["z1"]))
            offB = APPROACH + SEG_LEN + MID_GAP
            for b in bB:
                boxes.append(dict(x0=b["x0"] + offB, x1=b["x1"] + offB,
                                  y0=b["y0"], y1=b["y1"], z0=b["z0"], z1=b["z1"]))
            x0, y0 = 0.0, 0.0
            name = "seg"
            self.boxes = boxes
        self.H = H
        asset, geoms = build_env_terrain_xml(
            H, boxes, x0=x0, y0=y0, name=name,
            exact_height=self.fixed is None or self.material_track is not None)
        if self.material_track is not None:
            geoms += '\n'.join(self.material_track.geoms)
        xml = self.robot_xml
        if "<asset>" in xml:
            xml = xml.replace("<asset>", f"<asset>\n{asset}")
        else:
            xml = xml.replace("<worldbody>",
                              f"<asset>\n{asset}</asset>\n\n  <worldbody>")
        xml = xml.replace("<worldbody>", f"<worldbody>\n{geoms}", 1)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        hfield_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
        z_base = float(m.geom_pos[hfield_geom_id][2])
        z_span = float(m.hfield_size[0][2])
        nrow, ncol = int(m.hfield_nrow[0]), int(m.hfield_ncol[0])
        actual_H = z_base + z_span * m.hfield_data[:nrow * ncol].reshape(nrow, ncol).T
        self.hfield = (self.material_track if self.material_track is not None
                       else HeightField(actual_H, x0, y0))
        self.surface_hfield = self.hfield
        self.nominal_geom_friction = m.geom_friction.copy()
        self.leg_qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                                  for j in LEG_JOINTS])
        self.leg_vadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                                  for j in LEG_JOINTS])
        self.leg_act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{j}")
                                 for j in LEG_JOINTS])
        self.wheel_vadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                                    for j in WHEEL_JOINTS])
        self.wheel_act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{j}")
                                   for j in WHEEL_JOINTS])
        self.body_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.foot_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{l}_foot")
                            for l in LEGS]
        self.foot_geom_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f'{l}_foot_0')
                              for l in LEGS]
        if any(i < 0 for i in self.foot_geom_ids):
            raise ValueError('required wheel collision geometry is missing')
        self.bad_bodies = set([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                               for n in ["trunk"] +
                               [f"{l}_{p}" for l in LEGS for p in ("hip", "thigh", "calf")]])
        self.leg_effort = np.tile(np.array([EFFORT["hip"], EFFORT["thigh"],
                                            EFFORT["calf"]]), 4)

    # ------------------------------------------------------------- reset
    def _update_centering(self):
        """Cross-track feedback wrapper: adjusts vy/yaw-rate COMMANDS (not the
        policy) to hold the lane center; identical in training and eval so the
        policy learns to follow heading corrections."""
        d = self.data
        R = d.xmat[self.body_root].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        x, y = d.xpos[self.body_root][0], d.xpos[self.body_root][1]
        # On the fixed track's narrow bridge (cols 38..41 -> true center y=3.95m),
        # target the exact geometric bridge centerline 3.95m.
        target_y = 3.95 if (self.fixed is not None and 55.5 <= x <= 64.0) else 4.0
        ey = y - target_y
        # base command (randomized per episode, paper-style command sampling)
        # + cross-track feedback; the policy must learn to FOLLOW commands
        self.wyaw_cmd = float(np.clip(
            getattr(self, "base_wyaw", 0.0) - 2.0 * ey - 1.5 * yaw, -0.8, 0.8))
        self.vy_cmd = float(np.clip(
            getattr(self, "base_vy", 0.0) - 1.0 * ey, -0.5, 0.5))

    def reset(self):
        if self.train_stage == "mixed" and self.fixed is None and not self.eval_mode:
            # crawl (index 8) gets 2x sampling: it is the least-practiced
            # obstacle (mastery loop keeps its level low)
            step = 2 if self.cur_type_idx == 8 else 1
            self.cur_type_idx = (self.cur_type_idx + step) % len(TERRAIN_TYPES)
        self._build_model()
        if not hasattr(self, "stats"):
            self.stats = dict(episodes=0, success=0, level_sum=0.0,
                              types={})
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.reward_profile in ('stair_v1', 'integrated_v1', 'tire_v1', 'tire_l1_v1'):
            from stage3_rl.stair_reward import StairReward
            self.stair_shaper = StairReward()
        if self.reward_profile in ('tire_v1', 'tire_l1_v1'):
            from stage3_rl.tire_reward import TireReward
            self.tire_shaper = TireReward()
        if self.material_eval_start is not None:
            local_rng = np.random.default_rng(self.seed ^ 0xC0FFEE)
            x0 = self.material_eval_start + float(local_rng.uniform(-0.05, 0.05))
            y0 = 4.0 + float(local_rng.uniform(-0.1, 0.1))
            yaw0 = float(local_rng.uniform(-0.02, 0.02))
        elif self.local_training or (self.train_stage in ("flat", "rough_flat")
                                   and self.eval_mode):
            local_rng = (np.random.default_rng(self.seed ^ 0xC0FFEE)
                         if self.eval_mode else self.rng)
            # Begin inside a safe run-up, away from the x=16/24 seams.
            base_x = (15.0 if self.train_stage == "rough_slope_retain" else
                      24.5 if self.local_training else 0.5)
            x0 = base_x + float(local_rng.uniform(-0.05, 0.05))
            y0 = 4.0 + float(local_rng.uniform(-0.1, 0.1))
            yaw0 = float(local_rng.uniform(-0.02, 0.02))
        elif self.train_stage == "full_track" and not self.eval_mode:
            # Multi-section practice starts along actual flat plateaus
            # on the training track; official eval_mode=True never enters this branch.
            safe_starts = (
                0.5,                    # 0: start -> rough_flat -> rough_slope
                14.8,                   # 2: rough_slope -> gap
                24.5, 24.5, 24.5,       # 3: gap (0.5m + 0.4m abyss)
                32.2, 32.2,             # 4: continuous_gap (washboard)
                39.3, 39.3,             # 5: stairs_up (valley descent + ascent)
                47.9, 47.9,             # 6: stairs_down (hill ascent + descent)
                56.1, 56.1, 56.1,       # 7: stepping_one_bridge (narrow 0.4m beam)
                63.5, 63.5, 63.5,       # 8: stool (three 0.49m box obstacles)
                72.0,                   # 9: crawl (0.40m overhead bar)
                79.5, 79.5,             # 10: stairs_cliff (0.86m cliff drop)
                87.5,                   # 11: slope_cliff (0.67m cliff drop)
            )
            if self.material_track is not None:
                # The legacy 104m map's starts overlap tires and stairs in this
                # scene. Mix safe start-line and tire-approach run-ups instead.
                safe_starts = {
                    'default': (0.5, 0.5, 21.5, 21.5),
                    'stair_mix': (0.5, 0.5, 21.5, 21.5, 30.2, 30.2),
                    'flat_only': (0.5,),
                    'stair_only': (30.2,),
                    'tire_only': (21.5,),
                }[self.material_start_profile]
            base_x = float(self.rng.choice(safe_starts))
            y_center = 3.95 if 55.5 <= base_x <= 63.0 else 4.0
            y_jitter = 0.03 if 55.5 <= base_x <= 63.0 else 0.10
            x0 = base_x + float(self.rng.uniform(-0.05, 0.05))
            y0 = y_center + float(self.rng.uniform(-y_jitter, y_jitter))
            yaw0 = float(self.rng.uniform(-0.02, 0.02))
        elif self.fixed is None and not self.eval_mode:
            x0, y0, yaw0 = float(self.rng.choice([0.5, 1.2])), 4.0, 0.0
        else:
            x0, y0, yaw0 = initial_pose(self.seed, self.eval_protocol)
        self.requested_initial_pose = [x0, y0, yaw0]
        h0 = float(self.hfield.height(x0, y0))
        d.qpos[0:3] = [x0, y0, h0 + 0.36]
        d.qpos[3:7] = [np.cos(yaw0 / 2.0), 0, 0, np.sin(yaw0 / 2.0)]
        d.qpos[self.leg_qadr] = Q_DEFAULT
        if self.local_training:
            self.success_x = {"single_gap": 28.3, "double_gap": 32.0,
                              "rough_slope_retain": 23.5}[self.train_stage]
        elif self.fixed is not None:
            self.success_x = self.track_len - 4.0
        else:
            self.success_x = APPROACH + 2 * SEG_LEN + MID_GAP - 0.4
        # settle 0.25 s with PD hold
        for i in range(125):
            mujoco.mj_forward(m, d)
            q = d.qpos[self.leg_qadr]
            vq = d.qvel[self.leg_vadr]
            tau = np.clip(KP * (Q_DEFAULT - q) - KD * vq,
                          -self.leg_effort, self.leg_effort)
            d.ctrl[self.leg_act] = tau
            d.ctrl[self.wheel_act] = 0.0
            mujoco.mj_step(m, d)
        # domain randomization (light, paper-style)
        if not self.eval_mode:
            self.motor_gain = float(self.rng.uniform(0.9, 1.1))
            self.friction_gain = float(self.rng.uniform(0.85, 1.15))
            for g in range(m.ngeom):
                if self.nominal_geom_friction[g][0] > 0:
                    m.geom_friction[g][0] = (self.nominal_geom_friction[g][0]
                                             * self.friction_gain)
            # drop-preview types get a slow-command curriculum: the panic-slam
            # on seeing a descent is only fixable if careful approaches are
            # discovered first
            tn = (self.train_stage if self.train_stage != "mixed"
                  else TERRAIN_TYPES[self.cur_type_idx])
            if tn in ("stairs_up", "stairs_cliff", "stepping_one_bridge",
                      "crawl", "slope_cliff") and self.cur_level <= 4:
                self.vx_cmd = float(self.rng.choice([0.5, 0.8, 1.0, 1.2]))
            else:
                self.vx_cmd = float(self.rng.choice([0.5, 1.0, 1.5, 2.0, 2.5]))
            # paper-style command richness: vy in [-1,1] and target-heading
            # yaw-rate offsets so the policy learns command following
            self.base_vy = float(self.rng.uniform(-0.3, 0.3))
            self.base_wyaw = float(self.rng.uniform(-0.4, 0.4))
        else:
            self.motor_gain = 1.0
            self.vx_cmd = 1.5  # fixed eval pace: 104 m fits in the 90 s budget
            self.base_vy = 0.0
            self.base_wyaw = 0.0
        if self.command_speed is not None:
            self.vx_cmd = self.command_speed
        elif self.command_ramp is not None:
            self.vx_cmd = command_speed_at_step(*self.command_ramp, 0)
        elif self.train_stage in ("full_track", "single_gap", "double_gap",
                                  "rough_slope_retain") and not self.eval_mode:
            seg = min(max(int(float(d.qpos[0]) // 8.0), 0), 12)
            self.vx_cmd = (0.8 if 22 <= d.qpos[0] < 47 else 1.5) if self.material_track is not None else (0.8 if seg in {5, 6, 7, 9, 10, 11} else 1.5)
            self.base_vy = float(self.rng.uniform(-0.1, 0.1))
            self.base_wyaw = float(self.rng.uniform(-0.15, 0.15))
        # allow every command to finish the strip: 12.4 m at vx (>=6 s);
        # the fixed track uses the explicit episode length
        if self.fixed is not None:
            self.env_episode_len = self.episode_len
        else:
            self.env_episode_len = max(900, int(21.5 / max(self.vx_cmd, 0.4) * 100))
        self.vy_cmd = 0.0
        self.wyaw_cmd = 0.0
        # gait state
        self.gait_sel = 0.0
        self.gait_out = 0.5
        self.gate_timer = 0.0
        self.prev_gait_sel = 0.0
        self.gait_sel_hist = np.zeros(10, dtype=np.float32)
        self.gait_out_hist = np.full(10, 0.5, dtype=np.float32)
        self.a_latent = np.zeros(16, dtype=np.float32)
        self.a_latent_hist = np.zeros(16, dtype=np.float32)
        self.a_aux = np.zeros(12, dtype=np.float32)
        self.a_aux_hist = np.zeros(12, dtype=np.float32)
        self.q_hist = d.qpos[self.leg_qadr].copy()
        self.dq_hist = d.qvel[self.leg_vadr].copy()
        self.tau_applied = np.zeros(12, dtype=np.float32)
        self.tau_hist = np.zeros(12, dtype=np.float32)
        self.prev_dq = d.qvel[self.leg_vadr].copy()
        self.t = 0.0
        self.steps = 0
        self.gait_mean = 0.5
        self.gait_change_acc = 0.0
        self.airtime = np.zeros(4)
        self.stancetime = np.zeros(4)
        self.last_contact = np.zeros(4)
        mujoco.mj_forward(m, d)
        R = d.xmat[self.body_root].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        self.settled_initial_pose = [float(d.qpos[0]), float(d.qpos[1]), float(yaw)]
        self.max_x = float(d.qpos[0])
        self._prev_reward_max_x = float(d.qpos[0])
        return self._get_obs()

    # -------------------------------------------------------------- obs
    def _velocities(self):
        """Free-joint qvel convention (verified): [linear WORLD, angular BODY]."""
        d = self.data
        R = d.xmat[self.body_root].reshape(3, 3)
        v_b = R.T @ d.qvel[0:3]
        w_b = d.qvel[3:6].copy()
        return v_b, w_b, R

    def _contacts(self):
        m, d = self.model, self.data
        foot_contact = np.zeros(4)
        bad_contact = 0.0
        for c in range(d.ncon):
            con = d.contact[c]
            b1 = m.geom_bodyid[con.geom1]
            b2 = m.geom_bodyid[con.geom2]
            for k, fb in enumerate(self.foot_bodies):
                if ((b1 == fb and b2 not in self.bad_bodies) or
                        (b2 == fb and b1 not in self.bad_bodies)):
                    foot_contact[k] = 1.0
            if is_bad_contact_pair(b1, b2, self.bad_bodies):
                bad_contact = 1.0
        return foot_contact, bad_contact

    def _terminal_contact_details(self):
        """Return JSON-safe details for contacts using the terminal body rule."""
        m, d = self.model, self.data
        details = []
        for i in range(d.ncon):
            contact = d.contact[i]
            body1 = int(m.geom_bodyid[contact.geom1])
            body2 = int(m.geom_bodyid[contact.geom2])
            if not is_bad_contact_pair(body1, body2, self.bad_bodies):
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            details.append({
                "geom1": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom1)
                         or f"geom_{geom1}",
                "geom2": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom2)
                         or f"geom_{geom2}",
                "body1": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body1)
                         or f"body_{body1}",
                "body2": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body2)
                         or f"body_{body2}",
                "distance": float(contact.dist),
                "position": [float(value) for value in contact.pos],
            })
        return details

    def _foot_heights(self):
        d = self.data
        hs = np.zeros(4, dtype=np.float32)
        for k, fb in enumerate(self.foot_bodies):
            x, y, z = d.xpos[fb]
            hs[k] = z - float(self.hfield.height(x, y)) - WHEEL_R
        return hs

    def _extero(self):
        d = self.data
        R = d.xmat[self.body_root].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        bx, by, bz = d.xpos[self.body_root]
        gx = (_HS_X[None, :] * c - _HS_Y[:, None] * s + bx).reshape(-1)
        gy = (_HS_X[None, :] * s + _HS_Y[:, None] * c + by).reshape(-1)
        h = (self.material_track.height(gx, gy, include_overhead=True)
             if self.material_track is not None else self.hfield.height(gx, gy))
        # boxes (stools, crawl bars) into the 2.5D elevation preview
        for b in self.boxes:
            inside = ((gx >= b["x0"]) & (gx <= b["x1"]) &
                      (gy >= b["y0"]) & (gy <= b["y1"]))
            h = np.where(inside, np.maximum(h, b["z1"]), h)
        rel = np.clip(h - bz, -1.0, 1.0).astype(np.float32)
        lx = bx + _LIDAR_D * c
        ly = by + _LIDAR_D * s
        lh = (self.material_track.height(lx, ly, include_overhead=True)
              if self.material_track is not None else self.hfield.height(lx, ly))
        for b in self.boxes:
            inside = ((lx >= b["x0"]) & (lx <= b["x1"]) &
                      (ly >= b["y0"]) & (ly <= b["y1"]))
            lh = np.where(inside, np.maximum(lh, b["z1"]), lh)
        lidar = np.clip(lh - bz, -1.0, 1.0).astype(np.float32)
        return np.concatenate([rel, lidar])

    def _get_obs(self):
        v_b, w_b, R = self._velocities()
        foot_h = self._foot_heights()
        foot_c, _ = self._contacts()
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        q = self.data.qpos[self.leg_qadr]
        dq = self.data.qvel[self.leg_vadr]
        ext = self._extero()
        gait_flag = 1.0 if self.gate_timer < 1e-9 else 0.0
        bx, by, bz = self.data.xpos[self.body_root]
        rel_bz = float(np.clip(bz - self.hfield.height(bx, by), -1.0, 2.0))
        proprio = np.concatenate([
            v_b, w_b, [rel_bz], foot_h, foot_c,
            [self.vx_cmd, self.vy_cmd, self.wyaw_cmd], grav,
            q, dq, self.q_hist, self.dq_hist,
            self.tau_applied * self.motor_gain, self.tau_hist,
            self.a_aux, self.a_aux_hist,
        ]).astype(np.float32)
        gait = np.concatenate([
            self.gait_sel_hist, self.gait_out_hist,
            self.a_latent, self.a_latent_hist,
            [gait_flag], [1.0 - gait_flag],
        ]).astype(np.float32)
        return np.concatenate([proprio, ext, gait])

    # -------------------------------------------------------------- step
    def step(self, action):
        m, d = self.model, self.data
        action = np.asarray(action, dtype=np.float64)
        expected_dim = 30 if self.material_wheel_control == 'residual_v1' else 29
        if action.shape != (expected_dim,) or not np.isfinite(action).all():
            raise ValueError(f'expected {expected_dim} finite actions')
        action = np.clip(action, -8, 8)
        wheel_scale = material_wheel_scale(action[29]) if expected_dim == 30 else 1.0
        a_latent = action[:16]
        a_aux = action[16:28]
        # Experimental bound applies only to the versioned material scene.
        # The policy distribution remains over raw actions; the environment
        # executes this deterministic action transform, as with its [-8,8] clip.
        if self.material_track is not None and self.material_aux_limit is not None:
            a_aux = transform_material_aux(a_aux, self.material_aux_limit)
        gait_logit = action[28]

        # gait gate at 2 Hz; the overhead-bar segment forces trot (navigation-
        # layer gait hint: bound jumps collide with the bar, a low trot walks
        # under — the gate output remains learned for all other terrains)
        if self.gate_timer <= 1e-9:
            self.gait_out = 1.0 / (1.0 + np.exp(-gait_logit))
            self.gait_sel = 1.0 if self.gait_out >= 0.5 else 0.0
            # navigation-layer gait hint: trot under overhead bars (training
            # hints by segment type; eval hints by track segment 9)
            if (self.fixed is None and
                    TERRAIN_TYPES[self.cur_type_idx] == "crawl") or                     (self.fixed is not None and
                     self.material_track is None and 72.0 <= self.data.qpos[0] < 80.0):
                self.gait_sel = 0.0
            self.gait_sel_hist = np.roll(self.gait_sel_hist, -1)
            self.gait_sel_hist[-1] = self.gait_sel
            self.gait_out_hist = np.roll(self.gait_out_hist, -1)
            self.gait_out_hist[-1] = self.gait_out
            self.gait_change_acc += abs(self.gait_sel - self.prev_gait_sel)
            self.prev_gait_sel = self.gait_sel
        self.gate_timer += 0.01
        if self.gate_timer >= 0.5 - 1e-9:
            self.gate_timer = 0.0

        self.a_latent_hist = self.a_latent.copy()
        self.a_latent = a_latent
        self.a_aux_hist = self.a_aux.copy()
        self.a_aux = a_aux

        gait_name = "trot" if self.gait_sel < 0.5 else "bound"
        tau_dec = decode_torque_np(self.dec, a_latent, gait_name)
        tau_dec = tau_dec * self.motor_gain
        wheel_contact = self._contacts()[0]
        self.wheel_contact_last = wheel_contact

        for sub in range(5):
            q = d.qpos[self.leg_qadr]
            vq = d.qvel[self.leg_vadr]
            tau = tau_dec + KP * (Q_DEFAULT - q + AUX_SCALE * a_aux) - KD * vq
            tau = np.clip(tau, -self.leg_effort, self.leg_effort)
            d.ctrl[self.leg_act] = tau
            # differential drive: left/right wheel speeds follow vx_cmd and
            # the yaw-rate command (half-track 0.1934 m) — this gives the
            # platform real yaw authority, without which wyaw commands are
            # physically untrackable
            half_track = 0.1934
            w_diff = self.wyaw_cmd * half_track / WHEEL_R
            w_left = (self.vx_cmd / WHEEL_R) - w_diff
            w_right = (self.vx_cmd / WHEEL_R) + w_diff
            w_cmd = np.array([w_left, w_right, w_left, w_right]) * wheel_contact * wheel_scale
            w_now = d.qvel[self.wheel_vadr]
            tw = np.clip(WHEEL_KW * (w_cmd - w_now), -WHEEL_TMAX, WHEEL_TMAX)
            # mild yaw damping (smooths uncommanded yaw; the diff drive
            # overrides it when turning is commanded)
            wz = d.qvel[5]
            td = np.clip(-0.8 * wz, -3.0, 3.0)
            tw += self.wheel_contact_last * np.array([td, -td, td, -td])
            d.ctrl[self.wheel_act] = tw
            self.tau_applied = tau.copy()
            mujoco.mj_step(m, d)

        if getattr(self, "centering", True):
            self._update_centering()
        # push perturbation: ~1.25 Hz velocity kick on the base (training only)
        if not self.eval_mode and self.steps % 80 == 0 and self.steps > 0:
            kick = self.rng.normal(0.0, 0.25, 2)
            apply_base_velocity_kick(self.data.qvel, kick)
        self.t += 0.01
        self.steps += 1
        if self.command_ramp is not None:
            self.vx_cmd = command_speed_at_step(*self.command_ramp, self.steps)
        elif self.train_stage == "full_track" and not self.eval_mode:
            seg = min(max(int(float(d.qpos[0]) // 8.0), 0), 12)
            self.vx_cmd = (0.8 if 22 <= d.qpos[0] < 47 else 1.5) if self.material_track is not None else (0.8 if seg in {5, 6, 7, 9, 10, 11} else 1.5)
        self.max_x = max(getattr(self, "max_x", 0.0), float(d.qpos[0]))
        obs = self._get_obs()
        rew, terms = self._reward()
        done, reason, success = self._check_done()
        if self.reward_profile in ('stair_v1', 'integrated_v1', 'tire_v1', 'tire_l1_v1') and not self.eval_mode:
            wheel_contacts = []
            for contact in d.contact:
                for obstacle_geom, wheel_geom in ((contact.geom1, contact.geom2),
                                                   (contact.geom2, contact.geom1)):
                    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, obstacle_geom) or ''
                    body = int(m.geom_bodyid[wheel_geom])
                    if name.startswith(('stair_up_', 'stair_down_')) and body in self.foot_bodies:
                        wheel_contacts.append(self.foot_bodies.index(body))
            wheels = d.xpos[self.foot_bodies]
            stair_terms = self.stair_shaper.step(
                d.qpos[0], d.qpos[1], self.vx_cmd, done,
                wheel_x=wheels[:, 0], wheel_y=wheels[:, 1], wheel_z=wheels[:, 2],
                stair_wheel_contacts=wheel_contacts)
            terms.update(stair_terms)
            rew += sum(stair_terms.values())
        if self.reward_profile in ('tire_v1', 'tire_l1_v1') and not self.eval_mode:
            tire_wheel_contact = False
            for contact in d.contact:
                for obstacle_geom, wheel_geom in ((contact.geom1, contact.geom2),
                                                   (contact.geom2, contact.geom1)):
                    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, obstacle_geom) or ''
                    if name.startswith('tire_') and int(m.geom_bodyid[wheel_geom]) in self.foot_bodies:
                        tire_wheel_contact = True
            tire_terms = self.tire_shaper.step(
                d.qpos[0], d.qpos[1], done, wheel_xy=d.geom_xpos[self.foot_geom_ids, :2],
                tire_wheel_contact=tire_wheel_contact)
            terms.update(tire_terms)
            rew += sum(tire_terms.values())
        # A completed local segment gets a bonus; absolute position never pays
        # a failed episode, including local starts already beyond a global cap.
        if done and reason not in ("timeout",):
            if success:
                bonus = 25.0 if not self.eval_mode else 10.0
                rew += bonus
                terms["success_bonus"] = bonus
            else:
                penalty = failed_termination_adjustment(self.max_x)
                if not self.eval_mode and self.reward_profile == 'legacy':
                    penalty -= 23.5
                rew += penalty
                terms["termination"] = penalty
        # update histories
        self.q_hist = d.qpos[self.leg_qadr].copy()
        self.dq_hist = d.qvel[self.leg_vadr].copy()
        self.tau_hist = self.tau_applied.copy()
        info = dict(terms=terms, reason=reason, success=success)
        if done:
            info["terminal_contacts"] = self._terminal_contact_details()
        return obs, rew, done, info

    # ------------------------------------------------------------ reward
    def _reward(self):
        d = self.data
        v_b, w_b, R = self._velocities()
        foot_c, bad = self._contacts()
        foot_h = self._foot_heights()
        q = d.qpos[self.leg_qadr]
        dq = d.qvel[self.leg_vadr]
        dt = 0.01
        foot_v = np.array([d.cvel[fb][3:6] for fb in self.foot_bodies])
        foot_speed = np.linalg.norm(foot_v, axis=1)
        t = {}

        # ---- task ----
        # paper Table S3 tracks the commanded BODY XY velocity (both axes);
        # the vy channel is what lets the centering wrapper correct drift
        evx = self.vx_cmd - v_b[0]
        evy = self.vy_cmd - v_b[1]
        err = np.sqrt(evx * evx + evy * evy) / max(abs(self.vx_cmd), 1.0)
        velocity_weight = 1.5 if self.reward_profile != 'legacy' else 2.5
        t["lin_vel"] = velocity_weight * float(np.exp(-err * err / 0.25))
        if not self.eval_mode and self.reward_profile in ('integrated_v1', 'tire_v1', 'tire_l1_v1'):
            v_world_x = float(d.qvel[0])
            t['forward_progress'] = float(np.clip(v_world_x / max(self.vx_cmd, 0.5), -0.5, 1.5))
            t['alive'] = 0.5
        if not self.eval_mode and self.reward_profile == 'legacy':
            v_world_x = float(d.qvel[0])
            t["forward_progress"] = 1.0 * float(np.clip(v_world_x / max(self.vx_cmd, 0.5), -0.5, 1.5))
            t["alive"] = 0.5
            cur_max = float(getattr(self, "max_x", d.qpos[0]))
            prev_max = float(getattr(self, "_prev_reward_max_x", cur_max))
            if cur_max > prev_max:
                t["new_max_x"] = 4.0 * (cur_max - prev_max)
                # Milestone bonuses for clearing discrete obstacles in training
                milestones = (27.1, 30.6, 39.8, 47.8, 55.8, 63.8, 66.3, 68.9, 71.5, 86.2, 96.2, 100.0)
                m_bonus = sum(12.0 for m in milestones if prev_max < m <= cur_max)
                if m_bonus > 0.0:
                    t["milestone"] = m_bonus
            self._prev_reward_max_x = cur_max
        err_w = self.wyaw_cmd - w_b[2]
        t["ang_vel"] = 0.5 * float(np.exp(-err_w * err_w / 0.25))

        # ---- regularization ----
        dl = self.a_latent - self.a_latent_hist
        t["latent_smooth"] = -0.01 * float(dl @ dl) / dt * dt
        da = self.a_aux - self.a_aux_hist
        t["aux_smooth"] = -0.01 * float(da @ da) / dt * dt
        t["aux_scale"] = -0.007 * float(self.a_aux @ self.a_aux)
        t["torque"] = -1.1e-7 * float(self.tau_applied @ self.tau_applied)
        t["joint_acc"] = joint_acceleration_reward(dq, self.prev_dq, dt)

        # ---- gait related ----
        c_gd = max(0.0, 1.0 - self.train_iter / 250.0)
        self.gait_mean = 0.995 * self.gait_mean + 0.005 * self.gait_sel
        t["gait_diversity"] = -0.3 * c_gd * (self.gait_mean - 0.5) ** 2
        t["gait_change"] = -0.08 * self.gait_change_acc
        self.gait_change_acc *= 0.5

        # ---- style ----
        swing = foot_c < 0.5
        t["foot_clearance"] = -0.6 * float(np.sum(
            np.maximum(0.0, 0.12 - np.clip(foot_h[swing], -0.2, 0.5)) ** 2))
        # brushing: swing feet near ground moving fast
        near_ground = swing & (foot_h < 0.025) & (foot_h > -0.15)
        t["foot_brushing"] = -0.02 * float(np.sum(foot_speed[near_ground] ** 2))
        # airtime on touchdown / stance time on liftoff
        at = 0.0
        for k in range(4):
            if foot_c[k] > 0.5 and self.last_contact[k] < 0.5:
                at += min(self.airtime[k], 0.25)
            if foot_c[k] < 0.5 and self.last_contact[k] > 0.5:
                at += min(self.stancetime[k], 0.3)
            if foot_c[k] > 0.5:
                self.airtime[k] = 0.0
                self.stancetime[k] += dt
            else:
                self.airtime[k] += dt
                self.stancetime[k] = 0.0
        t["foot_airtime"] = 0.1 * at
        self.last_contact = foot_c.copy()
        t["nominal"] = -0.015 * float((q - Q_DEFAULT) @ (q - Q_DEFAULT)) / (1.0 + self.vx_cmd)
        foot_vz = np.clip(np.array([d.cvel[fb][5] for fb in self.foot_bodies]), -2.5, 2.5)
        t["foot_contact_vel"] = -0.015 * float(np.sum(foot_vz ** 2 * foot_c))
        hip_off = np.abs(q[[0, 3, 6, 9]])
        t["roll_limit"] = -0.04 * float(np.sum(hip_off > np.pi / 4))
        t["body_shaking"] = -0.15 * (float(v_b[2]) ** 2 +
                                    0.5 * (float(w_b[0]) ** 2 + float(w_b[1]) ** 2))
        n_contact = float(foot_c.sum())
        expect_set = (2.0, 4.0) if self.gait_sel < 0.5 else (0.0, 2.0)
        t["simultaneous"] = -0.025 * min((n_contact - e) ** 2 for e in expect_set)

        rew = sum(t.values())
        self.prev_dq = dq.copy()
        return rew, t

    # ------------------------------------------------------- termination
    def _check_done(self):
        m, d = self.model, self.data
        _, bad = self._contacts()
        R = d.xmat[self.body_root].reshape(3, 3)
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
        z = d.xpos[self.body_root][2]
        x = d.xpos[self.body_root][0]
        y = d.xpos[self.body_root][1]
        h0 = float(self.hfield.height(x, y))
        reason, success = None, False
        if bad > 0.5:
            reason = "body_contact"
        elif abs(roll) > 0.6 or abs(pitch) > 1.0:
            reason = "attitude"
        elif z < h0 + 0.10:
            reason = "fall"
        elif not within_track_lane(y):
            reason = "off_lane"
        elif self.steps >= getattr(self, "env_episode_len", self.episode_len):
            reason = "timeout"
        if reason in (None, "timeout") and track_success(x, y, self.success_x):
            reason, success = "success", True
        elif (reason == "timeout" and getattr(self, "fixed", None) is None and
              getattr(self, "max_x", 0.0) >= self.success_x - 4.0):
            # Curriculum-only partial progress; fixed-track evaluation requires
            # the exact finish line at success_x.
            reason, success = "success", True
        return reason is not None, reason, success

    # ------------------------------------------------------- curriculum
    def report(self, success):
        if self.train_stage != "mixed":
            st = self.stats
            st["episodes"] += 1
            st["success"] += int(success)
            st["level_sum"] += self.cur_level
            st["types"][self.train_stage] = st["types"].get(self.train_stage, 0) + 1
            return
        if self.fixed is not None:
            return
        st = self.stats
        st["episodes"] += 1
        st["success"] += int(success)
        st["level_sum"] += self.cur_level
        tn = TERRAIN_TYPES[self.cur_type_idx]
        st["types"][tn] = st["types"].get(tn, 0) + 1
        if success:
            self.cur_level = min(self.cur_level + 1, 10)
        else:
            self.cur_level = max(self.cur_level - 1, 1)
        # terrain sig is derived from (type, level); _build_model rebuilds
        # only when it actually changes

