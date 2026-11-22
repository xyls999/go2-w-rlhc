"""APT-RL PPO with async env workers (CPU policy replicas + GPU learner).

Throughput design for Windows (pipes ~1-2 ms latency and limited bandwidth):
- Each worker process holds a CPU copy of the actor + obs normalizer, runs
  H=50 policy steps autonomously on its envs, writes each trajectory chunk to
  a DISTINCT shared-memory slot (K slots per iteration, no flow control
  needed) and announces the slot via a tiny pipe message.
- The learner aggregates K chunks per worker until batch >= ~204800, runs
  GAE + PPO update on GPU, then broadcasts weights + norm stats.
- Policy staleness is one chunk (0.5 s experience); adaptive KL keeps the
  update conservative (paper Table S4).

Networks follow paper S2.3/S2.4: teacher extero encoder 255->32, actor
203->[512,256,128]->29 (16 latent + 12 aux + 1 gait logit), critic on raw 426.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.eval_protocol import material_command_speed
from common.material_acceptance import policy_material_command_profile

PROPRIO, EXTERO, GAIT = 117, 255, 54
OBS_WORKER = PROPRIO + EXTERO + GAIT
ACT_DIM = 29
LATENT, AUXD = 16, 12
CHUNK_H = 50
TRAIN_STAGES = ("mixed", "flat", "rough_flat", "single_gap",
                "double_gap", "full_track", "gap_rough_retain",
                "track_curriculum")
GAP_WIDTHS = (0.1, 0.2, 0.3, 0.4, 0.5)
AUX_W = 32          # 29 action + logp + rew + done
REW_SCALE = 0.1


def apply_material_start_profile(env, profile):
    """Apply recorded start ramp only to material episodes born at the start."""
    if profile not in ('instant_v1', 'start_ramp_100_v1'):
        raise ValueError('unsupported material command profile')
    if (profile == 'start_ramp_100_v1' and env.material_track is not None and
            env.requested_initial_pose[0] < 1.0):
        env.vx_cmd = material_command_speed(env.steps, float(env.data.qpos[0]), profile)


def compute_chunk_count(batch_target, workers, envs_per_worker,
                        chunk_h=CHUNK_H) -> int:
    """Return the minimum whole chunks needed to meet a sample target."""
    values = (batch_target, workers, envs_per_worker, chunk_h)
    if any(value <= 0 for value in values):
        raise ValueError("batch, worker, environment, and chunk counts must be positive")
    return int(np.ceil(batch_target / (workers * envs_per_worker * chunk_h)))


def should_probe(iteration_zero_based: int, interval: int = 25) -> bool:
    """Probe after every complete interval, including a 25-update pilot."""
    if interval <= 0:
        raise ValueError("probe interval must be positive")
    return (iteration_zero_based + 1) % interval == 0


def ensure_empty_output_dir(out_dir):
    """Create a run directory or reject an existing non-empty directory."""
    path = os.path.abspath(os.fspath(out_dir))
    os.makedirs(path, exist_ok=True)
    if os.listdir(path):
        raise FileExistsError(f"refusing to reuse non-empty training directory: {path}")
    return path


def write_probe_record(path, record) -> None:
    """Append a JSON-safe probe record as one line without truncating history."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Not JSON serializable: {type(value).__name__}")

    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=convert, sort_keys=True) + "\n")
        stream.flush()


def worker_stage_for_index(train_stage: str, index: int, workers: int) -> str:
    """Keep mixed retention assignments fixed for a complete PPO run."""
    if workers <= 0 or not 0 <= index < workers:
        raise ValueError("invalid worker index or count")
    if train_stage == "gap_rough_retain":
        if workers != 4:
            raise ValueError("gap_rough_retain requires exactly four workers")
        return "single_gap" if index % 2 == 0 else "rough_slope_retain"
    if train_stage == "track_curriculum":
        stages = ("full_track", "full_track", "full_track", "double_gap")
        return stages[index % len(stages)]
    return train_stage


def material_start_for_worker(profile, index, workers):
    """Assign a deterministic start distribution for each worker."""
    if workers <= 0 or not 0 <= index < workers:
        raise ValueError('invalid worker index or count')
    if profile == 'full_start':
        return 'flat_only'
    if profile in ('stair_retain', 'tire_retain'):
        if workers != 4:
            raise ValueError(f'{profile} requires exactly four workers')
        target = 'stair_only' if profile == 'stair_retain' else 'tire_only'
        return 'flat_only' if index % 2 == 0 else target
    if profile not in ('default', 'stair_mix'):
        raise ValueError('unknown material start profile')
    return profile


def validate_training_args(args):
    """Fail before worker launch if a training-only profile is incomplete."""
    selected = [name for name in ('residual_actor', 'residual_v2', 'residual_v3', 'residual_v4')
                if getattr(args, name, False)]
    if len(selected) > 1:
        raise ValueError('choose only one residual architecture')
    if any(name in selected for name in ('residual_v2', 'residual_v3', 'residual_v4')) and not getattr(args, 'init', None):
        raise ValueError('trainable residual requires an initial base checkpoint')
    if 'residual_v4' in selected and (not getattr(args, 'train_track', None) or
            not str(args.train_track).lower().endswith('.xml') or
            not str(getattr(args, 'probe_track', '')).lower().endswith('.xml')):
        raise ValueError('residual_v4 requires material training and probe tracks')
    local = args.train_stage in ("single_gap", "double_gap", "full_track",
                                 "gap_rough_retain", "track_curriculum")
    if local and (not args.train_track or not os.path.isfile(args.train_track)
                  or not os.access(args.train_track, os.R_OK)):
        raise ValueError(f"{args.train_stage} requires a readable --train-track")
    if (getattr(args, 'reward_profile', 'legacy') != 'legacy' and
            (args.train_stage != 'full_track' or not args.train_track or
             not args.train_track.lower().endswith('.xml'))):
        raise ValueError('experimental rewards require full_track with a material XML')
    if (getattr(args, 'material_aux_limit', 0.2) != 0.2 and
            (args.train_stage != 'full_track' or not args.train_track or
             not args.train_track.lower().endswith('.xml'))):
        raise ValueError('custom material auxiliary limit requires full_track with a material XML')
    if (getattr(args, 'material_start_profile', 'default') != 'default' and
            (args.train_stage != 'full_track' or not args.train_track or
             not args.train_track.lower().endswith('.xml'))):
        raise ValueError('stair start profile requires full_track with a material XML')
    if getattr(args, 'material_start_profile', 'default') in ('stair_retain', 'tire_retain') and (args.workers != 4 or args.epw != 1):
        raise ValueError('retention profiles require four workers and one environment each')
    if args.train_stage == "single_gap":
        if args.gap_width not in GAP_WIDTHS:
            raise ValueError("single_gap requires --gap-width 0.1/0.2/0.3/0.4/0.5")
    elif args.train_stage == "gap_rough_retain":
        if args.gap_width != 0.2 or args.workers != 4 or args.epw != 1:
            raise ValueError("gap_rough_retain requires width=0.2, workers=4, epw=1")
    elif args.gap_width is not None:
        raise ValueError("--gap-width is only valid for single_gap")
    ramp_values = (args.command_ramp_start, args.command_ramp_end,
                   args.command_ramp_steps)
    if any(value is not None for value in ramp_values):
        if any(value is None for value in ramp_values):
            raise ValueError("command ramp requires all three parameters")
        if args.command_speed is not None:
            raise ValueError("fixed command speed and command ramp are mutually exclusive")
        if args.train_stage != "flat":
            raise ValueError("command ramp is only valid with flat training")
        from stage3_rl.env import command_speed_at_step
        command_speed_at_step(args.command_ramp_start, args.command_ramp_end,
                              args.command_ramp_steps, 0)
    if args.command_speed is not None:
        from stage3_rl.env import TRAIN_COMMAND_SPEEDS
        if args.train_stage != "flat":
            raise ValueError("--command-speed is only valid with flat training")
        if not any(np.isclose(args.command_speed, value) for value in TRAIN_COMMAND_SPEEDS):
            raise ValueError("unsupported command speed")
    if args.train_stage in ("full_track", "track_curriculum") and args.episode_len < 9000:
        raise ValueError(f"{args.train_stage} requires --episode-len >= 9000")
    if args.train_track and not local:
        raise ValueError("--train-track is only valid for gap/full-track stages")


def worker_env_config(args):
    command_ramp = None
    if args.command_ramp_start is not None:
        command_ramp = (args.command_ramp_start, args.command_ramp_end,
                        args.command_ramp_steps)
    config = dict(train_stage=args.train_stage, train_track=args.train_track,
                  gap_width_m=args.gap_width, episode_len=args.episode_len,
                  decoder_path=args.decoder_path, command_speed=args.command_speed,
                  command_ramp=command_ramp)
    if getattr(args, 'reward_profile', 'legacy') != 'legacy':
        config['reward_profile'] = args.reward_profile
    if getattr(args, 'material_start_profile', 'default') != 'default':
        config['material_start_profile'] = args.material_start_profile
    if getattr(args, 'material_aux_limit', 0.2) != 0.2:
        config['material_aux_limit'] = args.material_aux_limit
    return config


def _sha256(path):
    if path is None:
        return None
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_training_run_config(args, out_dir):
    """Write immutable, independently checkable training provenance."""
    sources = ("stage3_rl/env.py", "stage3_rl/ppo_train.py", "stage3_rl/stair_reward.py", "stage3_rl/tire_reward.py",
               "stage3_rl/curriculum_gate.py", "common/training_terrain.py",
               "common/eval_protocol.py",
               "common/terrain.py", "common/material_track.py", "stage1_to/trajectory_opt.py",
               "stage2_tvae/train_tvae.py", "stage2_tvae/export_decoders.py",
               "REPORT.md")
    config = dict(train_stage=args.train_stage,
                  worker_stages=[worker_stage_for_index(args.train_stage, i, args.workers)
                                 for i in range(args.workers)],
                  allocation=("2 gap : 2 rough_slope_retain"
                              if args.train_stage == "gap_rough_retain" else None),
                  train_track=args.train_track,
                  train_track_sha256=_sha256(args.train_track),
                  material_aux_limit=(getattr(args, 'material_aux_limit', 0.2) if args.train_track and
                                      args.train_track.lower().endswith('.xml') else None),
                  reward_profile=getattr(args, 'reward_profile', 'legacy'),
                  material_start_profile=getattr(args, 'material_start_profile', 'default'),
                  material_worker_start_profiles=[material_start_for_worker(
                      getattr(args, 'material_start_profile', 'default'), i, args.workers)
                      for i in range(args.workers)],
                  gap_width_m=args.gap_width, episode_len=args.episode_len,
                  command_speed=args.command_speed,
                  command_ramp=(None if args.command_ramp_start is None else
                                [args.command_ramp_start, args.command_ramp_end,
                                 args.command_ramp_steps]),
                  init=args.init, init_sha256=_sha256(args.init),
                  decoder_path=os.path.abspath(args.decoder_path),
                  decoder_sha256=_sha256(args.decoder_path),
                  probe_track=args.probe_track,
                  probe_track_sha256=_sha256(args.probe_track),
                  probe_episodes=getattr(args, 'probe_episodes', 5),
                  probe_max_steps=getattr(args, 'probe_max_steps', 11000),
                  eval_protocol=args.eval_protocol, iters=args.iters,
                  workers=args.workers, epw=args.epw, seed=args.seed,
                  epochs=args.epochs, batch_target=args.batch_target,
                  freeze_norm=getattr(args, 'freeze_norm', False),
                  bc_coef=getattr(args, 'bc_coef', 0.0),
                  learning_rate=getattr(args, 'lr', 3e-4),
                  critic_warmup=getattr(args, 'critic_warmup', 0),
                  latent_std=getattr(args, 'latent_std', None),
                  aux_std=getattr(args, 'aux_std', None),
                  gait_std=getattr(args, 'gait_std', None),
                  wheel_std=getattr(args, 'wheel_std', None),
                  residual_actor=getattr(args, 'residual_actor', False),
                  residual_v2=getattr(args, 'residual_v2', False),
                  residual_v3=getattr(args, 'residual_v3', False),
                  residual_v4=getattr(args, 'residual_v4', False),
                  material_command_profile=getattr(args, 'material_command_profile', 'instant_v1'),
                  source_sha256={name: _sha256(os.path.join(ROOT, name))
                                 for name in sources})
    path = os.path.join(out_dir, "run_config.json")
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
    return path


def parse_material_aux_limit(value):
    if str(value).lower() == 'none':
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('must be none or a number in (0,8]') from exc
    if not np.isfinite(result) or not 0 < result <= 8:
        raise argparse.ArgumentTypeError('must be none or a number in (0,8]')
    return result


def positive_int(value):
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class ExteroEncoder(nn.Module):
    def __init__(self, split_frozen=False, passthrough_raw=False):
        super().__init__()
        self.split_frozen = bool(split_frozen)
        self.passthrough_raw = bool(passthrough_raw)
        self.net = mlp([EXTERO, 128, 64, 32], nn.ReLU, nn.Tanh())
        if self.split_frozen:
            self.res_net = mlp([EXTERO, 256, 128, 64], nn.ReLU, nn.Tanh())

    def forward(self, x):
        if self.split_frozen and self.passthrough_raw:
            return torch.cat([self.net(x), self.res_net(x), x], dim=-1)
        if self.split_frozen:
            return torch.cat([self.net(x), self.res_net(x)], dim=-1)
        return self.net(x)


class TemporalResidualActor(nn.Module):
    """Frozen base locomotion policy + 8-frame temporal-context maneuver specialist."""
    HIST_K = 8

    def __init__(self, hidden=768):
        super().__init__()
        self.hidden = int(hidden)
        self.base_body = mlp([PROPRIO + 32 + GAIT, 512, 256, 128], nn.ReLU)
        self.base_head = nn.Linear(128, ACT_DIM)
        ctx_in_dim = self.HIST_K * (PROPRIO + 64 + GAIT)
        self.ctx_proj = nn.Sequential(
            nn.Linear(ctx_in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 384), nn.GELU(),
        )
        self.res_head = nn.Linear(384, ACT_DIM)
        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)
        self.gate_net = mlp([ctx_in_dim, 256, 64, 1], nn.GELU)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, -6.0)
        self.log_std = nn.Parameter(torch.cat([
            torch.zeros(LATENT), torch.full((AUXD,), float(np.log(0.5))),
            torch.full((1,), float(np.log(2.0)))]))
        self._ring = None

    def reset_context(self):
        self._ring = None

    def _split_single(self, obs_step):
        proprio = obs_step[:, :PROPRIO]
        ext_base = obs_step[:, PROPRIO:PROPRIO + 32]
        ext_res = obs_step[:, PROPRIO + 32:PROPRIO + 96]
        gait = obs_step[:, PROPRIO + 96:]
        base_in = torch.cat([proprio, ext_base, gait], dim=-1)
        res_in = torch.cat([proprio, ext_res, gait], dim=-1)
        return base_in, res_in

    def forward_window(self, base_in, ctx_flat):
        base_mu = self.base_head(self.base_body(base_in))
        raw_gate = torch.sigmoid(self.gate_net(ctx_flat))
        gate = torch.clamp((raw_gate - 0.50) / 0.35, 0.0, 1.0)
        res_mu = self.res_head(self.ctx_proj(ctx_flat))
        return base_mu, gate, res_mu, raw_gate

    def dist(self, obs):
        base_in, res_in = self._split_single(obs)
        B = res_in.shape[0]
        # Detect episode reset when initial action history & gait_out_hist are at reset values
        if (self._ring is None or self._ring.shape[0] != B
                or self._ring.device != res_in.device):
            self._ring = res_in.unsqueeze(1).repeat(1, self.HIST_K, 1).detach()
        else:
            # Auto-reset per batch item if a_aux & a_latent in obs are exact zeros (fresh reset!)
            is_fresh = (obs[:, 93:117].abs().sum(dim=-1) < 1e-7) & (obs[:, -34:-2].abs().sum(dim=-1) < 1e-7)
            rolled = torch.cat([self._ring[:, 1:, :], res_in.unsqueeze(1)], dim=1)
            if is_fresh.any():
                fresh_stack = res_in.unsqueeze(1).repeat(1, self.HIST_K, 1)
                rolled = torch.where(is_fresh[:, None, None], fresh_stack, rolled)
            self._ring = rolled.detach()
        ctx_flat = self._ring.reshape(B, -1)
        base_mu, gate, res_mu, _ = self.forward_window(base_in, ctx_flat)
        mu = base_mu + gate * res_mu
        base_std = self.log_std.exp().expand_as(mu)
        eff_std = (0.04 + 0.96 * gate.detach()) * base_std
        return Normal(mu, eff_std)


class TemporalWheelResidualActor(nn.Module):
    """Stateful recurrent + 16-frame temporal-context + causal navigation/phase tracker 30-action policy."""
    HIST_K = 16
    KIN_DIM = 48

    def __init__(self, hidden=768, gru_hidden=256, raw_norm_mean=None, raw_norm_std=None, extero_passthrough=False):
        super().__init__()
        self.hidden = int(hidden)
        self.gru_hidden = int(gru_hidden)
        self.extero_passthrough = bool(extero_passthrough)
        self.base_body = mlp([PROPRIO + 32 + GAIT, 512, 256, 128], nn.ReLU)
        self.base_head = nn.Linear(128, ACT_DIM)
        step_dim = PROPRIO + 64 + GAIT + self.KIN_DIM
        ctx_in_dim = self.HIST_K * step_dim
        self.gru = nn.GRU(step_dim, self.gru_hidden, batch_first=True)
        self.ctx_proj = nn.Sequential(
            nn.Linear(ctx_in_dim + self.gru_hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 384), nn.GELU(),
        )
        self.res_head = nn.Linear(384, ACT_DIM)
        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)
        self.wheel_head = nn.Linear(384, 1)
        nn.init.zeros_(self.wheel_head.weight)
        nn.init.zeros_(self.wheel_head.bias)
        self.gate_net = mlp([ctx_in_dim + self.gru_hidden, 256, 64, 1], nn.GELU)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, -2.0)
        self.log_std = nn.Parameter(torch.cat([
            torch.zeros(LATENT), torch.full((AUXD,), float(np.log(0.5))),
            torch.full((1,), float(np.log(2.0)))]))
        self.wheel_log_std = nn.Parameter(torch.tensor(float(np.log(0.5))))
        nm426 = np.zeros(426, dtype=np.float32) if raw_norm_mean is None else np.asarray(raw_norm_mean, dtype=np.float32)
        ns426 = np.ones(426, dtype=np.float32) if raw_norm_std is None else np.asarray(raw_norm_std, dtype=np.float32)
        self.register_buffer("p_mean", torch.as_tensor(nm426[:33], dtype=torch.float32))
        self.register_buffer("p_std", torch.as_tensor(ns426[:33], dtype=torch.float32))
        self.register_buffer("e_mean", torch.as_tensor(nm426[PROPRIO:PROPRIO+EXTERO], dtype=torch.float32))
        self.register_buffer("e_std", torch.as_tensor(ns426[PROPRIO:PROPRIO+EXTERO], dtype=torch.float32))
        self.register_buffer("grid_dx", torch.as_tensor(
            np.concatenate([np.arange(-0.8, -0.1, 0.1), np.arange(0.2, 1.51, 0.1)]), dtype=torch.float32))
        self.reset_context()

    def reset_context(self):
        self._ring = None
        self._h = None
        self._x_int = None
        self._yaw_int = None
        self._ey_int = None
        self._prev_vcmd = None
        self._step_cnt = None
        self._edge_mem = None
        self._nav_state = None

    def _compute_kin_step(self, proprio_norm, extero_norm=None):
        """Causal single-step update of 48-D localized kinematic & maneuver-progress features!"""
        B = proprio_norm.shape[0]
        dev = proprio_norm.device
        raw33 = proprio_norm[:, :33] * self.p_std.to(dev) + self.p_mean.to(dev)
        v_b = raw33[:, 0:3]
        w_b = raw33[:, 3:6]
        vx_cmd = raw33[:, 15]
        vy_cmd = raw33[:, 16]
        wyaw_cmd = raw33[:, 17]
        grav = raw33[:, 18:21]
        q_jnt = raw33[:, 21:33]

        roll = torch.atan2(-grav[:, 1], -grav[:, 2])
        pitch = -torch.asin(torch.clamp(-grav[:, 0], -1.0, 1.0))
        wz_world = (-torch.sin(pitch) * w_b[:, 0]
                    + torch.cos(pitch) * torch.sin(roll) * w_b[:, 1]
                    + torch.cos(pitch) * torch.cos(roll) * w_b[:, 2])

        if self._x_int is None or self._x_int.shape[0] != B or self._x_int.device != dev:
            self._x_int = torch.full((B,), 0.50, device=dev, dtype=torch.float32)
            self._ey_int = -vy_cmd.clone()
            self._yaw_int = -(wyaw_cmd + 2.0 * self._ey_int) / 1.5
            self._prev_vcmd = vx_cmd.clone()
            self._step_cnt = torch.zeros((B,), device=dev, dtype=torch.float32)
            self._edge_mem = [{} for _ in range(B)]
            self._nav_state = [{
                "stg": 0, "st_loc": 0, "sub_al": 0, "set_cnt": 0,
                "p_idx": 0, "p_mode": 0, "hop_el": 0, "t_trig": -1, "r_trig": -1, "rb_trig": -1, "rb_rtrig": -1
            } for _ in range(B)]
            is_step0 = True
        else:
            is_step0 = False

        unsat_y = vy_cmd.abs() < 0.495
        unsat_yaw = unsat_y & (wyaw_cmd.abs() < 0.795)
        ey_new = torch.where(unsat_y, -vy_cmd, self._ey_int)
        yaw_cand = -(wyaw_cmd + 2.0 * ey_new) / 1.5
        yaw_int_step = self._yaw_int + (0.0 if is_step0 else 1.0) * wz_world * 0.01
        yaw_new = torch.where(unsat_yaw, yaw_cand, yaw_int_step)

        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw_new), torch.sin(yaw_new)
        vx = (cy * cp) * v_b[:, 0] + (cy * sp * sr - sy * cr) * v_b[:, 1] + (cy * sp * cr + sy * sr) * v_b[:, 2]
        vy = (sy * cp) * v_b[:, 0] + (sy * sp * sr + cy * cr) * v_b[:, 1] + (sy * sp * cr - cy * sr) * v_b[:, 2]

        dist_riser = torch.full((B,), 2.0, device=dev, dtype=torch.float32)
        dist_drop = torch.full((B,), 2.0, device=dev, dtype=torch.float32)
        ext_np = None
        if extero_norm is not None:
            ext_np = (extero_norm * self.e_std.to(dev) + self.e_mean.to(dev)).detach().cpu().numpy()

        if not is_step0:
            ey_sat = self._ey_int + vy * 0.01
            ey_sat = torch.where(vy_cmd <= -0.495, torch.clamp(ey_sat, min=0.495), ey_sat)
            ey_sat = torch.where(vy_cmd >= 0.495, torch.clamp(ey_sat, max=-0.495), ey_sat)
            ey_new = torch.where(unsat_y, -vy_cmd, ey_sat)
            x_next = self._x_int + vx * 0.01
            crossed_22 = (self._prev_vcmd > 1.2) & (vx_cmd < 1.0) & (x_next >= 20.5) & (x_next <= 23.5)
            x_next = torch.where(crossed_22, 22.000 + 0.5 * vx * 0.01, x_next)

            if ext_np is not None:
                gdx = self.grid_dx.cpu().numpy()
                for bi in range(B):
                    xp = float(x_next[bi].item())
                    vxi = float(vx[bi].item())
                    yawi = float(yaw_new[bi].item())
                    eyi = float(ey_new[bi].item())
                    cyi = max(0.90, float(np.cos(yawi)))
                    ext_b = ext_np[bi]
                    grid = ext_b[:210].reshape(10, 21)
                    h_grid = grid - float(np.min(ext_b))
                    center_a = 0.5 * (h_grid[4, 7:] + h_grid[5, 7:])
                    d_up = np.diff(center_a)
                    rs = np.flatnonzero(d_up > 0.08)
                    dr = np.flatnonzero(d_up < -0.08)
                    if len(rs) > 0:
                        dist_riser[bi] = min(2.0, float((gdx[7 + int(rs[0])] + 0.05) * cyi))
                    if len(dr) > 0:
                        dist_drop[bi] = min(2.0, float((gdx[7 + int(dr[0])] + 0.05) * cyi))
                    em = self._edge_mem[bi]
                    if 29.3 <= xp <= 30.85 and abs(eyi) < 0.25:
                        sc = np.flatnonzero(h_grid[3:7, 7:].max(axis=0) > 0.095)
                        cc = int(sc[0]) if len(sc) > 0 else None
                        pc = em.get("su")
                        if cc is not None and pc is not None and cc == pc - 1:
                            xp = 31.000 - float(gdx[7 + cc]) * cyi + 0.5 * vxi * 0.01
                        em["su"] = cc
                    if 31.02 <= xp <= 32.35 and abs(eyi) < 0.15 and abs(vxi) < 0.08 and len(rs) > 0:
                        r0 = int(rs[0])
                        dx_m = (float(gdx[7 + r0]) + 0.05) * cyi
                        kn = int(np.clip(round((xp + dx_m - 31.00) / 0.32), 1, 4))
                        xr = 31.00 + 0.32 * kn
                        xp = float(np.clip(xp, xr - float(gdx[7 + r0 + 1]) * cyi, xr - float(gdx[7 + r0]) * cyi))
                    if 33.5 <= xp <= 34.95:
                        bc = np.flatnonzero(np.maximum(h_grid[0, 7:], h_grid[9, 7:]) > 0.55)
                        cc = int(bc[0]) if len(bc) > 0 else None
                        pc = em.get("bar")
                        if cc is not None and pc is not None and cc == pc - 1:
                            xp = 35.160 - float(gdx[7 + cc]) * cyi + 0.5 * vxi * 0.01
                        em["bar"] = cc
                    if 37.1 <= xp <= 38.60:
                        dc = np.flatnonzero(h_grid[3:7, 7:].max(axis=0) > 0.08)
                        cc = int(dc[0]) if len(dc) > 0 else None
                        pc = em.get("dr")
                        if cc is not None and pc is not None and cc == pc - 1:
                            xp = 38.820 - float(gdx[7 + cc]) * cyi + 0.5 * vxi * 0.01
                        em["dr"] = cc
                    if 40.28 <= xp <= 40.95 and abs(eyi) < 0.20 and abs(vxi) < 0.08 and len(dr) > 0:
                        d0 = int(dr[0])
                        xlo = 41.370 - float(gdx[7 + d0 + 1]) * cyi
                        xhi = 41.370 - float(gdx[7 + d0]) * cyi
                        if abs(0.5 * (xlo + xhi) - xp) < 0.25:
                            xp = float(np.clip(xp, xlo, xhi))
                    if 43.3 <= xp <= 44.75:
                        fc = np.flatnonzero(h_grid[3:7, 7:].max(axis=0) > 0.20)
                        cc = int(fc[0]) if len(fc) > 0 else None
                        pc = em.get("fr")
                        if cc is not None and pc is not None and cc == pc - 1:
                            xp = 44.980 - float(gdx[7 + cc]) * cyi + 0.5 * vxi * 0.01
                        em["fr"] = cc
                    x_next[bi] = xp

            self._x_int = x_next.detach()
            self._step_cnt = (self._step_cnt + 1.0).detach()

        self._ey_int = ey_new.detach()
        self._yaw_int = yaw_new.detach()
        self._prev_vcmd = vx_cmd.detach()

        # Update causal navigation FSM descriptors (purely from obs-reconstructed state!)
        nav_feats = np.zeros((B, 22), dtype=np.float32)
        q_np = q_jnt.detach().cpu().numpy()
        for bi in range(B):
            ns = self._nav_state[bi]
            xb = float(self._x_int[bi].item())
            eyb = float(ey_new[bi].item())
            ywb = float(yaw_new[bi].item())
            ptb = float(pitch[bi].item())
            rlb = float(roll[bi].item())
            st_i = int(self._step_cnt[bi].item())
            cp_b, sp_b = float(np.cos(ptb)), float(np.sin(ptb))
            qb = q_np[bi]
            wx_arr = [xb + cp_b*(xh - 0.213*np.sin(qb[li*3+1]) - 0.2264*np.sin(qb[li*3+1]+qb[li*3+2]))
                      + sp_b*(-0.213*np.cos(qb[li*3+1]) - 0.2264*np.cos(qb[li*3+1]+qb[li*3+2]))
                      for li, xh in enumerate((+0.1934, +0.1934, -0.1934, -0.1934))]
            min_wx = min(wx_arr)
            x_rl = 0.5 * (wx_arr[2] + wx_arr[3])

            # Stage transitions: 0:app, 1:tires, 2:t_dism, 3:b_shift, 4:b_trot, 5:st_up, 6:pre_rb, 7:rb, 8:b_roll, 9:b_up, 10:pre_sd, 11:sd, 12:post
            sg = ns["stg"]
            if sg == 0 and xb >= 23.50:
                sg = 1; ns["st_loc"] = 0
            elif sg == 1 and xb >= 26.82:
                sg = 2; ns["st_loc"] = 0
            elif sg == 2 and (xb >= 27.40 or ns["st_loc"] >= 65):
                sg = 3; ns["st_loc"] = 0
            elif sg == 3 and ((abs(eyb - 0.72) <= 0.10 and abs(ywb) <= 0.06) or xb >= 29.55):
                sg = 4; ns["st_loc"] = 0
            elif sg == 4 and xb >= 29.78:
                sg = 5; ns["st_loc"] = 0; ns["sub_al"] = 0; ns["set_cnt"] = 0; ns["p_idx"] = 0; ns["p_mode"] = 0; ns["hop_el"] = 0
            elif sg == 5 and ns["p_idx"] >= 6:
                sg = 6; ns["st_loc"] = 0; ns["set_cnt"] = 0; ns["sub_al"] = 0
            elif sg == 6 and ns["set_cnt"] >= 20:
                sg = 7; ns["st_loc"] = 0; ns["set_cnt"] = 0; ns["sub_al"] = 0
            elif sg == 7 and xb >= 36.40:
                sg = 8; ns["st_loc"] = 0
            elif sg == 8 and xb >= 37.85:
                sg = 9; ns["st_loc"] = 0
            elif sg == 9 and xb >= 40.50:
                sg = 10; ns["st_loc"] = 0; ns["set_cnt"] = 0; ns["sub_al"] = 0
            elif sg == 10 and ns["set_cnt"] >= 20:
                sg = 11; ns["st_loc"] = 0; ns["set_cnt"] = 0; ns["sub_al"] = 0
            elif sg == 11 and xb >= 43.35:
                sg = 12; ns["st_loc"] = 0
            ns["stg"] = sg

            # Encode stage one-hot (13) + internal timers/triggers (9) = 22 dims!
            nav_feats[bi, sg] = 1.0
            nav_feats[bi, 13] = min(3.0, ns["st_loc"] / 30.0)
            nav_feats[bi, 14] = float(ns["sub_al"])
            nav_feats[bi, 15] = ns["set_cnt"] / 20.0
            nav_feats[bi, 16] = ns["p_idx"] / 5.0
            nav_feats[bi, 17] = float(ns["p_mode"])
            nav_feats[bi, 18] = ns["hop_el"] / 30.0
            nav_feats[bi, 19] = float(min_wx <= 38.28)

            # Advance internal counters for next step identically to closed-loop navigation logic!
            if sg == 1:
                if ns["t_trig"] < 0 and xb >= 24.134:
                    ns["t_trig"] = ns["st_loc"]
                if ns["t_trig"] >= 0:
                    nav_feats[bi, 20] = min(3.0, (ns["st_loc"] - ns["t_trig"]) / 20.0)
                    if ns["st_loc"] - ns["t_trig"] >= 58 and ext_np is not None:
                        bz_fl = -float(np.min(ext_np[bi]))
                        crk, srk = float(np.cos(rlb)), float(np.sin(rlb))
                        z_rls = [bz_fl - sp_b*(-0.1934 - 0.213*np.sin(qb[li*3+1]) - 0.2264*np.sin(qb[li*3+1]+qb[li*3+2]))
                                 + cp_b*srk*yo + cp_b*crk*(-0.213*np.cos(qb[li*3+1]) - 0.2264*np.cos(qb[li*3+1]+qb[li*3+2]))
                                 for li, yo in ((2, +0.142), (3, -0.142))]
                        if ns["r_trig"] < 0 and x_rl >= 24.45 and 0.5*(z_rls[0]+z_rls[1]) < 0.19:
                            ns["r_trig"] = ns["st_loc"]
                    if ns["r_trig"] >= 0:
                        nav_feats[bi, 21] = min(3.0, (ns["st_loc"] - ns["r_trig"]) / 15.0)
                ns["st_loc"] += 1
            elif sg in (2, 3, 4, 12):
                ns["st_loc"] += 1
            elif sg == 5:
                perch_x = (30.52, 30.78, 31.10, 31.42, 31.74, 32.06)[min(ns["p_idx"], 5)]
                on_st = ns["p_idx"] > 0
                ph4 = 2.0 * np.pi * 4.0 * (st_i * 0.01)
                if ns["p_mode"] == 0:
                    if on_st:
                        if ns["set_cnt"] == 0 and (abs(eyb) > 0.014 or abs(ywb) > 0.018 or (ns["sub_al"] == 0 and abs(np.sin(ph4)) >= 0.28)):
                            ns["sub_al"] = 0
                        elif ns["set_cnt"] == 0 and xb < perch_x - 0.006:
                            ns["sub_al"] = 1
                        else:
                            ns["set_cnt"] += 1
                            if ns["set_cnt"] >= 20:
                                ns["p_mode"] = 1; ns["hop_el"] = 0; ns["set_cnt"] = 0; ns["sub_al"] = 0
                    else:
                        if ns["sub_al"] == 0:
                            if not (abs(eyb) > 0.010 or abs(ywb) > 0.012 or xb > 30.52 or xb < 30.22 or abs(np.sin(ph4)) >= 0.28):
                                ns["sub_al"] = 1
                            elif (xb >= 29.95 and abs(eyb) <= 0.10 and abs(ywb) <= 0.15
                                  and abs(np.sin(ph4)) < 0.40):
                                ns["sub_al"] = 1
                        elif not (ns["set_cnt"] == 0 and xb < perch_x - 0.35
                                    and float(dist_riser[bi]) > 0.22):
                            ns["set_cnt"] += 1
                            if ns["set_cnt"] >= 22:
                                ns["p_mode"] = 1; ns["hop_el"] = 0; ns["set_cnt"] = 0; ns["sub_al"] = 0
                else:
                    el = ns["hop_el"]
                    ns["hop_el"] += 1
                    h_dur = (5 + 12 + 13 + 18) if ns["p_idx"] == 0 else ((6 + 19 + 13 + 18) if ns["p_idx"] < 5 else (19 + 22 + 22 + 18))
                    if el >= h_dur - 1:
                        if xb >= 32.62: ns["p_idx"] = 6
                        elif xb >= 32.06: ns["p_idx"] = 5
                        elif xb >= 31.74: ns["p_idx"] = 4
                        elif xb >= 31.42: ns["p_idx"] = 3
                        elif xb >= 31.10: ns["p_idx"] = 2
                        elif xb >= 30.80: ns["p_idx"] = 1
                        else: ns["p_idx"] = 0
                        ns["p_mode"] = 0; ns["set_cnt"] = 0; ns["sub_al"] = 0
                ns["st_loc"] += 1
            elif sg == 6:
                ph35 = 2.0 * np.pi * 3.5 * (st_i * 0.01)
                if ns["sub_al"] == 0:
                    if not (abs(eyb) > 0.012 or abs(ywb) > 0.014 or xb > 33.04 or xb < 32.88 or abs(np.sin(ph35)) >= 0.28):
                        ns["sub_al"] = 1
                elif not (ns["set_cnt"] == 0 and xb < 33.150):
                    ns["set_cnt"] += 1
                ns["st_loc"] += 1
            elif sg == 7:
                if ns["rb_trig"] < 0 and xb >= 34.575:
                    ns["rb_trig"] = ns["st_loc"]
                if ns["rb_trig"] >= 0:
                    nav_feats[bi, 20] = min(3.0, (ns["st_loc"] - ns["rb_trig"]) / 20.0)
                    if ns["st_loc"] - ns["rb_trig"] >= 47:
                        if ns["rb_rtrig"] < 0 and x_rl >= 35.022 and ptb < -0.08:
                            ns["rb_rtrig"] = ns["st_loc"]
                        if ns["rb_rtrig"] >= 0:
                            nav_feats[bi, 21] = min(3.0, (ns["st_loc"] - ns["rb_rtrig"]) / 15.0)
                ns["st_loc"] += 1
            elif sg == 9:
                # Periodic hop phase for beam_upramp (period = 6+9+6+16 = 37)!
                nav_feats[bi, 20] = (ns["st_loc"] % 37) / 37.0
                ns["st_loc"] += 1
            elif sg == 10:
                ph35 = 2.0 * np.pi * 3.5 * (st_i * 0.01)
                if ns["sub_al"] == 0:
                    if not (abs(eyb) > 0.014 or abs(ywb) > 0.018 or xb > 40.88 or xb < 40.35 or abs(np.sin(ph35)) >= 0.28):
                        ns["sub_al"] = 1
                elif not (ns["set_cnt"] == 0 and xb < 40.700):
                    ns["set_cnt"] += 1
                ns["st_loc"] += 1
            elif sg == 11:
                # Periodic hop phase for stairs_down (period = 22+6+7+18 = 53)!
                nav_feats[bi, 20] = (ns["st_loc"] % 53) / 53.0
                ns["st_loc"] += 1

        t_sec = self._step_cnt * 0.01
        x_e = self._x_int
        kin_base = torch.stack([
            (x_e - 25.0) / 15.0,
            torch.clamp((x_e - 24.13) * 5.0, -3.0, 3.0),
            torch.clamp((x_e - 26.82) * 5.0, -3.0, 3.0),
            torch.clamp((x_e - 27.40) * 4.0, -3.0, 3.0),
            torch.clamp((x_e - 29.78) * 4.0, -3.0, 3.0),
            torch.clamp((x_e - 30.65) * 6.0, -3.0, 3.0),
            torch.clamp((x_e - 32.18) * 6.0, -3.0, 3.0),
            torch.clamp((x_e - 33.17) * 5.0, -3.0, 3.0),
            torch.clamp((x_e - 34.56) * 5.0, -3.0, 3.0),
            torch.clamp((x_e - 37.85) * 4.0, -3.0, 3.0),
            torch.clamp((x_e - 40.70) * 5.0, -3.0, 3.0),
            torch.clamp((x_e - 43.35) * 4.0, -3.0, 3.0),
            torch.clamp((x_e - 45.00) * 3.0, -3.0, 3.0),
            dist_riser - 1.0,
            dist_drop - 1.0,
            torch.sin(x_e * 8.0), torch.cos(x_e * 8.0),
            ey_new, yaw_new, vx / 2.0, vy,
            torch.sin(2.0 * np.pi * 3.5 * t_sec), torch.cos(2.0 * np.pi * 3.5 * t_sec),
            torch.sin(2.0 * np.pi * 3.6 * t_sec), torch.cos(2.0 * np.pi * 3.6 * t_sec),
            torch.sin(2.0 * np.pi * 4.0 * t_sec),
        ], dim=-1)  # 26 dims + 22 nav_feats = 48 dims!
        return torch.cat([kin_base, torch.from_numpy(nav_feats).to(dev)], dim=-1)

    def _split_single(self, obs_step):
        proprio = obs_step[:, :PROPRIO]
        ext_base = obs_step[:, PROPRIO:PROPRIO + 32]
        ext_res = obs_step[:, PROPRIO + 32:PROPRIO + 96]
        if self.extero_passthrough:
            ext_raw_norm = obs_step[:, PROPRIO + 96:PROPRIO + 96 + EXTERO]
            gait = obs_step[:, PROPRIO + 96 + EXTERO:]
        else:
            ext_raw_norm = None
            gait = obs_step[:, PROPRIO + 96:]
        kin = self._compute_kin_step(proprio, ext_raw_norm)
        base_in = torch.cat([proprio, ext_base, gait], dim=-1)
        res_in = torch.cat([proprio, ext_res, gait, kin], dim=-1)
        return base_in, res_in

    def forward_seq(self, base_in_seq, res_in_seq, ring_seq, h0=None):
        """Batch sequence forward for fast GPU training!"""
        gru_out, hn = self.gru(res_in_seq, h0)
        feat_in = torch.cat([ring_seq, gru_out], dim=-1)
        base_mu = self.base_head(self.base_body(base_in_seq))
        bounded_base = torch.cat([
            base_mu[..., :LATENT],
            base_mu[..., LATENT:LATENT + AUXD].clamp(-0.2, 0.2),
            base_mu[..., LATENT + AUXD:]
        ], dim=-1)
        gate = torch.sigmoid(self.gate_net(feat_in))
        h_feat = self.ctx_proj(feat_in)
        res_mu = self.res_head(h_feat)
        wheel_mu = gate * self.wheel_head(h_feat)
        mu29 = bounded_base + gate * res_mu
        return torch.cat([mu29, wheel_mu], dim=-1), gate, hn

    def dist(self, obs):
        base_in, res_in = self._split_single(obs)
        B = res_in.shape[0]
        if (self._ring is None or self._ring.shape[0] != B
                or self._ring.device != res_in.device):
            self._ring = res_in.unsqueeze(1).repeat(1, self.HIST_K, 1).detach()
            self._h = torch.zeros(1, B, self.gru_hidden, device=res_in.device, dtype=res_in.dtype)
        else:
            self._ring = torch.cat([self._ring[:, 1:, :], res_in.unsqueeze(1)], dim=1).detach()
        gru_out, h_next = self.gru(res_in.unsqueeze(1), self._h)
        self._h = h_next.detach()
        ctx_flat = torch.cat([self._ring.reshape(B, -1), gru_out.squeeze(1)], dim=-1)
        base_mu = self.base_head(self.base_body(base_in))
        bounded_base = torch.cat([
            base_mu[:, :LATENT],
            base_mu[:, LATENT:LATENT + AUXD].clamp(-0.2, 0.2),
            base_mu[:, LATENT + AUXD:]
        ], dim=-1)
        gate = torch.sigmoid(self.gate_net(ctx_flat))
        h_feat = self.ctx_proj(ctx_flat)
        res_mu = self.res_head(h_feat)
        wheel_mu = gate * self.wheel_head(h_feat)
        mu29 = bounded_base + gate * res_mu
        mu = torch.cat([mu29, wheel_mu], dim=-1)
        base_std = self.log_std.exp().expand_as(mu29)
        eff_std = (0.04 + 0.96 * gate.detach()) * base_std
        wheel_std = (0.04 + 0.96 * gate.detach()) * self.wheel_log_std.exp()
        return Normal(mu, torch.cat([eff_std, wheel_std], dim=-1))


class FeedforwardKinActor(TemporalWheelResidualActor):
    """Zero-latency feed-forward variant: identical causal kinematics, no recurrent smoothing (impulse-faithful)."""
    HIST_K = 1

    def __init__(self, hidden=768, gru_hidden=256, raw_norm_mean=None, raw_norm_std=None, extero_passthrough=False):
        super().__init__(hidden=hidden, gru_hidden=gru_hidden, raw_norm_mean=raw_norm_mean,
                         raw_norm_std=raw_norm_std, extero_passthrough=extero_passthrough)
        step_dim = PROPRIO + 64 + GAIT + self.KIN_DIM
        self.gru = nn.Identity()
        self.ff_proj = nn.Sequential(nn.Linear(step_dim, 512), nn.GELU(),
                                     nn.Linear(512, self.gru_hidden), nn.GELU())

    def forward_seq(self, base_in_seq, res_in_seq, ring_seq, h0=None):
        feat_in = torch.cat([ring_seq, self.ff_proj(res_in_seq)], dim=-1)
        base_mu = self.base_head(self.base_body(base_in_seq))
        bounded_base = torch.cat([
            base_mu[..., :LATENT],
            base_mu[..., LATENT:LATENT + AUXD].clamp(-0.2, 0.2),
            base_mu[..., LATENT + AUXD:]
        ], dim=-1)
        gate = torch.sigmoid(self.gate_net(feat_in))
        h_feat = self.ctx_proj(feat_in)
        res_mu = self.res_head(h_feat)
        wheel_mu = gate * self.wheel_head(h_feat)
        mu29 = bounded_base + gate * res_mu
        return torch.cat([mu29, wheel_mu], dim=-1), gate, None

    def dist(self, obs):
        base_in, res_in = self._split_single(obs)
        feat_in = torch.cat([res_in, self.ff_proj(res_in)], dim=-1)
        base_mu = self.base_head(self.base_body(base_in))
        bounded_base = torch.cat([
            base_mu[:, :LATENT],
            base_mu[:, LATENT:LATENT + AUXD].clamp(-0.2, 0.2),
            base_mu[:, LATENT + AUXD:]
        ], dim=-1)
        gate = torch.sigmoid(self.gate_net(feat_in))
        h_feat = self.ctx_proj(feat_in)
        res_mu = self.res_head(h_feat)
        wheel_mu = gate * self.wheel_head(h_feat)
        mu29 = bounded_base + gate * res_mu
        mu = torch.cat([mu29, wheel_mu], dim=-1)
        base_std = self.log_std.exp().expand_as(mu29)
        eff_std = (0.04 + 0.96 * gate.detach()) * base_std
        wheel_std = (0.04 + 0.96 * gate.detach()) * self.wheel_log_std.exp()
        return Normal(mu, torch.cat([eff_std, wheel_std], dim=-1))


class ResidualActor(nn.Module):
    """Frozen base locomotion policy + terrain-gated maneuver specialist."""
    def __init__(self, hidden=512, split_extero=True):
        super().__init__()
        self.split_extero = bool(split_extero)
        res_ext_dim = 64 if split_extero else 32
        self.base_body = mlp([PROPRIO + 32 + GAIT, 512, 256, 128], nn.ReLU)
        self.base_head = nn.Linear(128, ACT_DIM)
        self.res_body = mlp([PROPRIO + res_ext_dim + GAIT, hidden, hidden, 256], nn.ReLU)
        self.res_head = nn.Linear(256, ACT_DIM)
        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)
        self.gate_net = mlp([PROPRIO + res_ext_dim + GAIT, 128, 64, 1], nn.ReLU)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, -6.0)
        self.log_std = nn.Parameter(torch.cat([
            torch.zeros(LATENT), torch.full((AUXD,), float(np.log(0.5))),
            torch.full((1,), float(np.log(2.0)))]))

    def forward_parts(self, obs):
        if self.split_extero:
            proprio = obs[:, :PROPRIO]
            ext_base = obs[:, PROPRIO:PROPRIO + 32]
            ext_res = obs[:, PROPRIO + 32:PROPRIO + 96]
            gait = obs[:, PROPRIO + 96:]
            base_in = torch.cat([proprio, ext_base, gait], dim=-1)
            res_in = torch.cat([proprio, ext_res, gait], dim=-1)
        else:
            base_in = res_in = obs
        base_mu = self.base_head(self.base_body(base_in))
        raw_gate = torch.sigmoid(self.gate_net(res_in))
        gate = torch.clamp((raw_gate - 0.50) / 0.35, 0.0, 1.0)
        res_mu = self.res_head(self.res_body(res_in))
        return base_mu, gate, res_mu, raw_gate

    def dist(self, obs):
        base_mu, gate, res_mu, _ = self.forward_parts(obs)
        mu = base_mu + gate * res_mu
        base_std = self.log_std.exp().expand_as(mu)
        eff_std = (0.04 + 0.96 * gate.detach()) * base_std
        return Normal(mu, eff_std)


class TrainableResidualActor(ResidualActor):
    """Versioned soft-gated residual: initially identical mean, nonzero gradient."""
    def __init__(self, hidden=512, split_extero=True):
        super().__init__(hidden=hidden, split_extero=split_extero)
        nn.init.constant_(self.gate_net[-1].bias, -2.0)

    def forward_parts(self, obs):
        base_mu, _, res_mu, raw_gate = super().forward_parts(obs)
        return base_mu, raw_gate, res_mu, raw_gate


class BoundedBaseResidualActor(TrainableResidualActor):
    """Residual control around the historical ±0.2 material base action."""
    def dist(self, obs):
        base_mu, gate, res_mu, _ = self.forward_parts(obs)
        bounded_base = torch.cat([base_mu[:, :LATENT],
                                  base_mu[:, LATENT:LATENT + AUXD].clamp(-0.2, 0.2),
                                  base_mu[:, LATENT + AUXD:]], dim=-1)
        mu = bounded_base + gate * res_mu
        base_std = self.log_std.exp().expand_as(mu)
        return Normal(mu, (0.04 + 0.96 * gate.detach()) * base_std)


class WheelResidualActor(BoundedBaseResidualActor):
    """Versioned 30-action policy with a separate trainable wheel-speed head."""
    def __init__(self, hidden=512, split_extero=True):
        super().__init__(hidden=hidden, split_extero=split_extero)
        self.wheel_head = nn.Linear(256, 1)
        nn.init.zeros_(self.wheel_head.weight)
        nn.init.zeros_(self.wheel_head.bias)
        self.wheel_log_std = nn.Parameter(torch.tensor(float(np.log(0.5))))

    def dist(self, obs):
        base_dist = super().dist(obs)
        _, gate, _, _ = self.forward_parts(obs)
        if self.split_extero:
            res_in = torch.cat([obs[:, :PROPRIO], obs[:, PROPRIO + 32:PROPRIO + 96],
                                obs[:, PROPRIO + 96:]], dim=-1)
        else:
            res_in = obs
        wheel_mean = gate * self.wheel_head(self.res_body(res_in))
        wheel_std = (0.04 + 0.96 * gate.detach()) * self.wheel_log_std.exp()
        return Normal(torch.cat([base_dist.mean, wheel_mean], dim=-1),
                      torch.cat([base_dist.stddev, wheel_std], dim=-1))


def clamp_wheel_log_std(actor):
    if hasattr(actor, 'wheel_log_std'):
        with torch.no_grad():
            actor.wheel_log_std.clamp_(min=float(np.log(0.14)), max=float(np.log(2.0)))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = mlp([PROPRIO + 32 + GAIT, 512, 256, 128], nn.ReLU)
        self.head = nn.Linear(128, ACT_DIM)
        # aux exploration std 0.5 (paper: 1.0) — 0.2-rad aux offsets at std 1
        # thrash the stance before any learning signal stabilizes
        self.log_std = nn.Parameter(torch.cat([
            torch.zeros(LATENT), torch.full((AUXD,), float(np.log(0.5))),
            torch.full((1,), float(np.log(2.0)))]))

    def dist(self, obs):
        mu = self.head(self.body(obs))
        return Normal(mu, self.log_std.exp().expand_as(mu))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp([OBS_WORKER, 512, 256, 128], nn.ReLU)
        self.head = nn.Linear(128, 1)

    def forward(self, obs):
        return self.head(self.net(obs)).squeeze(-1)


class RunningNorm:
    def __init__(self, dim, clip=10.0):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def normalize(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8),
                       -self.clip, self.clip).astype(np.float32)

    def state(self):
        return dict(mean=self.mean.copy(), var=self.var.copy(),
                    count=self.count)

    def load(self, st):
        self.mean = st["mean"].copy()
        self.var = st["var"].copy()
        self.count = st["count"]


def summarize_reward_infos(infos):
    """Aggregate raw environment reward terms; no scaled PPO rewards."""
    totals, terminations = {}, {}
    for info in infos:
        for name, value in info.get('terms', {}).items():
            totals[name] = totals.get(name, 0.0) + float(value)
        reason = info.get('reason')
        if reason:
            terminations[reason] = terminations.get(reason, 0) + 1
    positions = [float(info['_diagnostic_x']) for info in infos if '_diagnostic_x' in info]
    tire_steps = sum(1 for info in infos if 21.4 <= info.get('_diagnostic_start_x', -1) <= 21.6)
    stair_steps = sum(1 for info in infos if 30.1 <= info.get('_diagnostic_start_x', -1) <= 30.3)
    return {'steps': len(infos), 'terms_sum': totals, 'terminations': terminations,
            'max_x': max(positions) if positions else None, 'tire_runup_steps': tire_steps,
            'stair_runup_steps': stair_steps}


# ------------------------------------------------------------------ worker
def worker_main(conn, cfg):
    """Async worker: CPU actor replica + SHM chunk ring (K distinct slots)."""
    import faulthandler, tempfile
    hb_path = os.path.join(tempfile.gettempdir(), f"aptrl_hb_{cfg['seed']}.txt")
    hb = open(hb_path, "w", buffering=1)
    def beat(m):
        hb.write(str(time.time()) + " " + str(m) + "\n")
    try:
        fh_log = open(os.path.join(tempfile.gettempdir(),
                                   f"aptrl_worker_{cfg['seed']}.log"), "w")
        faulthandler.enable(fh_log)
    except Exception:
        pass
    import multiprocessing.shared_memory as sm
    import stage3_rl.env as envmod
    epw = cfg.get("epw", 4)
    envs = [envmod.Go2WEnv(seed=cfg["seed"] * 97 + k,
                           fixed_track=cfg.get("fixed_track"),
                           eval_mode=cfg.get("eval_mode", False),
                           episode_len=cfg["episode_len"],
                           train_stage=cfg.get("train_stage", "mixed"),
                           train_track=cfg.get("train_track"),
                           gap_width_m=cfg.get("gap_width_m"),
                           decoder_path=cfg.get("decoder_path"),
                           command_speed=cfg.get("command_speed"),
                           command_ramp=cfg.get("command_ramp"),
                           reward_profile=cfg.get('reward_profile', 'legacy'),
                           material_start_profile=cfg.get('material_start_profile', 'default'),
                           material_aux_limit=cfg.get('material_aux_limit', 0.2),
                           material_wheel_control=cfg.get('material_wheel_control', 'none'))
            for k in range(epw)]
    use_res = cfg.get("residual_actor", False)
    enc = ExteroEncoder(split_frozen=use_res)
    if use_res:
        actor_cls = {'v1': ResidualActor, 'v2': TrainableResidualActor,
                     'v3': BoundedBaseResidualActor, 'v4': WheelResidualActor}[cfg.get('residual_variant', 'v1')]
        actor = actor_cls(hidden=512, split_extero=True)
    else:
        actor = Actor()
    norm = RunningNorm(OBS_WORKER)
    torch.set_num_threads(1)
    H = CHUNK_H
    n_slot = cfg["n_slot"]
    action_dim = cfg.get('action_dim', ACT_DIM)
    aux_width = action_dim + 3

    # keep SharedMemory handles alive: a temporary object would be GC'd and
    # release its memoryview, invalidating the ndarray buffer
    _shm_obs = sm.SharedMemory(name=cfg["obs_shm"])
    _shm_aux = sm.SharedMemory(name=cfg["aux_shm"])
    obs_arr = np.ndarray((n_slot, H, epw, OBS_WORKER), dtype=np.float32,
                         buffer=_shm_obs.buf)
    aux_arr = np.ndarray((n_slot, H, epw, aux_width), dtype=np.float32,
                         buffer=_shm_aux.buf)

    def load_state(st):
        enc.load_state_dict(st["enc"])
        actor.load_state_dict(st["actor"])
        norm.load(st["norm"])

    reward_log = (open(cfg['reward_log'], 'x', encoding='utf-8')
                  if cfg.get('reward_log') else None)

    def collect_chunk(slot):
        infos = []
        obs_l = np.empty((H, epw, OBS_WORKER), dtype=np.float32)
        aux_l = np.empty((H, epw, aux_width), dtype=np.float32)
        for env in envs:
            apply_material_start_profile(env, cfg.get('material_command_profile', 'instant_v1'))
        obs = np.stack([e._get_obs() for e in envs])
        for h in range(H):
            ob_n = norm.normalize(obs)
            o_t = torch.from_numpy(ob_n)
            with torch.no_grad():
                p = o_t[:, :PROPRIO]
                e_ext = o_t[:, PROPRIO:PROPRIO + EXTERO]
                g = o_t[:, PROPRIO + EXTERO:]
                ao = torch.cat([p, enc(e_ext), g], dim=1)
                dist = actor.dist(ao)
                a = dist.sample()
                logp = dist.log_prob(a).sum(-1)
            a_np = a.numpy()
            obs_l[h] = obs
            rew_h, done_h = [], []
            for k, env in enumerate(envs):
                o2, r, done, info = env.step(a_np[k])
                if reward_log is not None:
                    infos.append({**info, '_diagnostic_x': float(env.data.qpos[0]),
                                  '_diagnostic_start_x': float(env.requested_initial_pose[0])})
                if done:
                    env.report(info["success"])
                    o2 = env.reset()
                rew_h.append(r)
                done_h.append(1.0 if done else 0.0)
            aux_l[h, :, :action_dim] = a_np
            aux_l[h, :, action_dim] = logp.numpy()
            aux_l[h, :, action_dim + 1] = np.asarray(rew_h, dtype=np.float32) * REW_SCALE
            aux_l[h, :, action_dim + 2] = np.asarray(done_h, dtype=np.float32)
            for env in envs:
                apply_material_start_profile(env, cfg.get('material_command_profile', 'instant_v1'))
            obs = np.stack([env._get_obs() for env in envs])
        if reward_log is not None:
            reward_log.write(json.dumps(summarize_reward_infos(infos), sort_keys=True) + '\n')
            reward_log.flush()
        obs_arr[slot] = obs_l
        aux_arr[slot] = aux_l

    conn.send(("ready", None))
    while True:
        try:
            msg, payload = conn.recv()
        except (EOFError, KeyboardInterrupt, OSError):
            return
        if msg == "weights":
            try:
                load_state(payload)
                conn.send(("ok", None))
            except Exception:
                import traceback
                conn.send(("error", traceback.format_exc()))
            continue
        elif msg == "collect":
            n_chunks = payload
        elif msg == "set_iter":
            for e in envs:
                e.train_iter = float(payload)
            conn.send(("ok", None))
            continue
        elif msg == "stats":
            agg = dict(episodes=0, success=0, level_sum=0.0, types={})
            for e in envs:
                s = e.stats
                agg["episodes"] += s["episodes"]
                agg["success"] += s["success"]
                agg["level_sum"] += s["level_sum"]
                for k, v in s["types"].items():
                    agg["types"][k] = agg["types"].get(k, 0) + v
                e.stats = dict(episodes=0, success=0, level_sum=0.0, types={})
            conn.send(("stats", agg))
            continue
        elif msg == "close":
            conn.close()
            return
        else:
            continue

        try:
            for k in range(n_chunks):
                beat(f"chunk {k} start")
                collect_chunk(k % n_slot)
                beat(f"chunk {k} done, sending")
                conn.send(("chunk", (k, k % n_slot)))
                beat(f"chunk {k} sent")
        except Exception:
            import traceback
            try:
                conn.send(("error", traceback.format_exc()))
            except Exception:
                pass
            return
        beat("waiting weights")
        try:
            msg2, payload2 = conn.recv()
        except (EOFError, KeyboardInterrupt, OSError):
            beat("weights recv failed")
            return
        beat(f"got {msg2}")
        if msg2 == "weights":
            load_state(payload2)
            conn.send(("ok", None))
            beat("loaded weights")
        elif msg2 == "close":
            return


# ------------------------------------------------------------------ learner
class Learner:
    def __init__(self, n_workers=24, envs_per_worker=4, seed=0,
                 episode_len=600, device=None, out_dir=None,
                 batch_target=None, train_stage="mixed", train_track=None,
                 gap_width_m=None, decoder_path=None, command_speed=None,
                 command_ramp=None, residual_actor=False, reward_profile='legacy',
                 material_start_profile='default', material_aux_limit=0.2,
                 residual_variant='v1', material_command_profile='instant_v1'):
        import multiprocessing as mp
        import multiprocessing.shared_memory as sm
        from multiprocessing.connection import wait
        self._wait = wait
        self.residual_actor = bool(residual_actor)
        if residual_variant not in ('v1', 'v2', 'v3', 'v4') or (residual_variant != 'v1' and not residual_actor):
            raise ValueError('invalid residual architecture selection')
        self.residual_variant = residual_variant
        self.action_dim = 30 if residual_variant == 'v4' else ACT_DIM
        self.aux_width = self.action_dim + 3
        self.material_wheel_control = 'residual_v1' if residual_variant == 'v4' else 'none'
        self.material_command_profile = policy_material_command_profile({'material_command_profile': material_command_profile})
        self.material_aux_limit = material_aux_limit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_envs = n_workers * envs_per_worker
        self.epw = envs_per_worker
        self.batch_target = 65536 if batch_target is None else batch_target
        self.W = n_workers
        self.K = compute_chunk_count(
            self.batch_target, n_workers, envs_per_worker, CHUNK_H)
        self.n_slot = self.K
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        self.shms = []
        for w in range(n_workers):
            assigned_stage = worker_stage_for_index(train_stage, w, n_workers)
            obs_shm = sm.SharedMemory(create=True,
                size=self.n_slot * CHUNK_H * envs_per_worker * OBS_WORKER * 4)
            aux_shm = sm.SharedMemory(create=True,
                size=self.n_slot * CHUNK_H * envs_per_worker * self.aux_width * 4)
            self.shms.append((obs_shm, aux_shm))
            parent, child = ctx.Pipe()
            cfg = dict(seed=seed * 1000 + w,
                       episode_len=(3000 if (train_stage == "track_curriculum"
                                             and assigned_stage == "full_track")
                                    else episode_len if assigned_stage == "full_track"
                                    else min(episode_len, 1200)),
                       eval_mode=False, epw=envs_per_worker,
                       train_stage=assigned_stage,
                       train_track=(train_track if assigned_stage in
                                    ("single_gap", "double_gap", "full_track",
                                     "rough_slope_retain") else None),
                       gap_width_m=((gap_width_m or 0.3)
                                    if assigned_stage == "single_gap" else None),
                       decoder_path=decoder_path, command_speed=command_speed,
                       command_ramp=command_ramp, residual_actor=self.residual_actor,
                       residual_variant=self.residual_variant, action_dim=self.action_dim,
                       material_wheel_control=self.material_wheel_control,
                       material_command_profile=self.material_command_profile,
                       reward_profile=reward_profile,
                       material_start_profile=material_start_for_worker(material_start_profile, w, n_workers),
                       material_aux_limit=material_aux_limit,
                       reward_log=(os.path.join(out_dir, f'reward-worker-{w}.jsonl')
                                   if reward_profile != 'legacy' and out_dir else None),
                       obs_shm=obs_shm.name, aux_shm=aux_shm.name,
                       n_slot=self.n_slot)
            p = ctx.Process(target=worker_main, args=(child, cfg), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)
        for c in self.conns:
            assert c.recv()[0] == "ready"
        self.conn_index = {id(c): i for i, c in enumerate(self.conns)}
        self.obs_rings = [np.ndarray((self.n_slot, CHUNK_H, envs_per_worker,
                                      OBS_WORKER), dtype=np.float32,
                                     buffer=shm.buf)
                          for shm, _ in self.shms]
        self.aux_rings = [np.ndarray((self.n_slot, CHUNK_H, envs_per_worker,
                                      self.aux_width), dtype=np.float32, buffer=shm.buf)
                          for _, shm in self.shms]

        use_res = bool(getattr(self, "residual_actor", False))
        self.enc = ExteroEncoder(split_frozen=use_res).to(self.device)
        actor_cls = {'v1': ResidualActor, 'v2': TrainableResidualActor,
                     'v3': BoundedBaseResidualActor, 'v4': WheelResidualActor}[self.residual_variant]
        self.actor = (actor_cls(hidden=512, split_extero=True)
                      if use_res else Actor()).to(self.device)
        self.critic = Critic().to(self.device)
        if self.residual_actor:
            for p in list(self.enc.net.parameters()) + list(self.actor.base_body.parameters()) + list(self.actor.base_head.parameters()):
                p.requires_grad_(False)
            train_actor_params = [p for p in list(self.enc.parameters()) + list(self.actor.parameters()) if p.requires_grad]
        else:
            train_actor_params = list(self.enc.parameters()) + list(self.actor.parameters())
        self.opt = torch.optim.Adam([
            {"params": train_actor_params, "lr": 3e-4},
            {"params": list(self.critic.parameters()), "lr": 3e-4},
        ])
        self.norm = RunningNorm(OBS_WORKER)
        self.minibatch = 16384
        self.gamma, self.lam = 0.997, 0.97  # long credit horizon: jump -> cross
        self.clip, self.entropy_coef = 0.18, 0.0005
        self.kl_target = 0.012
        self.decoder_path = os.path.abspath(decoder_path or envmod.TVAE_DEC)
        self.decoder_sha256 = _sha256(self.decoder_path)
        self.out_dir = out_dir or os.path.join(HERE, "runs")
        os.makedirs(self.out_dir, exist_ok=True)
        self.iter_count = 0

    # ------------------------------------------------------------ plumbing
    def _state_payload(self):
        return dict(
            enc={k: v.detach().cpu() for k, v in self.enc.state_dict().items()},
            actor={k: v.detach().cpu() for k, v in self.actor.state_dict().items()},
            norm=self.norm.state(),
        )

    def _broadcast_weights(self, verbose=False):
        payload = self._state_payload()
        if verbose: print("  [bc] sending weights...", flush=True)
        for wi, c in enumerate(self.conns):
            c.send(("weights", payload))
            tag, _ = c.recv()
            assert tag == "ok", tag
            if verbose: print(f"  [bc] worker {wi} ok", flush=True)

    def fetch_stats(self):
        aggs = []
        for c in self.conns:
            c.send(("stats", None))
        for c in self.conns:
            tag, agg = self._recv_checked(c)
            aggs.append(agg)
        tot = dict(episodes=0, success=0, level_sum=0.0, types={})
        for a in aggs:
            tot["episodes"] += a["episodes"]
            tot["success"] += a["success"]
            tot["level_sum"] += a["level_sum"]
            for k, v in a["types"].items():
                tot["types"][k] = tot["types"].get(k, 0) + v
        return tot

    def broadcast_iter(self, it):
        for c in self.conns:
            c.send(("set_iter", it))
        for c in self.conns:
            tag, _ = c.recv()
            assert tag == "ok", tag

    @staticmethod
    def _recv_checked(c):
        tag, payload = c.recv()
        if tag == "error":
            raise RuntimeError("worker error: " + str(payload))
        return tag, payload

    # ------------------------------------------------------------- collect
    def collect(self):
        W, K = self.W, self.K
        for c in self.conns:
            c.send(("collect", K))
        order = {i: [] for i in range(W)}
        conns = list(self.conns)
        remaining = W * K
        while remaining > 0:
            for c in self._wait(conns):
                i = self.conn_index[id(c)]
                tag, (k, slot) = self._recv_checked(c)
                if tag != "chunk":
                    continue
                order[i].append((k, slot))
                remaining -= 1
        obs_w, aux_w = [], []
        for i in range(W):
            slots = [s for _, s in order[i]]
            # (K, H, epw, X) -> (K*H, epw, X): flatten the worker time axis
            obs_w.append(self.obs_rings[i][slots].reshape(K * CHUNK_H, self.epw, -1))
            aux_w.append(self.aux_rings[i][slots].reshape(K * CHUNK_H, self.epw, -1))
        obs = np.concatenate(obs_w, axis=1)              # (T, N, 426)
        aux = np.concatenate(aux_w, axis=1)              # (T, N, 32)
        d = self.action_dim
        B = dict(obs=obs, act=aux[..., :d], logp=aux[..., d],
                 rew=aux[..., d + 1], done=aux[..., d + 2] > 0.5)
        return B

    # ------------------------------------------------------- obs to actor
    def actor_obs(self, ob_n_t):
        p = ob_n_t[:, :PROPRIO]
        e = ob_n_t[:, PROPRIO:PROPRIO + EXTERO]
        g = ob_n_t[:, PROPRIO + EXTERO:]
        return torch.cat([p, self.enc(e), g], dim=1)

    # ---------------------------------------------------------------- GAE
    def _norm_batch(self, obs):
        shape = obs.shape
        flat = obs.reshape(-1, OBS_WORKER).astype(np.float64)
        return self.norm.normalize(flat).reshape(shape)

    def compute_gae(self, B):
        T, N = B["rew"].shape
        obs_n = self._norm_batch(B["obs"])
        flat = torch.from_numpy(obs_n.reshape(T * N, -1)).to(self.device)
        with torch.no_grad():
            vals = self.critic(flat).cpu().numpy().reshape(T, N)
        adv = np.zeros((T, N), dtype=np.float32)
        lastgaelam = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            next_v = vals[t + 1] if t < T - 1 else vals[t]
            mask = 1.0 - B["done"][t].astype(np.float32)
            delta = B["rew"][t] + self.gamma * next_v * mask - vals[t]
            lastgaelam = delta + self.gamma * self.lam * mask * lastgaelam
            adv[t] = lastgaelam
        return adv, vals, obs_n

    # ------------------------------------------------------------- update
    def update(self, B):
        T, N = B["rew"].shape
        adv, vals, obs_n = self.compute_gae(B)
        ret = adv + vals
        obs_flat = obs_n.reshape(T * N, -1)
        act_flat = B["act"].reshape(T * N, -1)
        logp_flat = B["logp"].reshape(T * N)
        ret_flat = ret.reshape(T * N)
        n = len(obs_flat)
        idx = np.arange(n)
        dev = self.device
        stats = {}
        if getattr(self, "_critic_warmup_remaining", 0) > 0:
            self._critic_warmup_remaining -= 1
            c_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
            for _ in range(max(6, getattr(self, "_EPOCHS", 6))):
                np.random.shuffle(idx)
                for b0 in range(0, n, min(self.minibatch, 8192)):
                    sel = idx[b0:b0 + min(self.minibatch, 8192)]
                    ob = torch.from_numpy(obs_flat[sel]).to(dev)
                    v = self.critic(ob)
                    v_loss = 0.5 * nn.functional.huber_loss(
                        v, torch.from_numpy(ret_flat[sel]).to(dev), delta=10.0)
                    c_opt.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    c_opt.step()
                    stats = dict(pi=0.0, v=float(v_loss.detach()), ent=0.0,
                                 kl=0.0, lr=self.opt.param_groups[0]["lr"])
            if not getattr(self, "freeze_norm", False):
                self.norm.update(B["obs"].reshape(-1, OBS_WORKER).astype(np.float64))
            return stats
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)
        adv_flat = adv.reshape(T * N)
        bc_coef = float(getattr(self, "bc_coef", 0.0))
        has_bc = bc_coef > 0.0 and getattr(self, "_bc_actor", None) is not None
        # Anchor ResidualActor on flat/gentle cruise so PPO never corrupts approach!
        is_res_model = isinstance(self.actor, ResidualActor)
        for epoch in range(getattr(self, "_EPOCHS", 6)):
            np.random.shuffle(idx)
            early_stop = False
            for b0 in range(0, n, self.minibatch):
                sel = idx[b0:b0 + self.minibatch]
                ob = torch.from_numpy(obs_flat[sel]).to(dev)
                ao = self.actor_obs(ob)
                dist = self.actor.dist(ao)
                a_old = torch.from_numpy(act_flat[sel]).to(dev)
                logp = dist.log_prob(a_old).sum(-1)
                old_lp = torch.from_numpy(logp_flat[sel]).to(dev)
                log_ratio = logp - old_lp
                approx_kl = float(((log_ratio.exp() - 1.0) - log_ratio).mean().detach())
                if approx_kl > self.kl_target * 2.5 and b0 > 0:
                    early_stop = True
                    break
                ent = dist.entropy()
                ratio = log_ratio.exp()
                a_t = torch.from_numpy(adv_flat[sel]).to(dev)
                s1 = ratio * a_t
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a_t
                pi_loss = -torch.min(s1, s2).mean()
                v = self.critic(ob)
                v_loss = 0.5 * nn.functional.huber_loss(
                    v, torch.from_numpy(ret_flat[sel]).to(dev), delta=10.0)
                ent_loss = -self.entropy_coef * ent[:, :LATENT + AUXD].mean()
                loss = pi_loss + v_loss + ent_loss
                if is_res_model:
                    ext_raw = torch.from_numpy(
                        B["obs"].reshape(T * N, -1)[sel, PROPRIO:PROPRIO + EXTERO]).to(dev)
                    # When neither local heightmap nor 5m LiDAR sees a >12cm step/drop, keep gate=0!
                    safe_mask = ((ext_raw[:, :210].max(dim=1).values - ext_raw[:, :210].min(dim=1).values) < 0.14) & (
                        (ext_raw[:, 210:].max(dim=1).values - ext_raw[:, 210:].min(dim=1).values) < 0.25
                    )
                    if safe_mask.any():
                        _, _, res_m, raw_g = self.actor.forward_parts(ao[safe_mask])
                        loss = loss + 0.5 * (raw_g ** 2).mean() + 0.05 * (res_m ** 2).mean()
                if has_bc:
                    # Anchor skill toward expert/init policy!
                    ext_raw = torch.from_numpy(
                        B["obs"].reshape(T * N, -1)[sel, PROPRIO:PROPRIO + 210]).to(dev)
                    bc_mask = (torch.ones(len(sel), dtype=torch.bool, device=dev)
                               if is_res_model else (ext_raw.std(dim=1) < 0.03))
                    if bc_mask.any():
                        with torch.no_grad():
                            p_b = ob[bc_mask, :PROPRIO]
                            e_b = ob[bc_mask, PROPRIO:PROPRIO + EXTERO]
                            g_b = ob[bc_mask, PROPRIO + EXTERO:]
                            ao_ref = torch.cat([p_b, self._bc_enc(e_b), g_b], dim=1)
                            mu_ref = self._bc_actor.dist(ao_ref).mean
                        mu_cur = dist.mean[bc_mask]
                        bc_loss = nn.functional.mse_loss(mu_cur[:, :28], mu_ref[:, :28]) + 0.05 * nn.functional.mse_loss(mu_cur[:, 28], mu_ref[:, 28])
                        if mu_cur.shape[-1] == 30:
                            bc_loss = bc_loss + nn.functional.mse_loss(mu_cur[:, 29], mu_ref[:, 29])
                        loss = loss + bc_coef * bc_loss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.enc.parameters()) + list(self.actor.parameters()), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt.step()
                with torch.no_grad():
                    min_ls = float(np.log(getattr(self, "min_std", 0.10)))
                    self.actor.log_std[:LATENT].clamp_(min=min_ls, max=float(np.log(0.45)))
                    self.actor.log_std[LATENT:LATENT + AUXD].clamp_(min=float(np.log(0.14)), max=float(np.log(1.0)))
                    self.actor.log_std[LATENT + AUXD:].clamp_(min=float(np.log(0.30)), max=float(np.log(1.6)))
                clamp_wheel_log_std(self.actor)
                stats = dict(pi=float(pi_loss.detach()), v=float(v_loss.detach()),
                             ent=float(ent[:, :LATENT + AUXD].mean().detach()))
            if early_stop:
                break
        if not getattr(self, "freeze_norm", False):
            self.norm.update(B["obs"].reshape(-1, OBS_WORKER).astype(np.float64))
        with torch.no_grad():
            ob = torch.from_numpy(obs_flat[:8192]).to(dev)
            dist = self.actor.dist(self.actor_obs(ob))
            logp_new = dist.log_prob(torch.from_numpy(act_flat[:8192]).to(dev)).sum(-1)
            lr_chk = logp_new - torch.from_numpy(logp_flat[:8192]).to(dev)
            kl = float(((lr_chk.exp() - 1.0) - lr_chk).mean().item())
        if kl > self.kl_target * 2:
            self.opt.param_groups[0]["lr"] = max(self.opt.param_groups[0]["lr"] / 1.5, 5e-6)
        elif kl < self.kl_target / 2:
            self.opt.param_groups[0]["lr"] = min(
                self.opt.param_groups[0]["lr"] * 1.25, getattr(self, "_max_lr", 3e-4))
        stats["kl"] = kl
        stats["lr"] = self.opt.param_groups[0]["lr"]
        return stats

    def track_probe(self, track_path, n_ep=3, max_steps=4000, seed_base=77,
                    eval_protocol="legacy"):
        """Evaluate this policy on an explicit fixed track without worker state."""
        from collections import Counter
        import stage3_rl.env as envmod

        if not track_path:
            raise ValueError("track_probe requires an explicit track_path")
        if n_ep <= 0 or max_steps <= 0:
            raise ValueError("n_ep and max_steps must be positive")
        if not os.path.isfile(track_path):
            raise FileNotFoundError(track_path)

        sequence = ["start", "rough_flat", "rough_slope", "gap",
                    "continuous_gap", "stairs_up", "stairs_down",
                    "stepping_one_bridge", "stool", "crawl",
                    "stairs_cliff", "slope_cliff", "start"]
        slow_segments = {5, 6, 7, 9, 10, 11}
        material = str(track_path).lower().endswith('.xml')
        if material:
            sequence = ['approach'] * 3 + ['tires_brick', 'stairs_bridge',
                       'stairs_frame'] + ['exit'] * 7
        results = []
        for episode_index in range(n_ep):
            seed = seed_base + episode_index
            material_kwargs = ({'material_aux_limit': self.material_aux_limit,
                                'material_wheel_control': getattr(self, 'material_wheel_control', 'none')}
                               if hasattr(self, 'material_aux_limit') else {})
            env = envmod.Go2WEnv(seed=seed, fixed_track=track_path,
                                 eval_mode=True, episode_len=max_steps,
                                 eval_protocol=eval_protocol,
                                 decoder_path=getattr(
                                     self, "decoder_path",
                                     os.path.join(ROOT, "stage2_tvae", "tvae_decoders.npz")),
                                 **material_kwargs)
            obs = env.reset()[None]
            reset_ctx = getattr(self.actor, "reset_context", None)
            if callable(reset_ctx):
                reset_ctx()
            requested_pose = list(env.requested_initial_pose)
            settled_pose = list(env.settled_initial_pose)
            done = False
            steps = 0
            max_x = float(env.data.qpos[0])
            total_reward = 0.0
            info = {"reason": None, "success": False}
            while not done and steps < max_steps:
                segment = min(max(int(float(env.data.qpos[0]) // 8.0), 0), 12)
                env.vx_cmd = (material_command_speed(steps, float(env.data.qpos[0]),
                                                    getattr(self, 'material_command_profile', 'instant_v1'))
                              if material else (0.8 if segment in slow_segments else 1.5))
                obs_n = self.norm.normalize(obs)
                obs_tensor = torch.from_numpy(obs_n).to(self.device)
                with torch.no_grad():
                    proprio = obs_tensor[:, :PROPRIO]
                    extero = obs_tensor[:, PROPRIO:PROPRIO + EXTERO]
                    gait = obs_tensor[:, PROPRIO + EXTERO:]
                    actor_obs = torch.cat([proprio, self.enc(extero), gait], dim=1)
                    action = self.actor.dist(actor_obs).mean
                obs, reward, done, info = env.step(action.cpu().numpy()[0])
                obs = (env._get_obs() if hasattr(env, "_get_obs") else obs)[None]
                steps += 1
                total_reward += float(reward)
                max_x = max(max_x, float(env.data.qpos[0]))
            reason = info.get("reason") or "evaluation_limit"
            completed = min(max(int(max_x // 8.0), 0), len(sequence))
            furthest_index = min(max(int(max_x // 8.0), 0), len(sequence) - 1)
            results.append({
                "seed": int(seed),
                "success": bool(info.get("success", False)),
                "reason": reason,
                "steps": int(steps),
                "max_x": float(max_x),
                "final_x": float(env.data.qpos[0]),
                "furthest_segment_index": furthest_index,
                "furthest_segment": sequence[furthest_index],
                "completed_segments": completed,
                "total_reward": float(total_reward),
                "terminal_contacts": info.get("terminal_contacts", []),
                "requested_initial_pose": requested_pose,
                "settled_initial_pose": settled_pose,
            })

        max_xs = [result["max_x"] for result in results]
        success_count = sum(result["success"] for result in results)
        failures = Counter(result["reason"] for result in results
                           if not result["success"])
        completed_counts = {
            f"{index}:{name}": sum(result["completed_segments"] >= index + 1
                                    for result in results)
            for index, name in enumerate(sequence)
        }
        return {
            "best_x": float(max(max_xs)),
            "median_x": float(np.median(max_xs)),
            "n_ok": int(success_count),
            "success_count": int(success_count),
            "completed_segment_counts": completed_counts,
            "completed_segments": [result["completed_segments"]
                                    for result in results],
            "failure_counts": dict(failures),
            "seed_base": int(seed_base),
            "eval_protocol": eval_protocol,
            "episode_seeds": [result["seed"] for result in results],
            "episodes": results,
        }

    # ------------------------------------------------------------ persist
    def save(self, tag):
        import tempfile
        path = os.path.join(self.out_dir, f"model_{tag}.pt")
        tmp = path + ".tmp"
        save_dict = {
            "enc": self.enc.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "norm": self.norm.state(),
            "iter": self.iter_count,
            "decoder_path": self.decoder_path,
            "decoder_sha256": self.decoder_sha256,
            "material_aux_limit": self.material_aux_limit,
        }
        save_dict['material_wheel_control'] = getattr(self, 'material_wheel_control', 'none')
        save_dict['material_command_profile'] = getattr(self, 'material_command_profile', 'instant_v1')
        if getattr(self, "residual_actor", False):
            save_dict["actor_arch"] = ('residual' if self.residual_variant == 'v1'
                                       else 'residual_' + self.residual_variant)
            save_dict["actor_hidden"] = 512
        torch.save(save_dict, tmp)
        os.replace(tmp, path)  # atomic on Windows

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        for shm, _ in self.shms:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=positive_int, default=2000)
    parser.add_argument("--workers", type=positive_int, default=24)
    parser.add_argument("--epw", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True,
                        help="new or empty directory for this training run")
    parser.add_argument("--epochs", type=positive_int, default=6)
    parser.add_argument("--init", default=None)
    parser.add_argument("--decoder-path",
                        default=os.path.join(ROOT, "stage2_tvae", "tvae_decoders.npz"),
                        help="immutable torque-decoder NPZ for this run")
    parser.add_argument("--batch-target", type=positive_int, default=65536)
    parser.add_argument("--probe-track", required=True,
                        help="explicit fixed-track NPZ or material XML used by periodic probes")
    parser.add_argument('--probe-episodes', type=positive_int, default=5)
    parser.add_argument('--probe-max-steps', type=positive_int, default=11000)
    parser.add_argument("--eval-protocol", choices=("legacy", "pose-v1"),
                        default="legacy", help="fixed-track probe start protocol")
    parser.add_argument("--train-stage", choices=TRAIN_STAGES, default="mixed")
    parser.add_argument('--material-aux-limit', type=parse_material_aux_limit, default=0.2,
                        help='material auxiliary action magnitude (0,8], or none')
    parser.add_argument('--material-start-profile', choices=('default', 'stair_mix', 'stair_retain', 'tire_retain', 'full_start'),
                        default='default', help='training-only safe start distribution')
    parser.add_argument('--reward-profile', choices=('legacy', 'paper_adapted', 'stair_v1', 'integrated_v1', 'tire_v1', 'tire_l1_v1'),
                        default='legacy', help='experimental material reward profile')
    parser.add_argument("--train-track", default=None,
                        help="original NPZ used to construct training-only stages")
    parser.add_argument("--gap-width", type=float, default=None,
                        help="single-gap width in meters; never changes probe track")
    parser.add_argument("--episode-len", type=positive_int, default=600)
    parser.add_argument("--command-speed", type=float, default=None,
                        help="fixed command speed for targeted flat training only")
    parser.add_argument("--command-ramp-start", type=float, default=None)
    parser.add_argument("--command-ramp-end", type=float, default=None)
    parser.add_argument("--command-ramp-steps", type=positive_int, default=None)
    parser.add_argument("--critic-warmup", type=int, default=0,
                        help="critic-only warmup iterations when initializing from BC")
    parser.add_argument("--bc-coef", type=float, default=0.0,
                        help="flat-terrain MSE anchor weight toward --init policy")
    parser.add_argument("--latent-std", type=float, default=None,
                        help="override initial exploration std on 16-D latent")
    parser.add_argument("--aux-std", type=float, default=None,
                        help="override initial exploration std on 12-D aux")
    parser.add_argument("--gait-std", type=float, default=None,
                        help="override initial exploration std on 1-D gait logit")
    parser.add_argument('--wheel-std', type=float, default=None,
                        help='override v4 wheel exploration base std in [0.14,2]')
    parser.add_argument("--freeze-norm", action="store_true", default=False,
                        help="keep RunningNorm frozen from --init checkpoint")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="initial Adam learning rate")
    parser.add_argument("--residual-actor", action="store_true", default=False,
                        help="use frozen-base legacy ResidualActor")
    parser.add_argument('--residual-v2', action='store_true', default=False,
                        help='use frozen-base trainable soft-gated residual')
    parser.add_argument('--residual-v3', action='store_true', default=False,
                        help='soft-gated residual around historically clamped base actions')
    parser.add_argument('--residual-v4', action='store_true', default=False,
                        help='bounded-base residual plus policy-controlled wheel speed')
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if not os.path.isfile(args.probe_track):
        raise FileNotFoundError(args.probe_track)
    if args.init and not os.path.isfile(args.init):
        raise FileNotFoundError(args.init)
    if not os.path.isfile(args.decoder_path):
        raise FileNotFoundError(args.decoder_path)
    compute_chunk_count(args.batch_target, args.workers, args.epw, CHUNK_H)
    validate_training_args(args)
    args.out = ensure_empty_output_dir(args.out)
    args.probe_track = os.path.realpath(args.probe_track)
    if args.init:
        args.init = os.path.realpath(args.init)
    args.decoder_path = os.path.realpath(args.decoder_path)
    if args.train_track:
        args.train_track = os.path.realpath(args.train_track)
    args.material_command_profile = (policy_material_command_profile(
        torch.load(args.init, map_location='cpu', weights_only=False)) if args.init else 'instant_v1')
    write_training_run_config(args, args.out)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    init_is_res = False
    init_arch = None
    if args.init:
        ck_probe = torch.load(args.init, map_location="cpu", weights_only=False)
        init_arch = ck_probe.get('actor_arch')
        init_is_res = (init_arch in ('residual', 'residual_v2', 'residual_v3', 'residual_v4')
                       or "res_head.weight" in ck_probe.get("actor", {}))
    requested_arch = ('residual_v4' if args.residual_v4 else
                      'residual_v3' if args.residual_v3 else
                      'residual_v2' if args.residual_v2 else
                      'residual' if args.residual_actor else None)
    saved_arch = init_arch if init_arch in ('residual_v2', 'residual_v3', 'residual_v4') else 'residual'
    if requested_arch and init_is_res and requested_arch != saved_arch:
        raise ValueError('cannot silently reinterpret a different residual checkpoint architecture')
    actual_arch = requested_arch or (saved_arch if init_is_res else None)
    use_res = actual_arch is not None
    residual_variant = {'residual': 'v1', 'residual_v2': 'v2', 'residual_v3': 'v3',
                        'residual_v4': 'v4', None: 'v1'}[actual_arch]
    if residual_variant == 'v4' and (not args.train_track or not args.train_track.lower().endswith('.xml')
                                     or not args.probe_track.lower().endswith('.xml')):
        raise ValueError('residual_v4 requires material training and probe tracks')
    if args.wheel_std is not None and (residual_variant != 'v4' or not np.isfinite(args.wheel_std)
                                       or not 0.14 <= args.wheel_std <= 2.0):
        raise ValueError('wheel std requires residual_v4 and must be in [0.14,2]')
    lr = Learner(n_workers=args.workers, envs_per_worker=args.epw,
                 seed=args.seed, out_dir=args.out,
                 batch_target=args.batch_target, residual_actor=use_res,
                 residual_variant=residual_variant,
                 material_command_profile=args.material_command_profile, **worker_env_config(args))
    lr._max_lr = float(args.lr)
    lr.opt.param_groups[0]["lr"] = float(args.lr)
    lr.opt.param_groups[1]["lr"] = 3e-4
    try:
        if args.init:
            ck = torch.load(args.init, map_location=lr.device, weights_only=False)
            expected_decoder = ck.get("decoder_sha256")
            if expected_decoder is None:
                legacy_decoder = os.path.join(ROOT, "stage2_tvae", "tvae_decoders.npz")
                if lr.decoder_sha256 != _sha256(legacy_decoder):
                    raise ValueError("initial checkpoint has no decoder provenance; "
                                     "cannot pair it with a different decoder")
            elif expected_decoder != lr.decoder_sha256:
                raise ValueError("initial checkpoint was trained with a different decoder")
            if use_res and not init_is_res:
                enc_sub = {k.split("net.", 1)[1]: v for k, v in ck["enc"].items() if k.startswith("net.")}
                lr.enc.net.load_state_dict(enc_sub)
                lr.actor.base_body.load_state_dict(
                    {k.split("body.", 1)[1]: v for k, v in ck["actor"].items() if k.startswith("body.")}
                )
                lr.actor.base_head.load_state_dict(
                    {k.split("head.", 1)[1]: v for k, v in ck["actor"].items() if k.startswith("head.")}
                )
                with torch.no_grad():
                    lr.actor.log_std.copy_(ck['actor']['log_std'])
            else:
                lr.enc.load_state_dict(ck["enc"])
                lr.actor.load_state_dict(ck["actor"])
            lr.critic.load_state_dict(ck["critic"])
            lr.norm.load(ck["norm"])
            if (args.latent_std is not None or args.aux_std is not None
                    or args.gait_std is not None or args.wheel_std is not None):
                with torch.no_grad():
                    if args.latent_std is not None:
                        lr.actor.log_std[:LATENT] = float(np.log(args.latent_std))
                    if args.aux_std is not None:
                        lr.actor.log_std[LATENT:LATENT + AUXD] = float(np.log(args.aux_std))
                    if args.gait_std is not None:
                        lr.actor.log_std[LATENT + AUXD:] = float(np.log(args.gait_std))
                    if args.wheel_std is not None:
                        lr.actor.wheel_log_std.fill_(float(np.log(args.wheel_std)))
            if args.freeze_norm:
                lr.freeze_norm = True
            if args.bc_coef > 0.0:
                import copy
                lr._bc_enc = copy.deepcopy(lr.enc).eval()
                lr._bc_actor = copy.deepcopy(lr.actor).eval()
                for p in list(lr._bc_enc.parameters()) + list(lr._bc_actor.parameters()):
                    p.requires_grad_(False)
                lr.bc_coef = float(args.bc_coef)
            if args.critic_warmup > 0:
                lr._critic_warmup_remaining = int(args.critic_warmup)
            print(f"initialized from {args.init} (iter {ck.get('iter')})", flush=True)
        # Workers start with independent random networks: synchronize the loaded
        # actor/encoder/normalizer before collecting even the first PPO batch.
        lr._broadcast_weights()
        print("initial policy synchronized to all workers", flush=True)
        lr.save("initial")
        lr._EPOCHS = args.epochs
        t0 = time.time()
        rew_hist = []
        probe_path = os.path.join(args.out, "probe_metrics.jsonl")
        update_path = os.path.join(args.out, 'update_metrics.jsonl')
        for it in range(args.iters):
            lr.iter_count = it
            lr.broadcast_iter(it)
            stats_agg = lr.fetch_stats()   # workers idle here (safe point)
            sr = stats_agg["success"] / max(1, stats_agg["episodes"])
            ml = stats_agg["level_sum"] / max(1, stats_agg["episodes"])
            n_eps = stats_agg["episodes"]
            t1 = time.time()
            B = lr.collect()
            t2 = time.time()
            stats = lr.update(B)
            t3 = time.time()
            with torch.no_grad():
                raw_sample = B['obs'].reshape(-1, OBS_WORKER)[:2048]
                sample = torch.from_numpy(lr.norm.normalize(raw_sample)).to(lr.device)
                sample_dist = lr.actor.dist(lr.actor_obs(sample))
                std_record = {'aux_effective_mean': float(sample_dist.stddev[:, LATENT:LATENT+AUXD].mean().cpu()),
                              'aux_base_mean': float(lr.actor.log_std[LATENT:LATENT+AUXD].exp().mean().cpu())}
                if lr.action_dim == 30:
                    std_record.update(wheel_effective_mean=float(sample_dist.stddev[:, 29].mean().cpu()),
                                      wheel_base=float(lr.actor.wheel_log_std.exp().cpu()))
            with open(update_path, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps({'iter': it + 1, 'stats': stats, 'exploration_std': std_record}) + '\n')
            mean_reward = float(B["rew"].mean())
            rew_hist.append(mean_reward)
            lr.save("latest")
            if should_probe(it):
                lr.save(f"it{it+1:05d}")
                probe_seed = args.seed + 9000
                try:
                    probe = lr.track_probe(
                        args.probe_track, n_ep=args.probe_episodes, max_steps=args.probe_max_steps,
                        seed_base=probe_seed,
                        eval_protocol=args.eval_protocol)
                    record = {
                        "iter": it + 1,
                        "eval_protocol": args.eval_protocol,
                        "seed": probe["seed_base"],
                        "episode_seeds": probe["episode_seeds"],
                        "mean_reward": mean_reward,
                        "mean_training_success": float(sr),
                        "best_x": probe["best_x"],
                        "median_x": probe["median_x"],
                        "success_count": probe["success_count"],
                        "completed_segments": probe["completed_segments"],
                        "completed_segment_counts": probe["completed_segment_counts"],
                        "failure_counts": probe["failure_counts"],
                        "episodes": probe["episodes"],
                    }
                    write_probe_record(probe_path, record)
                    print(f"  [probe] it={it+1} best_x={probe['best_x']:.1f} "
                          f"median_x={probe['median_x']:.1f} "
                          f"ok={probe['success_count']}/{args.probe_episodes}", flush=True)
                    score = (probe["success_count"], probe["median_x"], probe["best_x"])
                    if score > getattr(lr, "_best_probe_score", (-1, -1.0, -1.0)):
                        lr._best_probe_score = score
                        lr._best_probe_x = probe["best_x"]
                        lr.save("best")
                        print(f"  [probe] new best saved (ok={probe['success_count']}/{args.probe_episodes}, "
                              f"median={probe['median_x']:.1f}m, best={probe['best_x']:.1f}m)",
                              flush=True)
                except Exception as exc:
                    write_probe_record(probe_path, {
                        "iter": it + 1,
                        "eval_protocol": args.eval_protocol,
                        "seed": probe_seed,
                        "mean_reward": mean_reward,
                        "mean_training_success": float(sr),
                        "probe_error": str(exc),
                    })
                    print("  [probe] failed:", exc, flush=True)
                print(f"iter {it:5d} rew {np.mean(rew_hist[-20:]):+.3f} "
                      f"kl {stats['kl']:.4f} lr {stats['lr']:.2e} "
                      f"pi {stats['pi']:+.3f} v {stats['v']:.1f} "
                      f"ent {stats['ent']:.3f} "
                      f"[collect {t2-t1:.1f}s upd {t3-t2:.1f}s "
                      f"total {t3-t0:.0f}s] sr {sr:.2f} lvl {ml:.1f} "
                      f"eps {n_eps}", flush=True)
            lr._broadcast_weights()
        lr.save("final")
    finally:
        lr.save("final")
        lr.close()
    print("done")


if __name__ == "__main__":
    main()
