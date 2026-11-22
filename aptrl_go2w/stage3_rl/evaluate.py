"""Evaluate an APT-RL policy on a fixed track with immutable run metadata."""
import argparse
import hashlib
from collections import Counter
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.eval_protocol import POSE_V1_RANGES, material_command_speed
from common.evaluation import (build_run_manifest, summarize_track,
                               write_run_manifest)
from common.terrain import load_npz_track
from stage3_rl.env import (EXTERO, PROPRIO, TVAE_DEC, Go2WEnv, LANE_CENTER,
                           LANE_HALF_WIDTH)
from stage3_rl.ppo_train import (Actor, ExteroEncoder, ResidualActor,
                                 TemporalResidualActor, TemporalWheelResidualActor,
                                 FeedforwardKinActor,
                                 TrainableResidualActor, BoundedBaseResidualActor,
                                 WheelResidualActor)
from common.material_acceptance import (policy_material_aux_limit, policy_material_wheel_control,
                                        policy_material_command_profile)

SEQ = ["start", "rough_flat", "rough_slope", "gap", "continuous_gap",
       "stairs_up", "stairs_down", "stepping_one_bridge", "stool", "crawl",
       "stairs_cliff", "slope_cliff", "start"]
SLOW_SEGMENTS = {5, 6, 7, 9, 10, 11}
SLOW_SPEED = 0.8
NORMAL_SPEED = 1.5


def positive_int(value):
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative_int(value):
    result = positive_int(value) if int(value) > 0 else int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=os.path.join(HERE, "runs_full", "model_final.pt"))
    parser.add_argument("--track", required=True, help="fixed race-track NPZ")
    parser.add_argument("--decoder-path", default=TVAE_DEC,
                        help="torque-decoder NPZ paired with the policy checkpoint")
    parser.add_argument("--episodes", type=positive_int, default=100)
    parser.add_argument("--episode-len", type=positive_int, default=9000)
    parser.add_argument("--seed", type=int, default=9000,
                        help="first deterministic per-episode seed")
    parser.add_argument("--eval-protocol", choices=("legacy", "pose-v1"),
                        default="legacy", help="seeded initial-pose protocol")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--video", default=None,
                        help="record the best successful episode by exact-seed replay")
    parser.add_argument("--record-episodes", type=nonnegative_int, default=1,
                        help="0 disables --video; otherwise record one best success")
    parser.add_argument("--out", required=True,
                        help="new or empty run directory (never overwritten)")
    return parser


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_decoder_pairing(checkpoint, decoder_path, legacy_decoder_path=TVAE_DEC):
    expected = checkpoint.get("decoder_sha256")
    actual = _sha256(decoder_path)
    if expected and expected != actual:
        raise ValueError("policy checkpoint and decoder artifact hashes do not match")
    if not expected and actual != _sha256(legacy_decoder_path):
        raise ValueError("legacy checkpoint lacks decoder provenance; only its original decoder is valid")
    return actual


def load_policy(ckpt_path, device, decoder_path=TVAE_DEC):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    validate_decoder_pairing(ck, decoder_path)
    policy_material_wheel_control(ck)
    is_temp = ck.get("actor_arch") == "temporal_residual" or "ctx_proj.0.weight" in ck["actor"]
    is_res = ck.get("actor_arch") == "residual" or "res_head.weight" in ck["actor"]
    split_ext = any(k.startswith("res_net.") for k in ck["enc"])
    pass_raw = ck.get("actor_arch") == "temporal_residual_v4" and bool(ck.get("extero_passthrough", False))
    enc = ExteroEncoder(split_frozen=split_ext, passthrough_raw=pass_raw).to(device)
    norm_mean_arr = np.asarray(ck["norm"]["mean"], dtype=np.float32)
    norm_std_arr = np.sqrt(np.asarray(ck["norm"]["var"], dtype=np.float32) + 1e-8)
    if ck.get("actor_arch") == "feedforward_kin_v4":
        actor = FeedforwardKinActor(
            hidden=int(ck.get("actor_hidden", 768)),
            gru_hidden=int(ck.get("gru_hidden", 256)),
            raw_norm_mean=norm_mean_arr,
            raw_norm_std=norm_std_arr,
            extero_passthrough=pass_raw,
        ).to(device)
    elif ck.get("actor_arch") == "temporal_residual_v4":
        actor = TemporalWheelResidualActor(
            hidden=int(ck.get("actor_hidden", 768)),
            gru_hidden=int(ck.get("gru_hidden", 256)),
            raw_norm_mean=norm_mean_arr,
            raw_norm_std=norm_std_arr,
            extero_passthrough=pass_raw,
        ).to(device)
    elif is_temp:
        actor = TemporalResidualActor(hidden=int(ck.get("actor_hidden", 768))).to(device)
    elif ck.get('actor_arch') in ('residual_v2', 'residual_v3', 'residual_v4'):
        actor_cls = {'residual_v2': TrainableResidualActor,
                     'residual_v3': BoundedBaseResidualActor,
                     'residual_v4': WheelResidualActor}[ck['actor_arch']]
        actor = actor_cls(hidden=int(ck.get('actor_hidden', 512)),
                          split_extero=split_ext).to(device)
    elif is_res:
        actor = ResidualActor(hidden=int(ck.get("actor_hidden", 512)),
                              split_extero=split_ext).to(device)
    else:
        actor = Actor().to(device)
    enc.load_state_dict(ck["enc"])
    try:
        actor.load_state_dict(ck["actor"])
    except RuntimeError:
        actor.load_state_dict(ck["actor"], strict=False)
    enc.eval()
    actor.eval()
    norm_mean = ck["norm"]["mean"]
    norm_var = ck["norm"]["var"]
    return enc, actor, norm_mean, norm_var, ck.get("iter", -1)


def make_obs(obs_raw, norm_mean, norm_var, device):
    ob = (obs_raw - norm_mean) / np.sqrt(norm_var + 1e-8)
    return np.clip(ob, -10, 10).astype(np.float32)


class PolicyRunner:
    def __init__(self, ckpt_path, device=None, decoder_path=TVAE_DEC):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        (self.enc, self.actor, self.nm, self.nv, self.train_iter) = \
            load_policy(ckpt_path, self.device, decoder_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.material_wheel_control = policy_material_wheel_control(checkpoint)
        self.material_command_profile = policy_material_command_profile(checkpoint)
        self.material_aux_limit = (policy_material_aux_limit(checkpoint)
                                   if 'material_aux_limit' in checkpoint else 0.2)

    def act(self, obs_raw, deterministic=True):
        ob = torch.from_numpy(make_obs(obs_raw, self.nm, self.nv,
                                       self.device)).to(self.device)
        p = ob[:, :PROPRIO]
        e = ob[:, PROPRIO:PROPRIO + EXTERO]
        g = ob[:, PROPRIO + EXTERO:]
        with torch.no_grad():
            ao = torch.cat([p, self.enc(e), g], dim=1)
            dist = self.actor.dist(ao)
            action = dist.mean if deterministic else dist.sample()
        return action.cpu().numpy()[0]


def seg_of_x(x):
    return min(max(int(float(x) // 8.0), 0), 12)


def run_episode(env, runner, deterministic=True, max_steps=None, recorder=None,
                collect_traj=False):
    obs = env.reset()
    reset_ctx = getattr(getattr(runner, "actor", None), "reset_context", None)
    if callable(reset_ctx):
        reset_ctx()
    requested_pose = list(env.requested_initial_pose)
    settled_pose = list(env.settled_initial_pose)
    done = False
    total_rew = 0.0
    steps = 0
    max_x = float(env.data.qpos[0])
    max_pos = [float(value) for value in env.data.qpos[:3]]
    seg_max_x = {}
    trajectory = []
    info = {"reason": None, "success": False}
    max_steps = max_steps or env.episode_len
    while not done and steps < max_steps:
        segment = seg_of_x(env.data.qpos[0])
        if env.material_track is not None:
            env.vx_cmd = material_command_speed(steps, float(env.data.qpos[0]),
                                                 runner.material_command_profile)
            obs = env._get_obs()
        else:
            env.vx_cmd = SLOW_SPEED if segment in SLOW_SEGMENTS else NORMAL_SPEED
        action = runner.act(obs[None], deterministic)
        obs, reward, done, info = env.step(action)
        obs = env._get_obs()
        total_rew += float(reward)
        steps += 1
        position = [float(value) for value in env.data.qpos[:3]]
        x = position[0]
        if x >= max_x:
            max_x, max_pos = x, position
        segment = seg_of_x(x)
        seg_max_x[segment] = max(seg_max_x.get(segment, -float("inf")), x)
        if collect_traj:
            trajectory.append([x, position[1], position[2], float(env.gait_sel)])
        if recorder is not None:
            recorder(env)
    final_pos = [float(value) for value in env.data.qpos[:3]]
    furthest_index = max(seg_max_x, default=seg_of_x(final_pos[0]))
    reason = info.get("reason") or ("evaluation_limit" if not done else "unknown")
    result = {
        "success": bool(info.get("success", False)),
        "reason": reason,
        "total_rew": float(total_rew),
        "steps": int(steps),
        "max_x": float(max_x),
        "max_pos": max_pos,
        "final_x": final_pos[0],
        "final_pos": final_pos,
        "furthest_segment_index": int(furthest_index),
        "furthest_segment": SEQ[furthest_index],
        "seg_max_x": {str(k): float(v) for k, v in seg_max_x.items()},
        "terminal_contacts": info.get("terminal_contacts", []),
        "requested_initial_pose": requested_pose,
        "settled_initial_pose": settled_pose,
    }
    if collect_traj:
        result["trajectory"] = trajectory
    return result


def evaluation_config(args, evaluator, workers=1):
    return {
        "evaluator": evaluator,
        "checkpoint": os.path.realpath(args.ckpt),
        "decoder_path": os.path.realpath(getattr(args, "decoder_path", TVAE_DEC)),
        "decoder_sha256": _sha256(getattr(args, "decoder_path", TVAE_DEC)),
        "track": os.path.realpath(args.track),
        "episodes": args.episodes,
        "episode_len": args.episode_len,
        "workers": workers,
        "seed_base": args.seed,
        "seed_policy": "seed_base + episode_index",
        "eval_protocol": args.eval_protocol,
        "pose_v1_requested_ranges": POSE_V1_RANGES if args.eval_protocol == "pose-v1" else None,
        "device": getattr(args, "device", "auto"),
        "deterministic": bool(args.deterministic),
        "slow_segments": sorted(SLOW_SEGMENTS),
        "slow_speed_mps": SLOW_SPEED,
        "normal_speed_mps": NORMAL_SPEED,
        "lane_center_y_m": LANE_CENTER,
        "lane_half_width_m": LANE_HALF_WIDTH,
        "success_x_m": 100.0,
    }


def _create_manifest(args, evaluator, workers=1):
    heightmap, boxes, seq, meta = load_npz_track(args.track, apply_fixes=True)
    manifest = build_run_manifest(
        checkpoint=args.ckpt,
        track=args.track,
        config=evaluation_config(args, evaluator, workers),
        fixes=meta["fixes"],
        source_files=[
            __file__,
            os.path.join(ROOT, "common", "evaluation.py"),
            os.path.join(ROOT, "common", "eval_protocol.py"),
            os.path.join(ROOT, "common", "terrain.py"),
            os.path.join(ROOT, "stage3_rl", "env.py"),
            os.path.join(ROOT, "stage3_rl", "ppo_train.py"),
            os.path.join(ROOT, "REPORT.md"),
        ],
        track_summary=summarize_track(heightmap, boxes, seq),
    )
    write_run_manifest(args.out, manifest)
    return meta


def _append_episode(stream, result):
    stream.write(json.dumps(result, sort_keys=True) + "\n")
    stream.flush()


def _summary(results, ckpt_path, train_iter, fixes):
    successes = sum(result["success"] for result in results)
    failures = Counter(result["reason"] for result in results if not result["success"])
    furthest = Counter(result["furthest_segment"] for result in results)
    return dict(
        rate=successes / len(results), n_ok=successes, n=len(results),
        fails=dict(failures), furthest=dict(furthest), ckpt=ckpt_path,
        train_iter=train_iter, fixes=fixes,
        best_success_seed=next((r["seed"] for r in sorted(
            (item for item in results if item["success"]),
            key=lambda item: item["max_x"], reverse=True)), None),
    )


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.video and os.path.exists(args.video):
        raise FileExistsError(f"refusing to overwrite video: {args.video}")
    meta = _create_manifest(args, "serial")
    runner = PolicyRunner(args.ckpt,
                          device=None if args.device == "auto" else args.device,
                          decoder_path=args.decoder_path)
    print(f"loaded ckpt {args.ckpt} (train iter {runner.train_iter})")

    results = []
    episodes_path = os.path.join(args.out, "episodes.jsonl")
    with open(episodes_path, "x", encoding="utf-8") as episode_file:
        for index in range(args.episodes):
            seed = args.seed + index
            env = Go2WEnv(seed=seed, fixed_track=args.track, eval_mode=True,
                          episode_len=args.episode_len,
                          eval_protocol=args.eval_protocol,
                          decoder_path=args.decoder_path,
                          material_aux_limit=runner.material_aux_limit,
                          material_wheel_control=runner.material_wheel_control)
            result = run_episode(env, runner, deterministic=args.deterministic,
                                 max_steps=args.episode_len)
            result["seed"] = seed
            results.append(result)
            _append_episode(episode_file, result)
            print(f"ep {index:3d}: {'SUCCESS' if result['success'] else 'FAIL'} "
                  f"({result['reason']}) x_max={result['max_x']:.1f} "
                  f"steps={result['steps']} rew={result['total_rew']:.1f}",
                  flush=True)

    summary = _summary(results, args.ckpt, runner.train_iter, meta["fixes"])
    summary_path = os.path.join(args.out, "eval_summary.json")
    with open(summary_path, "x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"\n=== completion rate: {summary['n_ok']}/{summary['n']} = "
          f"{summary['rate']:.1%} ===")
    print("failures:", summary["fails"])
    print("furthest segment:", summary["furthest"])
    print("saved manifest.json, episodes.jsonl, and eval_summary.json")

    if args.video and args.record_episodes:
        successful = [result for result in results if result["success"]]
        if not successful:
            print("no successful episode; video was not written")
        else:
            selected = max(successful, key=lambda item: item["max_x"])
            env = Go2WEnv(seed=selected["seed"], fixed_track=args.track,
                          eval_mode=True, episode_len=args.episode_len,
                          eval_protocol=args.eval_protocol,
                          decoder_path=args.decoder_path,
                          material_aux_limit=runner.material_aux_limit,
                          material_wheel_control=runner.material_wheel_control)
            recorder = VideoRecorder(env, args.video)
            recorder.start()
            replay = run_episode(env, runner, deterministic=args.deterministic,
                                 max_steps=args.episode_len, recorder=recorder)
            matches = (replay["success"] and
                       replay["reason"] == selected["reason"] and
                       abs(replay["max_x"] - selected["max_x"]) <= 1e-6)
            recorder.finish(matches)
            print(f"recorded successful seed {selected['seed']} to {args.video}")
    return summary


class VideoRecorder:
    """Offscreen camera-follow recorder piping frames to ffmpeg."""

    def __init__(self, env, path, fps=50, height=720, width=1280):
        import subprocess
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite video: {path}")
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self.env = env
        self.path = path
        self.fps = fps
        self.height, self.width = height, width
        self.proc = subprocess.Popen(
            ["ffmpeg", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        import mujoco
        self.renderer = mujoco.Renderer(env.model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.active = False
        self.frame_every = 2

    def start(self):
        self.active = True
        self.n = 0

    def __call__(self, env):
        if not self.active:
            return
        self.n += 1
        if self.n % self.frame_every:
            return
        x, y, z = env.data.xpos[env.body_root]
        self.cam.lookat[:] = [x, y, z]
        self.cam.distance = 3.2
        self.cam.elevation = -15
        self.cam.azimuth = 110
        self.renderer.update_scene(env.data, self.cam)
        self.proc.stdin.write(self.renderer.render().tobytes())

    def finish(self, success):
        if not self.active:
            return
        self.proc.stdin.close()
        # Drain stderr before wait: ffmpeg can block when its pipe fills.
        error_output = self.proc.stderr.read()
        self.proc.stderr.close()
        return_code = self.proc.wait()
        self.active = False
        if return_code != 0:
            error = error_output.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed with {return_code}: {error}")
        if not success:
            if os.path.exists(self.path):
                os.unlink(self.path)
            raise RuntimeError("video replay did not reproduce a successful episode")
        print("video saved", flush=True)


if __name__ == "__main__":
    main()
