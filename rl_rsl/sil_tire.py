"""自模仿学习（SIL）：用带噪声的当前策略多跑，只保留"真的越过轮胎"的回合，
用这些**自己采样出来的**动作做监督——动作天然落在策略分布内，确定性执行时更容易复现。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"E:\desktop-rl-apl\rl_finetune_v5")
sys.path.insert(0, r"E:\desktop-rl-apl\aptrl_go2w")
from finetune_full_course import build_from_checkpoint, EXTERO, PROPRIO, DEC, TRACK  # noqa: E402
from common.eval_protocol import material_command_speed  # noqa: E402
from common.material_route_monitor import MaterialRouteMonitor, frame_overlap_top  # noqa: E402
from stage3_rl.env import Go2WEnv  # noqa: E402
import mujoco  # noqa: E402

CK = None
THRESH = 26.4
XLO, XHI = 21.0, 30.5


def _init(ck, thresh):
    global CK, THRESH
    CK = ck; THRESH = thresh


def one(seed):
    from finetune_full_course import contact_names
    dev = "cpu"
    ck, arch, passthrough, enc, actor, nm, ns = build_from_checkpoint(Path(CK), dev)
    torch.manual_seed(seed); np.random.seed(seed)
    env = Go2WEnv(seed=seed, fixed_track=str(TRACK), eval_mode=True, episode_len=9000,
                  eval_protocol="pose-v1", decoder_path=str(DEC),
                  material_aux_limit=6.0, material_wheel_control="residual_v1",
                  material_eval_start=21.5)
    d = env.data
    wheels = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot_0")
              for l in ("FL", "FR", "RL", "RR")]
    obs = env.reset()
    route = MaterialRouteMonitor()
    route.update(float(d.qpos[0]), float(d.qpos[1]), d.geom_xpos[wheels, :2],
                 contact_names(env.model, d), frame_overlap_top(env.model, d, 44.98, 45.02))
    route.started = True
    route.violations.discard("invalid_start")
    actor.reset_context()
    O, A = [], []
    best = 0.0
    reason = "max_steps"
    for st in range(1800):
        x = float(d.qpos[0])
        env.vx_cmd = material_command_speed(st, x, "start_ramp_100_v1")
        obs = env._get_obs()
        ob_n = np.clip((obs - nm) / ns, -10, 10).astype(np.float32)
        ob_t = torch.from_numpy(ob_n[None])
        with torch.no_grad():
            ao = torch.cat([ob_t[:, :PROPRIO], enc(ob_t[:, PROPRIO:PROPRIO + EXTERO]),
                            ob_t[:, PROPRIO + EXTERO:]], dim=1)
            a = actor.dist(ao).sample().numpy()[0].astype(np.float64)
        if XLO <= x <= XHI:
            O.append(obs); A.append(a)
        obs, _r, done, info = env.step(a)
        route.update(float(d.qpos[0]), float(d.qpos[1]), d.geom_xpos[wheels, :2],
                     contact_names(env.model, d), frame_overlap_top(env.model, d, 44.98, 45.02))
        best = max(best, float(d.qpos[0]))
        if route.violations:
            reason = ",".join(sorted(route.violations)); break
        if done:
            reason = str(info.get("reason")); break
        if best >= 31.0:
            reason = "past_tire"; break
    return seed, (np.stack(O) if O else None), (np.stack(A) if A else None), best, reason


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed0", type=int, default=8500)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--thresh", type=float, default=26.4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    globals()["CK"] = args.ckpt
    globals()["THRESH"] = args.thresh
    CK = args.ckpt
    THRESH = args.thresh
    mp.freeze_support()
    O, A, E = [], [], []
    keep = 0
    with mp.get_context("spawn").Pool(8, initializer=_init, initargs=(CK, THRESH)) as pool:
        for seed, ob, ac, best, rsn in pool.imap_unordered(
                one, list(range(args.seed0, args.seed0 + args.seeds))):
            ok = best >= args.thresh
            print(f"  seed={seed} best={best:.2f} reason={rsn} keep={ok}", flush=True)
            if ok and ob is not None:
                O.append(ob); A.append(ac)
                E.append(np.full(len(ob), seed, np.int64))
                keep += 1
    if not O:
        print("no successful episodes"); sys.exit(1)
    np.savez_compressed(args.out, obs=np.concatenate(O), act=np.concatenate(A),
                        ep=np.concatenate(E))
    print(f"saved {keep}/{args.seeds} successful eps, {sum(len(o) for o in O)} steps -> {args.out}")
