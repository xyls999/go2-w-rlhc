"""并行评测：多 seed、确定性（均值动作）跑全程评估点，统计各里程碑通过率。"""
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
STOCH = False


def _init(ck, stoch):
    global CK, STOCH
    CK = ck; STOCH = stoch


def one(seed):
    from finetune_full_course import contact_names
    ck, arch, passthrough, enc, actor, nm, ns = build_from_checkpoint(Path(CK), "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    env = Go2WEnv(seed=seed, fixed_track=str(TRACK), eval_mode=True, episode_len=9000,
                  eval_protocol="pose-v1", decoder_path=str(DEC),
                  material_aux_limit=6.0, material_wheel_control="residual_v1")
    d = env.data
    wheels = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot_0")
              for l in ("FL", "FR", "RL", "RR")]
    env.reset()
    route = MaterialRouteMonitor()
    actor.reset_context()
    maxx = 0.0
    reason = "max_steps"
    ymax = 0.0
    for st in range(9000):
        x = float(d.qpos[0])
        env.vx_cmd = material_command_speed(st, x, "start_ramp_100_v1")
        obs = env._get_obs()
        ob_n = np.clip((obs - nm) / ns, -10, 10).astype(np.float32)
        ob_t = torch.from_numpy(ob_n[None])
        with torch.no_grad():
            ao = torch.cat([ob_t[:, :PROPRIO], enc(ob_t[:, PROPRIO:PROPRIO + EXTERO]),
                            ob_t[:, PROPRIO + EXTERO:]], dim=1)
            dist = actor.dist(ao)
            a = (dist.sample() if STOCH else dist.mean).numpy()[0].astype(np.float64)
        obs, _r, done, info = env.step(a)
        route.update(float(d.qpos[0]), float(d.qpos[1]), d.geom_xpos[wheels, :2],
                     contact_names(env.model, d), frame_overlap_top(env.model, d, 44.98, 45.02))
        maxx = max(maxx, float(d.qpos[0]))
        if abs(float(d.qpos[1]) - 4.0) > 1.5:
            ymax = max(ymax, abs(float(d.qpos[1]) - 4.0))
        if route.violations:
            reason = ",".join(sorted(route.violations)); break
        if done:
            reason = str(info.get("reason")); break
        if maxx >= 100.0:
            reason = "finish"; break
    return seed, round(maxx, 2), reason, round(ymax, 2), st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed0", type=int, default=8800)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--stoch", action="store_true")
    args = ap.parse_args()
    globals()["CK"] = args.ckpt
    globals()["STOCH"] = args.stoch
    mp.freeze_support()
    marks = [25.4, 26.4, 27.6, 29.8, 30.4, 33.5, 40.0, 100.0]
    res = []
    with mp.get_context("spawn").Pool(8, initializer=_init, initargs=(CK, STOCH)) as pool:
        for seed, maxx, reason, ymax, st in pool.imap_unordered(one, range(args.seed0, args.seed0 + args.seeds)):
            res.append((seed, maxx, reason))
            print(f"  seed={seed} max_x={maxx} reason={reason} steps={st} ydev={ymax}", flush=True)
    n = len(res)
    print(f"\n== {Path(args.ckpt).name} on {n} seeds (stoch={STOCH}) ==")
    for m in marks:
        c = sum(1 for _, mx, _ in res if mx >= m)
        print(f"  >= {m:6.1f} m : {c}/{n}")
    from collections import Counter
    print("  reasons:", dict(Counter(r for _, _, r in res)))
    print("  max_x mean=%.2f median=%.2f" % (np.mean([r[1] for r in res]), np.median([r[1] for r in res])))
