"""采集砖块段（27.35m 起、带横向偏移）的教师示范，用于教"横向纠偏 + 过砖"。

为什么需要：学生到 27.6m 后经常横向漂到 bypass 线或撞砖；示范里教师总是从正中出发，
没有"已经偏了怎么回来"的样本。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(r"E:\desktop-rl-apl\aptrl_go2w")
TMP = Path(r"C:\Users\Administrator\AppData\Local\Temp\opencode")
sys.path.insert(0, str(TMP))
sys.path.insert(0, str(REPO))

TRACK = REPO / "results/diagnostics/robocon-material-preview-20261119-v5/scene.xml"
DEC = REPO / "stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz"
X0, Y0S = 27.35, (3.62, 3.75, 3.88)


def _init(x0, ys):
    global X0, Y0S
    X0 = x0
    Y0S = tuple(float(v) for v in ys)


def one(args):
    seed, y0 = args
    from exact_edge_v4_expert import ExactEdgeObsExpert
    from eval_v4_multiseed_now import load_merged_params
    from common.eval_protocol import material_command_speed
    from stage3_rl.env import Go2WEnv

    ex = ExactEdgeObsExpert(load_merged_params())
    env = Go2WEnv(seed=seed, fixed_track=str(TRACK), eval_mode=True, episode_len=9000,
                  eval_protocol="pose-v1", decoder_path=str(DEC),
                  material_aux_limit=6.0, material_wheel_control="residual_v1",
                  material_eval_start=21.5)
    d = env.data
    env.reset()
    d.qpos[0] = float(X0)
    d.qpos[1] = float(y0)
    d.qvel[:] = 0.0
    mujoco.mj_forward(env.model, d)
    O, A = [], []
    reason = "max_steps"
    for st in range(1400):
        x = float(d.qpos[0])
        env.vx_cmd = material_command_speed(st, x, "start_ramp_100_v1")
        obs = env._get_obs()
        a = ex.act(obs[None], True)
        if x >= 27.0:
            O.append(obs.astype(np.float32)); A.append(a.astype(np.float32))
        obs, _r, done, info = env.step(a)
        if done:
            reason = str(info.get("reason")); break
        if float(d.qpos[0]) >= 31.2:
            reason = "past_stairs"; break
    return seed, y0, (np.stack(O) if O else None), (np.stack(A) if A else None), \
        float(d.qpos[0]), reason


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", type=int, default=9400)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--y0", default="3.62,3.75,3.88")
    ap.add_argument("--min-x", type=float, default=28.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ys = [float(v) for v in args.y0.split(",")]
    mp.freeze_support()
    O, A, E = [], [], []
    tasks = [(args.seed0 + i, ys[i % len(ys)]) for i in range(args.seeds)]
    with mp.get_context("spawn").Pool(8, initializer=_init, initargs=(X0, ys)) as pool:
        for seed, y0, ob, ac, mx, rsn in pool.imap_unordered(one, tasks):
            print(f"  seed={seed} y0={y0:.2f} max_x={mx:.2f} labels={0 if ob is None else len(ob)} reason={rsn}", flush=True)
            if ob is not None and mx >= args.min_x:
                O.append(ob); A.append(ac)
                E.append(np.full(len(ob), seed * 10 + int(round(y0 * 100)), np.int64))
    if not O:
        print("no episodes"); sys.exit(1)
    np.savez_compressed(args.out, obs=np.concatenate(O), act=np.concatenate(A), ep=np.concatenate(E))
    print(f"saved {len(O)} eps, {sum(len(o) for o in O)} steps -> {args.out}")
