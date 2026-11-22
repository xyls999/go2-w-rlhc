"""采集脚本教师**全程**示范（0.5m -> 尽量远），用于全程 BC。"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"E:\desktop-rl-apl\aptrl_go2w")
TMP = Path(r"C:\Users\Administrator\AppData\Local\Temp\opencode")
sys.path.insert(0, str(TMP))
sys.path.insert(0, str(REPO))

TRACK = REPO / "results/diagnostics/robocon-material-preview-20261119-v5/scene.xml"
DEC = REPO / "stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz"


START = None


def _init(start):
    global START
    START = start


def one(seed):
    from exact_edge_v4_expert import ExactEdgeObsExpert
    from eval_v4_multiseed_now import load_merged_params
    from common.eval_protocol import material_command_speed
    from stage3_rl.env import Go2WEnv

    ex = ExactEdgeObsExpert(load_merged_params())
    kw = {"material_eval_start": float(START)} if START is not None else {}
    env = Go2WEnv(seed=seed, fixed_track=str(TRACK), eval_mode=True, episode_len=9000,
                  eval_protocol="pose-v1", decoder_path=str(DEC),
                  material_aux_limit=6.0, material_wheel_control="residual_v1", **kw)
    d = env.data
    env.reset()
    O, A, X = [], [], []
    reason = "max_steps"
    for st in range(9000):
        x = float(d.qpos[0])
        env.vx_cmd = material_command_speed(st, x, "start_ramp_100_v1")
        obs = env._get_obs()
        a = ex.act(obs[None], True)
        O.append(obs.astype(np.float32)); A.append(a.astype(np.float32)); X.append(x)
        obs, _r, done, info = env.step(a)
        if done:
            reason = str(info.get("reason")); break
    return seed, np.stack(O), np.stack(A), float(d.qpos[0]), reason, st, np.asarray(X, np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", type=int, default=9100)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--min-x", type=float, default=27.6)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mp.freeze_support()
    O, A, E, XS = [], [], [], []
    with mp.get_context("spawn").Pool(8, initializer=_init, initargs=(args.start,)) as pool:
        for seed, ob, ac, mx, rsn, st, xs in pool.imap_unordered(one, range(args.seed0, args.seed0 + args.seeds)):
            print(f"  seed={seed} max_x={mx:.2f} steps={st} reason={rsn}", flush=True)
            if mx < args.min_x:
                continue
            O.append(ob); A.append(ac); XS.append(xs)
            E.append(np.full(len(ob), seed, np.int64))
    if not O:
        print("no episodes"); sys.exit(1)
    np.savez_compressed(args.out, obs=np.concatenate(O), act=np.concatenate(A),
                        ep=np.concatenate(E), x=np.concatenate(XS))
    print(f"saved {len(O)} eps, {sum(len(o) for o in O)} steps -> {args.out}")
