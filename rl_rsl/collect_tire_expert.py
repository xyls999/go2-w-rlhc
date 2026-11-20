"""采集"轮胎段"专家示范（同一 30 维动作空间），用于引导 RL 学会跳上轮胎。

为什么可行：脚本教师输出的就是本环境的 30 维动作（latent+aux+gait+wheel），
不存在"位置控制策略无法映射"的问题；只要状态对齐，示范就是可直接监督的目标。

做法：从起点跑完整回合（教师过轮胎 24/24），只截取 x ∈ [22.8, 27.8] 的 (obs, act) 片段。
"""
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
X_LO, X_HI = 21.0, 27.8


def one(seed):
    from exact_edge_v4_expert import ExactEdgeObsExpert
    from eval_v4_multiseed_now import load_merged_params
    from common.eval_protocol import material_command_speed
    from stage3_rl.env import Go2WEnv

    params = load_merged_params()
    env = Go2WEnv(seed=seed, fixed_track=str(TRACK), eval_mode=True, episode_len=9000,
                  eval_protocol="pose-v1", decoder_path=str(DEC),
                  material_aux_limit=6.0, material_wheel_control="residual_v1")
    d = env.data
    runner = ExactEdgeObsExpert(params)
    env.reset()
    obs_l, act_l = [], []
    for st in range(9000):
        x = float(d.qpos[0])
        env.vx_cmd = material_command_speed(st, x, "start_ramp_100_v1")
        obs = env._get_obs()
        a = runner.act(obs[None], True)
        if X_LO <= x <= X_HI:
            obs_l.append(obs); act_l.append(a)
        obs, _, done, info = env.step(a)
        if done:
            break
    max_x = float(d.qpos[0])
    if len(obs_l) >= 100:
        return seed, np.stack(obs_l), np.stack(act_l), max_x, info.get("reason")
    return seed, None, None, max_x, "too_few"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=8800)
    ap.add_argument("--out", default=r"E:\desktop-rl-apl\rl_rsl\data\tire_expert_v5.npz")
    args = ap.parse_args()
    mp.freeze_support()
    obs_all, act_all, ep_all, x_all = [], [], [], []
    with mp.get_context("spawn").Pool(8) as pool:
        for seed, ob, ac, mx, rsn in pool.imap_unordered(one, list(range(args.seed0, args.seed0 + args.seeds))):
            if ob is None:
                print(f"  skip seed={seed} max_x={mx:.2f} rsn={rsn}", flush=True)
                continue
            obs_all.append(ob); act_all.append(ac)
            ep_all.append(np.full(len(ob), seed, np.int64))
            print(f"  ok seed={seed} tire_steps={len(ob)} max_x={mx:.2f}", flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=np.concatenate(obs_all), act=np.concatenate(act_all),
                        ep=np.concatenate(ep_all))
    print(f"saved {sum(len(o) for o in obs_all)} tire steps from {len(obs_all)} seeds -> {out}")
