"""轮胎段 DAgger：用当前 BC 策略自己跑，在每个状态上标注脚本教师的动作，再回归。

解决 BC 的分布漂移：BC 只在教师轨迹上见过"完美对位"的状态，
一旦自己走偏（轮子顶到胎壁）就没有纠正样本，于是直接撞上去。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"E:\desktop-rl-apl\aptrl_go2w")
TMP = Path(r"C:\Users\Administrator\AppData\Local\Temp\opencode")
sys.path.insert(0, str(TMP))
sys.path.insert(0, str(REPO))
sys.path.insert(0, r"E:\desktop-rl-apl\rl_finetune_v5")

TRACK = REPO / "results/diagnostics/robocon-material-preview-20261119-v5/scene.xml"
DEC = REPO / "stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz"
LO, HI = 21.0, 28.2


def rollout_labels(ckpt, seeds, alpha_noise=0.0):
    from exact_edge_v4_expert import ExactEdgeObsExpert
    from eval_v4_multiseed_now import load_merged_params
    from common.eval_protocol import material_command_speed
    from stage3_rl.env import Go2WEnv
    from finetune_full_course import build_from_checkpoint, EXTERO, PROPRIO

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck, arch, passthrough, enc, actor, nm, ns = build_from_checkpoint(Path(ckpt), dev)
    teacher = ExactEdgeObsExpert(load_merged_params())
    obs_all, act_all, ep_all = [], [], []
    for seed in seeds:
        env = Go2WEnv(seed=seed, fixed_track=str(TRACK), eval_mode=True, episode_len=9000,
                      eval_protocol="pose-v1", decoder_path=str(DEC),
                      material_aux_limit=6.0, material_wheel_control="residual_v1")
        d = env.data
        actor.reset_context()
        env.reset()
        O, A = [], []
        for st in range(9000):
            x = float(d.qpos[0])
            env.vx_cmd = material_command_speed(st, x, "start_ramp_100_v1")
            obs = env._get_obs()
            ob_n = np.clip((obs - nm) / ns, -10, 10).astype(np.float32)
            ob_t = torch.from_numpy(ob_n[None]).to(dev)
            with torch.no_grad():
                ao = torch.cat([ob_t[:, :PROPRIO], enc(ob_t[:, PROPRIO:PROPRIO + EXTERO]),
                                ob_t[:, PROPRIO + EXTERO:]], dim=1)
                a = actor.dist(ao).mean.cpu().numpy()[0].astype(np.float64)
            if alpha_noise > 0:
                a = a + np.random.default_rng(seed * 1000 + st).normal(0, alpha_noise, a.shape)
            lab = teacher.act(obs[None], True)
            if LO <= x <= HI:
                O.append(obs); A.append(lab)
            obs, _, done, info = env.step(a)
            if done:
                break
        print(f"  seed={seed} steps={st} max_x={float(d.qpos[0]):.2f} labels={len(O)}", flush=True)
        if len(O) >= 80:
            obs_all.append(np.stack(O)); act_all.append(np.stack(A))
            ep_all.append(np.full(len(O), seed, np.int64))
    return obs_all, act_all, ep_all


def merge(paths, out):
    O, A, E, off = [], [], [], 0
    for p in paths:
        if not Path(p).exists():
            continue
        ds = np.load(p)
        eps = ds["ep"]
        uniq = {u: i + off for i, u in enumerate(np.unique(eps))}
        off += len(uniq)
        O.append(ds["obs"]); A.append(ds["act"])
        E.append(np.array([uniq[u] for u in eps], np.int64))
    np.savez_compressed(out, obs=np.concatenate(O), act=np.concatenate(A), ep=np.concatenate(E))
    print(f"merged {sum(len(o) for o in O)} steps -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out-dir", default=r"E:\desktop-rl-apl\rl_rsl\runs\tire_bc_dagger")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ck = Path(args.ckpt)
    base = r"E:\desktop-rl-apl\rl_rsl\data\tire_expert_v5_approach.npz"
    sets = [base]
    for r in range(1, args.rounds + 1):
        print(f"[dagger round {r}] labels from BC rollout ({ck})", flush=True)
        seeds = [8900 + i for i in range(args.seeds)]
        O, A, E = rollout_labels(ck, seeds, args.noise)
        if O:
            npz = out_dir / f"dagger_r{r}.npz"
            np.savez_compressed(npz, obs=np.concatenate(O), act=np.concatenate(A),
                                ep=np.concatenate(E))
            sets.append(str(npz))
        merged = out_dir / f"merged_r{r}.npz"
        merge(sets, merged)
        new_ck = out_dir / f"model_bc_r{r}.pt"
        cmd = [sys.executable, r"E:\desktop-rl-apl\rl_rsl\bc_temporal_tire.py",
               "--data", str(merged), "--out", str(new_ck), "--init", str(ck),
               "--epochs", str(args.epochs)]
        print("  retrain:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        ck = new_ck
        print(f"[dagger round {r}] -> {ck}", flush=True)


if __name__ == "__main__":
    main()
