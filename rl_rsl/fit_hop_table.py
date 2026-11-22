"""闭式拟合起跳相位表：table[p_idx, hop_el] = 教师在该相位的**绝对动作**均值。

关键点：教师的起跳是"开环"的（只看 x 与自身相位计数），所以用相位表直接替换
策略在该相位的输出，就能复现脉冲，不受策略自身姿态分布漂移影响。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"E:\desktop-rl-apl\rl_finetune_v5")
from finetune_full_course import build_from_checkpoint, precompute_kin, PROPRIO, EXTERO  # noqa: E402
from stage3_rl.ppo_train import GAIT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", type=float, default=3.0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck, arch, past, enc, actor, nm, ns = build_from_checkpoint(Path(args.ckpt), dev)
    _, _, _, enc_cpu, actor_cpu, _, _ = build_from_checkpoint(Path(args.ckpt), "cpu")
    ds = np.load(args.data)
    obs_all, act_all, ep_all = ds["obs"], ds["act"], ds["ep"]
    eps, start = [], 0
    for i in range(1, len(obs_all) + 1):
        if i == len(obs_all) or ep_all[i] != ep_all[start]:
            eps.append((obs_all[start:i], act_all[start:i])); start = i

    sums = np.zeros((6, 96, 29), np.float64)
    sumsw = np.zeros((6, 96, 1), np.float64)
    cnts = np.zeros((6, 96), np.float64)
    L = PROPRIO + 64 + GAIT + actor.KIN_DIM
    for ob, ac in eps:
        ob_n = np.clip((ob - nm) / ns, -10, 10).astype(np.float32)
        kin = precompute_kin(actor_cpu, ob_n)
        kin_np = kin
        pid = np.clip(np.round(kin_np[:, 26 + 16] * 5.0), 0, 5).astype(int)
        pm = kin_np[:, 26 + 17]
        hel = np.clip(np.round(kin_np[:, 26 + 18] * 30.0), 0, 95).astype(int)
        m = pm > 0.5
        for j in np.nonzero(m)[0]:
            sums[pid[j], hel[j]] += ac[j, :29]
            sumsw[pid[j], hel[j]] += ac[j, 29:30]
            cnts[pid[j], hel[j]] += 1

    filled = cnts > 0
    tab = np.zeros_like(sums)
    tabw = np.zeros_like(sumsw)
    tab[filled] = sums[filled] / cnts[filled][:, None]
    tabw[filled] = sumsw[filled] / cnts[filled][:, None]
    tab = np.clip(tab, -args.clip, args.clip)
    tabw = np.clip(tabw, -args.clip, args.clip)
    print(f"filled rows={int(filled.sum())}/576, max|aux|={np.abs(tab).max():.3f}, "
          f"max|wheel|={np.abs(tabw).max():.3f}")

    with torch.no_grad():
        actor.hop_table.copy_(torch.from_numpy(tab).to(actor.hop_table.device))
        actor.hop_wheel.copy_(torch.from_numpy(tabw).to(actor.hop_wheel.device))
    state = {"enc": enc.state_dict(), "actor": actor.state_dict(),
             "critic": ck.get("critic", {}), "norm": ck["norm"], "iter": int(ck.get("iter", 0)),
             "actor_arch": arch, "extero_passthrough": past,
             "actor_hidden": ck.get("actor_hidden", 768), "gru_hidden": ck.get("gru_hidden", 256),
             "decoder_path": ck.get("decoder_path"), "decoder_sha256": ck.get("decoder_sha256"),
             "material_aux_limit": 6.0, "material_wheel_control": "residual_v1",
             "material_command_profile": "start_ramp_100_v1"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
