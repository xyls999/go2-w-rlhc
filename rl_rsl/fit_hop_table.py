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

    sums = np.zeros((6, 130, 29), np.float64)
    csums = np.zeros((6, 40, 29), np.float64)
    csumsw = np.zeros((6, 40, 1), np.float64)
    ccnts = np.zeros((6, 40), np.float64)
    bsums = np.zeros((165, 29), np.float64)
    bsumsw = np.zeros((165, 1), np.float64)
    bcnts = np.zeros(165, np.float64)
    sumsw = np.zeros((6, 130, 1), np.float64)
    cnts = np.zeros((6, 130), np.float64)
    L = PROPRIO + 64 + GAIT + actor.KIN_DIM
    for ob, ac in eps:
        ob_n = np.clip((ob - nm) / ns, -10, 10).astype(np.float32)
        kin = precompute_kin(actor_cpu, ob_n)
        kin_np = kin
        pid = np.clip(np.round(kin_np[:, 26 + 16] * 5.0), 0, 5).astype(int)
        pm = kin_np[:, 26 + 17]
        hel = np.clip(np.round((kin_np[:, 0] * 15.0 + 25.0 - 30.40) / 0.02), 0, 129).astype(int)
        sg5 = kin_np[:, 26 + 5]
        xest = kin_np[:, 0] * 15.0 + 25.0
        ph = (np.arange(len(kin_np)) * 0.01 * 4.0) % 1.0
        cb = np.clip((ph * 40.0).astype(int), 0, 39)
        sg3 = kin_np[:, 26 + 3]; sg4 = kin_np[:, 26 + 4]
        bb = np.clip(np.round((xest - 27.30) / 0.02), 0, 164).astype(int)
        mb = (sg3 > 0.5) | (sg4 > 0.5)
        for j in np.nonzero(mb)[0]:
            bsums[bb[j]] += ac[j, :29]
            bsumsw[bb[j]] += ac[j, 29:30]
            bcnts[bb[j]] += 1
        mc = (sg5 > 0.5) & (pm <= 0.5) & (xest >= 30.10) & (xest <= 30.68)
        for j in np.nonzero(mc)[0]:
            csums[pid[j], cb[j]] += ac[j, :29]
            csumsw[pid[j], cb[j]] += ac[j, 29:30]
            ccnts[pid[j], cb[j]] += 1
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

    cfilled = ccnts > 0
    ctab = np.zeros_like(csums)
    ctabw = np.zeros_like(csumsw)
    ctab[cfilled] = csums[cfilled] / ccnts[cfilled][:, None]
    ctabw[cfilled] = csumsw[cfilled] / ccnts[cfilled][:, None]
    ctab = np.clip(ctab, -args.clip, args.clip)
    ctabw = np.clip(ctabw, -args.clip, args.clip)
    print(f"creep rows={int(cfilled.sum())}/240, max|aux|={np.abs(ctab).max():.3f}")
    bfilled = bcnts > 0
    btab = np.zeros_like(bsums); btabw = np.zeros_like(bsumsw)
    btab[bfilled] = bsums[bfilled] / bcnts[bfilled][:, None]
    btabw[bfilled] = bsumsw[bfilled] / bcnts[bfilled][:, None]
    btab = np.clip(btab, -args.clip, args.clip); btabw = np.clip(btabw, -args.clip, args.clip)
    print(f"brick rows={int(bfilled.sum())}/165, max|aux|={np.abs(btab[:, 16:28]).max():.3f}")
    with torch.no_grad():
        actor.hop_table.copy_(torch.from_numpy(tab).to(actor.hop_table.device))
        actor.hop_wheel.copy_(torch.from_numpy(tabw).to(actor.hop_wheel.device))
        actor.creep_table.copy_(torch.from_numpy(ctab).to(actor.creep_table.device))
        actor.creep_wheel.copy_(torch.from_numpy(ctabw).to(actor.creep_wheel.device))
        actor.hop_mask.copy_(torch.from_numpy(filled.astype(np.float32)).to(actor.hop_mask.device))
        actor.creep_mask.copy_(torch.from_numpy(cfilled.astype(np.float32)).to(actor.creep_mask.device))
        actor.brick_table.copy_(torch.from_numpy(btab).to(actor.brick_table.device))
        actor.brick_wheel.copy_(torch.from_numpy(btabw).to(actor.brick_wheel.device))
        actor.brick_mask.copy_(torch.from_numpy(bfilled.astype(np.float32)).to(actor.brick_mask.device))
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
