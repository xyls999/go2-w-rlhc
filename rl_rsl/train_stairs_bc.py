"""在楼梯段（30.0-31.8m）切片上做高权重 BC，专门把起跳动作拟合进网络。

每个切片用其真实起点做 kin 的 x 对齐（prime_x0），避免 x 特征错位。
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
    ap.add_argument("--lo", type=float, default=30.0)
    ap.add_argument("--hi", type=float, default=31.8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck, arch, past, enc, actor, nm, ns = build_from_checkpoint(Path(args.ckpt), dev)
    _, _, _, enc_cpu, actor_cpu, _, _ = build_from_checkpoint(Path(args.ckpt), "cpu")
    ds = np.load(args.data)
    obs, act, ep, xs = ds["obs"], ds["act"], ds["ep"], ds["x"]

    # 切出每回合的楼梯段
    slices = []
    order = np.argsort(ep, kind="stable")
    obs, act, ep, xs = obs[order], act[order], ep[order], xs[order]
    bounds = np.flatnonzero(np.diff(ep)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(ep)]])
    for s, e in zip(starts, ends):
        m = (xs[s:e] >= args.lo) & (xs[s:e] <= args.hi)
        idx = np.flatnonzero(m)
        if len(idx) < 40:
            continue
        a, b = s + idx[0], s + idx[-1] + 1
        slices.append((obs[a:b], act[a:b], float(xs[a]), float(xs[b - 1])))
    print(f"stairs slices={len(slices)} steps={sum(len(s[0]) for s in slices)}")

    parsed = []
    for ob, ac, x0, x1 in slices:
        ob_n = np.clip((ob - nm) / ns, -10, 10).astype(np.float32)
        kin = precompute_kin(actor_cpu, ob_n, x0=x0)
        parsed.append((torch.from_numpy(ob_n), torch.from_numpy(kin),
                       torch.from_numpy(ac.astype(np.float32))))
        if np.isnan(kin).any():
            print("warn: nan kin", x0, x1)

    w = torch.ones(29, device=dev)
    w[16:28] = 3.0
    w[28] = 3.0
    opt = torch.optim.AdamW([p for p in list(enc.parameters()) + list(actor.parameters()) if p.requires_grad],
                            lr=args.lr)
    L = PROPRIO + 64 + GAIT + actor.KIN_DIM
    for ep_i in range(1, args.epochs + 1):
        tot, cnt = 0.0, 0
        for ob_n, kin, ac in parsed:
            T = len(ac)
            ob_t = ob_n.to(dev); kin_t = kin.to(dev)
            flat = ob_t.reshape(-1, 426)
            ext = enc(flat[:, PROPRIO:PROPRIO + EXTERO])[:, :96]
            ao = torch.cat([flat[:, :PROPRIO], ext, flat[:, PROPRIO + EXTERO:]],
                           -1).reshape(1, T, PROPRIO + 96 + GAIT)
            base_in = torch.cat([ao[..., :PROPRIO], ao[..., PROPRIO:PROPRIO + 32],
                                 ao[..., PROPRIO + 96:]], -1)
            res_in = torch.cat([ao[..., :PROPRIO], ao[..., PROPRIO + 32:PROPRIO + 96],
                                ao[..., PROPRIO + 96:], kin_t.unsqueeze(0)], -1)
            K = actor.HIST_K
            pad = res_in[:, 0:1, :].repeat(1, K - 1, 1)
            ring = torch.cat([pad, res_in], dim=1).unfold(1, K, 1).permute(0, 1, 3, 2).reshape(1, T, K * L)
            mu, _, _ = actor.forward_seq(base_in, res_in, ring)
            loss = ((((mu[0, :, :29] - ac.to(dev)[:, :29]) ** 2) * w).sum()) / (w.sum() * T)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(enc.parameters()) + list(actor.parameters()) if p.requires_grad], 1.0)
            opt.step()
            tot += float(loss.detach()) * T; cnt += T
        if ep_i % 2 == 0 or ep_i == 1:
            print(f"  epoch {ep_i}: mse={tot / max(1, cnt):.5f}", flush=True)

    state = {"enc": enc.state_dict(), "actor": actor.state_dict(), "critic": ck.get("critic", {}),
             "norm": ck["norm"], "iter": int(ck.get("iter", 0)), "actor_arch": arch,
             "extero_passthrough": past, "actor_hidden": ck.get("actor_hidden", 768),
             "gru_hidden": ck.get("gru_hidden", 256), "decoder_path": ck.get("decoder_path"),
             "decoder_sha256": ck.get("decoder_sha256"), "material_aux_limit": 6.0,
             "material_wheel_control": "residual_v1",
             "material_command_profile": "start_ramp_100_v1"}
    torch.save(state, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
