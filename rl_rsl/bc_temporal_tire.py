"""在"带相位可观测"的时序 actor（temporal_residual_v4）上用轮胎示范做 BC。

为什么这次能成：该 actor 输入里含我加的 FSM 相位特征（触发步/hopping 帧号/阶段），
"同一位置不同相位要输出不同动作"这一多值问题被显式解开，监督回归不再是"平均化"。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"E:\desktop-rl-apl\rl_finetune_v5")
from finetune_full_course import (REPO, build_from_checkpoint, precompute_kin,
                                  save_ckpt, PROPRIO, EXTERO)

INIT = Path(r"E:\desktop-rl-apl\aptrl_go2w\stage3_rl\experiments\20261119_temporal_v4_nav_r1\model_best.pt")
DATA = Path(r"E:\desktop-rl-apl\rl_rsl\data\tire_expert_v5.npz")
OUT = Path(r"E:\desktop-rl-apl\rl_rsl\runs\tire_bc_temporal\model_bc.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(INIT))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=8e-5)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck, arch, passthrough, enc, actor, nm, ns = build_from_checkpoint(Path(args.init), dev)
    _, _, _, enc_cpu, actor_cpu, _, _ = build_from_checkpoint(Path(args.init), "cpu")
    actor.train()

    ds = np.load(args.data)
    obs_all, act_all, ep_all = ds["obs"], ds["act"], ds["ep"]
    eps = []
    start = 0
    for i in range(1, len(obs_all) + 1):
        if i == len(obs_all) or ep_all[i] != ep_all[start]:
            eps.append((obs_all[start:i], act_all[start:i]))
            start = i
    print(f"tire episodes: {len(eps)}, steps={len(obs_all)}, init={Path(args.init).name}", flush=True)

    parsed = []
    for ob, ac in eps:
        ob_n = np.clip((ob - nm) / ns, -10, 10).astype(np.float32)
        kin = precompute_kin(actor_cpu, ob_n)
        parsed.append((torch.from_numpy(ob_n), torch.from_numpy(kin),
                       torch.from_numpy(ac.astype(np.float32))))

    w = torch.ones(30, device=dev)
    w[16:28] = 6.0
    w[29] = 5.0
    opt = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=args.lr)
    from stage3_rl.ppo_train import GAIT
    L = PROPRIO + 64 + GAIT + actor.KIN_DIM

    for ep in range(1, args.epochs + 1):
        tot, cnt = 0.0, 0
        perm = np.random.permutation(len(parsed))
        for j in perm:
            ob_n, kin, ac = parsed[j]
            T = len(ac)
            actor.reset_context()
            # 整个 episode 一次性算出 actor 输入与 16 帧滑窗（再按 chunk 前向，h0 逐段传递）
            ob_t = ob_n.to(dev)
            kin_t = kin.to(dev)
            flat = ob_t.reshape(-1, 426)
            ext = enc(flat[:, PROPRIO:PROPRIO + EXTERO])[:, :96]
            ao = torch.cat([flat[:, :PROPRIO], ext, flat[:, PROPRIO + EXTERO:]],
                           -1).reshape(1, T, PROPRIO + 96 + GAIT)
            base_in = torch.cat([ao[..., :PROPRIO], ao[..., PROPRIO:PROPRIO + 32],
                                 ao[..., PROPRIO + 96:]], -1)
            res_in = torch.cat([ao[..., :PROPRIO], ao[..., PROPRIO + 32:PROPRIO + 96],
                                ao[..., PROPRIO + 96:], kin_t.unsqueeze(0)], -1)   # (1,T,283)
            K = actor.HIST_K
            pad = res_in[:, 0:1, :].repeat(1, K - 1, 1)
            cat_res = torch.cat([pad, res_in], dim=1)
            ring_all = cat_res.unfold(1, K, 1).permute(0, 1, 3, 2).reshape(1, T, K * L)
            ac_t_all = ac.to(dev)
            h = None
            for c0 in range(0, T, args.chunk):
                c1 = min(c0 + args.chunk, T)
                mu, _, h = actor.forward_seq(base_in[:, c0:c1], res_in[:, c0:c1],
                                             ring_all[:, c0:c1], h0=h)
                ac_t = ac_t_all[c0:c1]
                loss = (((mu[0] - ac_t) ** 2) * w).sum() / (w.sum() * len(ac_t))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in list(enc.parameters()) + list(actor.parameters()) if p.requires_grad], 1.0)
                opt.step()
                h = h.detach() if h is not None else None
                tot += float(loss.detach()) * len(ac_t); cnt += len(ac_t)
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:2d}: weighted_mse={tot/max(1,cnt):.5f}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    state = {"enc": enc.state_dict(), "actor": actor.state_dict(),
             "critic": ck.get("critic", {}),
             "norm": ck["norm"], "iter": int(ck.get("iter", 0)),
             "actor_arch": arch, "extero_passthrough": passthrough,
             "actor_hidden": ck.get("actor_hidden", 768),
             "gru_hidden": ck.get("gru_hidden", 256),
             "decoder_path": ck.get("decoder_path"), "decoder_sha256": ck.get("decoder_sha256"),
             "material_aux_limit": 6.0, "material_wheel_control": "residual_v1",
             "material_command_profile": "start_ramp_100_v1"}
    torch.save(state, out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()



