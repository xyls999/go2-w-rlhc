"""用轮胎段专家示范做监督微调（BC），把"跳上轮胎并保持稳定"注入现有策略，然后交给 PPO 继续。

要点：
  * 从当前 rsl_rl checkpoint 的 actor 初始化（保留已有能力）
  * 只在轮胎段示范上做加权 MSE（aux 与轮速权重更高）
  * 产物保存成 rsl_rl 格式，可直接 --resume 继续 PPO
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel


def build_actor():
    td = TensorDict({"policy": torch.zeros(1, 426)}, batch_size=[1])
    return MLPModel(td, {"actor": ["policy"]}, "actor", 30,
                    hidden_dims=[512, 256, 128], activation="elu",
                    obs_normalization=True,
                    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0,
                                      "std_range": [0.05, 4.0], "std_type": "scalar",
                                      "learn_std": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:\desktop-rl-apl\rl_rsl\data\tire_expert_v5.npz")
    ap.add_argument("--init", default=r"E:\desktop-rl-apl\rl_rsl\runs\v5_rsl_run12\model_7680.pt")
    ap.add_argument("--out", default=r"E:\desktop-rl-apl\rl_rsl\runs\tire_bc\model_bc.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = np.load(args.data)
    obs = torch.from_numpy(ds["obs"].astype(np.float32)).to(dev)
    act = torch.from_numpy(ds["act"].astype(np.float32)).to(dev)
    w = torch.ones(30, device=dev)
    w[16:28] = 6.0     # aux 权重更高
    w[29] = 5.0        # 轮速
    print(f"dataset: {len(obs)} steps, init={Path(args.init).name}", flush=True)

    actor = build_actor().to(dev)
    ck = torch.load(args.init, map_location=dev, weights_only=False)
    actor.load_state_dict(ck["actor_state_dict"])
    actor.train()

    opt = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=args.lr)
    n = len(obs)
    for ep in range(1, args.epochs + 1):
        idx = torch.randperm(n, device=dev)
        tot, cnt = 0.0, 0
        for b0 in range(0, n, args.batch):
            sel = idx[b0:b0 + args.batch]
            pred = actor(TensorDict({"policy": obs[sel]}, batch_size=[len(sel)]))
            loss = (((pred - act[sel]) ** 2) * w).sum() / (w.sum() * len(sel))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += float(loss.detach()) * len(sel); cnt += len(sel)
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:2d}: weighted_mse={tot/cnt:.5f}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": actor.state_dict(),
                "critic_state_dict": ck["critic_state_dict"],
                "optimizer_state_dict": ck.get("optimizer_state_dict"),
                "iter": int(ck.get("iter", 0)), "infos": None}, out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
