"""诊断：BC / 任意 ckpt 在轮胎段的闭环表现（不训练）。

用法: python diag_bc.py <ckpt> <start_x> <seeds_csv>
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"E:\desktop-rl-apl\rl_finetune_v5")
sys.path.insert(0, r"E:\desktop-rl-apl\aptrl_go2w")
from finetune_full_course import build_from_checkpoint, EXTERO, PROPRIO  # noqa: E402
from common.eval_protocol import material_command_speed  # noqa: E402
from train_mp import EnvWrap  # noqa: E402


def run(ckpt, start_x, seeds, max_steps=1200):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck, arch, passthrough, enc, actor, nm, ns = build_from_checkpoint(Path(ckpt), dev)
    print("arch:", arch, "passthrough:", passthrough)
    for seed in seeds:
        e = EnvWrap(seed, start_x)
        actor.reset_context()
        best = 0.0
        ys = []
        reason = "max_steps"
        for t in range(max_steps):
            d = e.env.data
            e.env.vx_cmd = material_command_speed(t, float(d.qpos[0]), "start_ramp_100_v1")
            ob = e.env._get_obs().astype(np.float32)
            ob_n = np.clip((ob - nm) / ns, -10, 10).astype(np.float32)
            ob_t = torch.from_numpy(ob_n[None]).to(dev)
            with torch.no_grad():
                ao = torch.cat([ob_t[:, :PROPRIO], enc(ob_t[:, PROPRIO:PROPRIO + EXTERO]),
                                ob_t[:, PROPRIO + EXTERO:]], dim=1)
                a = actor.dist(ao).mean
            _, _r, done, info = e.env.step(a.cpu().numpy()[0].astype(np.float64))
            x = float(d.qpos[0])
            best = max(best, x)
            if x > 24.6:
                ys.append(round(float(d.qpos[1]), 2))
            if done:
                reason = str(info.get("reason"))
                break
        print(f"seed={seed} max_x={best:.2f} y_after_tire={ys[:6]} reason={reason} steps={t}")


if __name__ == "__main__":
    ck = sys.argv[1]
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 21.5
    seeds = [int(s) for s in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["8800", "8801", "8802"])]
    run(ck, start, seeds)
