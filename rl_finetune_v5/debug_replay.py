"""检查 PPO 重放一致性：同一 checkpoints、同一段观测，采样时的 logp 与重放 logp 是否一致。"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"E:\desktop-rl-apl\rl_finetune_v5")
REPO = Path(r"E:\desktop-rl-apl\aptrl_go2w")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(REPO))
import finetune_full_course as F

ck = REPO / "stage3_rl/experiments/20261119_temporal_v4_nav_r1/model_best.pt"
device = "cuda"
ckpt, arch, pass_raw, enc, actor, nm, ns = F.build_from_checkpoint(ck, device)
critic = F.__dict__["Critic"]().to(device)
with torch.no_grad():
    actor.log_std[:16] = 0.45; actor.log_std[16:28] = 0.28; actor.log_std[28:] = 0.60
env = F.Go2WEnv(seed=8800, fixed_track=str(F.TRACK), eval_mode=True, episode_len=400,
                eval_protocol="pose-v1", decoder_path=str(F.DEC),
                material_aux_limit=6.0, material_wheel_control="residual_v1")
ep, steps = F.rollout(env, enc, actor, critic, nm, ns, device, "start_ramp_100_v1", F.RewardWeights())
print("rollout steps:", steps, "success:", ep.success)

# 重放前 300 步，比较 logp
actor.reset_context()
obs_n = np.clip((ep.obs - nm) / ns, -10, 10).astype(np.float32)
diffs = []
with torch.no_grad():
    for t in range(min(300, len(ep.rew))):
        ob_t = torch.from_numpy(obs_n[t][None]).to(device)
        ao = torch.cat([ob_t[:, :F.PROPRIO], enc(ob_t[:, F.PROPRIO:F.PROPRIO + F.EXTERO]),
                        ob_t[:, F.PROPRIO + F.EXTERO:]], dim=1)
        d = actor.dist(ao)
        a = torch.from_numpy(ep.act[t][None]).to(device)
        lp = float(d.log_prob(a).sum(-1).item())
        diffs.append(lp - float(ep.logp[t]))
print("logp replay diff: max", max(abs(x) for x in diffs), "mean", float(np.mean(diffs)))
print("first 5:", [round(x, 6) for x in diffs[:5]])
