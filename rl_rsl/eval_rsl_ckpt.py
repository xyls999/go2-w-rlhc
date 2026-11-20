"""快速评估 rsl_rl 训练出来的策略在 v5 赛道上能走多远（单进程单环境）。"""
import sys
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict

REPO = Path(r"E:\desktop-rl-apl\aptrl_go2w")
sys.path.insert(0, str(REPO))
sys.path.insert(0, r"E:\desktop-rl-apl\rl_finetune_v5")

from stage3_rl.env import Go2WEnv, PROPRIO
from course_reward import RewardWeights
from rsl_rl.models import MLPModel

TRACK = REPO / "results/diagnostics/robocon-material-preview-20261119-v5/scene.xml"
DEC = REPO / "stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz"
CKPT = sys.argv[1] if len(sys.argv) > 1 else r"E:\desktop-rl-apl\rl_rsl\runs\v5_rsl_run4\model_1199.pt"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 8800
START_X = float(sys.argv[3]) if len(sys.argv) > 3 else None

obs_td = TensorDict({"policy": torch.zeros(1, 426)}, batch_size=[1])
actor = MLPModel(obs_td, {"actor": ["policy"]}, "actor", 30,
                 hidden_dims=[512, 256, 128], activation="elu",
                 obs_normalization=True,
                 distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0,
                                   "std_range": [0.05, 4.0], "std_type": "scalar", "learn_std": True})
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
actor.load_state_dict(ck["actor_state_dict"])
actor.eval()

_kw = dict(seed=SEED, fixed_track=str(TRACK), eval_mode=True, episode_len=9000,
           eval_protocol="pose-v1", decoder_path=str(DEC), material_aux_limit=6.0,
           material_wheel_control="residual_v1")
if START_X is not None:
    _kw["material_eval_start"] = START_X
env = Go2WEnv(**_kw)
obs = env.reset()
max_x = float(env.data.qpos[0])
steps = 0
with torch.no_grad():
    for _ in range(9000):
        a = actor(TensorDict({"policy": torch.from_numpy(obs[None])}, batch_size=[1])).squeeze(0).numpy()
        obs, r, done, info = env.step(a)
        steps += 1
        max_x = max(max_x, float(env.data.qpos[0]))
        if done:
            break
print(f"ckpt={Path(CKPT).name} seed={SEED} steps={steps} max_x={max_x:.2f} "
      f"final_x={float(env.data.qpos[0]):.2f} y={float(env.data.qpos[1]):.2f} "
      f"reason={info.get('reason')} viol={sorted(getattr(env, 'violations', []))}")

