"""用开源 RL 库 rsl_rl（IsaacLab 同款 PPO）在冻结 v5 赛道上训练。

设计原则：尽量少自己写
  * PPO 算法 / 网络 / 训练循环  → rsl_rl（pip 安装，社区验证）
  * 机器人 + 物理 + 奖励 + TVAE 解码器 → 仓库既有 `stage3_rl/env.py`（不改）
  * 本文件只做"胶水"：
      1) 把 `Go2WEnv` 包成 rsl_rl 需要的 VecEnv（多进程并行）
      2) 组装 rsl_rl 的训练配置并启动

用法：
    $env:PYTHONPATH="E:\\desktop-rl-apl\\aptrl_go2w"
    python E:\\desktop-rl-apl\\rl_rsl\\train_rsl_v5.py --iters 30 --envs 8
    python E:\\desktop-rl-apl\\rl_rsl\\train_rsl_v5.py --iters 2 --envs 4 --smoke
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

REPO = Path(r"E:\desktop-rl-apl\aptrl_go2w")
TRACK = REPO / "results/diagnostics/robocon-material-preview-20261119-v5/scene.xml"
DEC = REPO / "stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz"

# ----------------------------------------------------------------- 子进程 worker
def _worker(conn, cfg):
    import sys
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, r"E:\desktop-rl-apl\rl_finetune_v5")
    from stage3_rl.env import Go2WEnv
    import numpy as _np
    import mujoco
    from course_reward import RewardWeights, step_reward, terminal_reward
    _w = RewardWeights(lane=60.0, lane_deadband=0.08)
    # 轮胎段里程碑加密（每次翻越一行给反馈），其余障碍里程碑
    TIRE_MILESTONES = ((25.0, 15.0), (25.6, 15.0), (26.2, 15.0), (26.8, 15.0), (27.2, 15.0))
    MILESTONES = ((29.6, 25.0), (33.5, 25.0), (36.5, 25.0), (40.5, 25.0), (43.4, 25.0), (45.2, 25.0), (47.0, 25.0))
    crossed = set()
    tire_rows = set()
    tire_wheels = {}
    from common.eval_protocol import material_command_speed
    env = Go2WEnv(seed=cfg["seed"], fixed_track=str(TRACK), eval_mode=True,
                  episode_len=cfg["episode_len"], eval_protocol="pose-v1",
                  decoder_path=str(DEC), material_aux_limit=cfg["aux_limit"],
                  material_wheel_control="residual_v1",
                  reward_profile=cfg["reward_profile"])
    obs = env.reset()
    conn.send(("obs", obs))
    while True:
        try:
            msg, payload = conn.recv()
        except (EOFError, KeyboardInterrupt, OSError):
            return
        if msg == "step":
            a = np.asarray(payload, dtype=np.float64)
            d = env.data
            R = d.xmat[env.body_root].reshape(3, 3)
            roll = float(_np.arctan2(R[2, 1], R[2, 2]))
            pitch = float(-_np.arcsin(_np.clip(R[2, 0], -1, 1)))
            v_world = R @ d.qvel[0:3]
            env.vx_cmd = material_command_speed(env.steps, float(d.qpos[0]), "start_ramp_100_v1")
            parts = step_reward(vx=float(v_world[0]), vx_cmd=env.vx_cmd, roll=roll, pitch=pitch,
                                z=float(d.qpos[2]), vz=float(v_world[2]),
                                y=float(d.qpos[1]), x=float(d.qpos[0]),
                                tau_leg=d.ctrl[env.leg_act].copy(),
                                tau_wheel=d.ctrl[env.wheel_act].copy(),
                                wheel_action=float(a[29]), action=a, prev_action=None, w=_w)
            obs, _env_rew, done, info = env.step(a)
            rew = parts["total"]
            x_now = float(d.qpos[0])
            for th, bonus in TIRE_MILESTONES + MILESTONES:
                if x_now >= th and th not in crossed:
                    crossed.add(th)
                    rew += bonus
            # 轮-胎接触奖励：按"轮胎行"去重，每行首次被轮子接触 +8；记录每行被哪些轮子踩过
            for c in d.contact:
                n1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
                n2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
                for tire_name, other in ((n1, n2), (n2, n1)):
                    if not tire_name.startswith("tire_"):
                        continue
                    if not other.endswith("_foot_0"):
                        continue
                    parts_t = tire_name.split("_")
                    if len(parts_t) < 3:
                        continue
                    row = parts_t[1]
                    if row not in tire_rows:
                        tire_rows.add(row)
                        rew += 8.0
                    tire_wheels.setdefault(row, set()).add(other)
            if done:
                ok = bool(info.get("success", False)) or float(d.qpos[0]) >= 100.0
                rew += terminal_reward(ok, _w)
                if len(tire_wheels) > 0 and all(len(w) == 4 for w in tire_wheels.values()):
                    rew += 15.0      # 四轮都踩上过胎顶的额外奖励
            conn.send(("step", (obs, float(rew), bool(done), info.get("reason"))))
            if done:
                obs = env.reset()
                crossed.clear()
                tire_rows.clear()
                tire_wheels.clear()
                conn.send(("obs", obs))
        elif msg == "close":
            conn.close()
            return


class Go2WVecEnv:
    """最小 VecEnv：只实现 rsl_rl 用到的接口。"""

    REW_SCALE = 0.01
    ALIVE_BONUS = 0.25        # 每步存活奖励（原始量纲），堵住"秒死省钱"的漏洞
    TERM_PENALTY = -25.0      # 非超时终止的一次性惩罚

    def __init__(self, num_envs: int, episode_len: int = 9000, aux_limit: float = 6.0,
                 reward_profile: str = "paper_adapted", seed0: int = 8800):
        self.num_envs = num_envs
        self.num_actions = 30
        self.max_episode_length = episode_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = {"num_envs": num_envs, "episode_len": episode_len,
                    "track": str(TRACK), "reward_profile": reward_profile}
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(num_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, daemon=True,
                            args=(child, dict(seed=seed0 + i, episode_len=episode_len,
                                              aux_limit=aux_limit, reward_profile=reward_profile)))
            p.start()
            self.conns.append(parent)
            self.procs.append(p)
        self._obs = np.zeros((num_envs, 426), dtype=np.float32)
        for i, c in enumerate(self.conns):
            tag, obs = c.recv()
            self._obs[i] = obs

    # -- rsl_rl 接口 -----------------------------------------------------
    def get_observations(self):
        return TensorDict({"policy": torch.from_numpy(self._obs).to(self.device)},
                          batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor):
        act_np = actions.detach().cpu().numpy()
        for i, c in enumerate(self.conns):
            c.send(("step", act_np[i]))
        rew = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=bool)
        for i, c in enumerate(self.conns):
            tag, (obs, r, d, _reason) = c.recv()
            rew[i] = r * self.REW_SCALE
            done[i] = d
            self._obs[i] = obs
            if d:
                tag, obs = c.recv()      # worker 自动 reset 后的新观测
                self._obs[i] = obs
                self.episode_length_buf[i] = 0
        self.episode_length_buf += 1
        dones_t = torch.from_numpy(done).to(self.device)
        obs_t = TensorDict({"policy": torch.from_numpy(self._obs).to(self.device)},
                           batch_size=[self.num_envs])
        extras = {"time_outs": torch.zeros_like(dones_t)}   # 终止都不是超时
        return obs_t, torch.from_numpy(rew).to(self.device), dones_t, extras

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()


# ----------------------------------------------------------------- 训练配置
def make_train_cfg(steps_per_env: int = 48, save_interval: int = 10) -> dict:
    return {
        "num_steps_per_env": steps_per_env,
        "save_interval": save_interval,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128],
                  "activation": "elu", "obs_normalization": True,
                  "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_range": [0.05, 4.0], "std_type": "scalar", "learn_std": True}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128],
                   "activation": "elu", "obs_normalization": True},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 5,
                      "num_mini_batches": 4, "clip_param": 0.2, "gamma": 0.997,
                      "lam": 0.97, "value_loss_coef": 1.0, "entropy_coef": 0.01,
                      "learning_rate": 3e-4, "max_grad_norm": 1.0, "optimizer": "adam",
                      "use_clipped_value_loss": True, "schedule": "adaptive",
                      "desired_kl": 0.01},
        "multi_gpu": None,
        "check_for_nan": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps-per-env", type=int, default=48)
    ap.add_argument("--aux-limit", type=float, default=6.0)
    ap.add_argument("--reward-profile", default="paper_adapted",
                    choices=("paper_adapted", "legacy", "integrated_v1"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "runs" / "v5_rsl"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default=None, help="从 rsl_rl 的 model_*.pt 继续训练")
    args = ap.parse_args()

    from common.material_acceptance import verify_target_scene
    from rsl_rl.runners import OnPolicyRunner

    verify_target_scene(str(TRACK))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    envs = args.envs
    iters = 2 if args.smoke else args.iters
    venv = Go2WVecEnv(envs, episode_len=9000, aux_limit=args.aux_limit,
                      reward_profile=args.reward_profile)
    cfg = make_train_cfg()
    (out / "run_config.json").write_text(json.dumps(
        {"envs": envs, "iters": iters, "steps_per_env": cfg["num_steps_per_env"],
         "aux_limit": args.aux_limit, "reward_profile": args.reward_profile,
         "track": str(TRACK), "decoder": str(DEC), "trainer": "rsl_rl OnPolicyRunner",
         "train_cfg": cfg}, indent=2), encoding="utf-8")

    runner = OnPolicyRunner(venv, cfg, log_dir=str(out), device=venv.device)
    if args.resume:
        runner.load(args.resume)
        print(f"[resume] loaded {args.resume} (iter={runner.current_learning_iteration})", flush=True)
    t0 = time.perf_counter()
    runner.learn(num_learning_iterations=iters, init_at_random_ep_len=False)
    print(f"[done] {iters} iters in {time.perf_counter()-t0:.1f}s -> {out}", flush=True)
    venv.close()


if __name__ == "__main__":
    main()







