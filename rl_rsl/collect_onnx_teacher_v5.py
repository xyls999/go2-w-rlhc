"""用 Unitree 预训练行走策略（ONNX，位置控制）在我们自己的 v5 环境里做教师：
把它的"期望关节角 + 轮速"映射成本环境 30 维动作，跑出轨迹并保存 (obs, act) 数据集。

映射关系（严格按本仓库 env.py 的语义）：
    q_des  = Q_DEFAULT + AUX_SCALE * aux          (AUX_SCALE = 0.2)
    w_cmd  = (vx_cmd / WHEEL_R) * contact * (1 + wa)
  teacher:  q_des_t = ACTION_DEFAULT_Q + 0.5 * a[:12]  (顺序 FL,FR,RL,RR)
            w_des  = clip(35 * a[12:], -35, 35)       (rad/s)
  =>        aux    = (q_des_t - Q_DEFAULT) / 0.2
            wa     = clip(mean(w_des)/(vx_cmd/0.086) - 1, -1.0, 0.5)
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(r"E:\desktop-rl-apl\aptrl_go2w")
sys.path.insert(0, str(REPO))

from common.eval_protocol import material_command_speed  # noqa: E402
from stage3_rl.env import AUX_SCALE, Go2WEnv, KP, KD, Q_DEFAULT, WHEEL_R  # noqa: E402
KP_T = np.asarray([20.0, 20.0, 40.0] * 4)   # teacher PD
KD_T = np.asarray([1.0, 1.0, 2.0] * 4)

TRACK = REPO / "results/diagnostics/robocon-material-preview-20261119-v5/scene.xml"
DEC = REPO / "stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz"
ONNX = Path(r"C:\Users\Administrator\AppData\Local\Temp\opencode\27rc-dog-race\artifacts\go2w_good\policy.onnx")

HISTORY = 5
DEFAULT_Q_T = np.asarray([0.1, 0.9, -1.8, -0.1, 0.9, -1.8,
                          0.1, 0.9, -1.8, -0.1, 0.9, -1.8])   # teacher 观测用
ACTION_DEFAULT_Q_T = np.asarray([-0.1, 0.9, -1.8, 0.1, 0.9, -1.8,
                                 -0.1, 0.9, -1.8, 0.1, 0.9, -1.8])
Z_TROT = np.asarray([-1.0852, 0.9451, -0.1542, -0.1974, 0.7814, -1.2543, 0.8814, 0.3162,
                     -0.9465, -1.0488, -1.1230, -0.1939, 0.4290, 0.2951, 0.9602, 1.3548])


class History:
    def __init__(self):
        self.values = deque(maxlen=HISTORY)

    def push(self, v):
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if not self.values:
            self.values.extend(v.copy() for _ in range(HISTORY))
        else:
            self.values.append(v.copy())
        return np.concatenate(self.values)


def inverse_rotate(quat, vec):
    w, x, y, z = quat
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R.T @ vec


class TeacherObs:
    names = ("ang", "gravity", "command", "phase", "leg_pos", "leg_vel", "wheel_vel", "action")

    def __init__(self):
        self.h = {n: History() for n in self.names}

    def build(self, *, ang, quat, command, elapsed, leg_pos, leg_vel, wheel_vel, action):
        moving = np.linalg.norm(command) >= 0.1
        a = 2 * math.pi * (elapsed % 0.6) / 0.6
        terms = {"ang": ang, "gravity": inverse_rotate(quat, np.array([0.0, 0.0, -1.0])),
                 "command": command,
                 "phase": (math.sin(a), math.cos(a)) if moving else (0.0, 0.0),
                 "leg_pos": leg_pos - DEFAULT_Q_T, "leg_vel": leg_vel,
                 "wheel_vel": wheel_vel, "action": action}
        return np.concatenate([self.h[n].push(terms[n]) for n in self.names]).astype(np.float32)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=8800)
    ap.add_argument("--out", default=r"E:\desktop-rl-apl\rl_rsl\data\teacher_onnx_v5.npz")
    args = ap.parse_args()

    import mujoco
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    obs_all, act_all, ep_all, x_all = [], [], [], []
    for k in range(args.episodes):
        seed = args.seed0 + k
        env = Go2WEnv(seed=seed, fixed_track=str(TRACK), eval_mode=True, episode_len=9000,
                      eval_protocol="pose-v1", decoder_path=str(DEC),
                      material_aux_limit=6.0, material_wheel_control="residual_v1")
        m, d = env.model, env.data
        leg_q = env.leg_qadr
        leg_v = env.leg_vadr
        wheel_v = env.wheel_vadr
        obs = env.reset()
        tob = TeacherObs()
        last_a = np.zeros(16, np.float32)
        max_x = float(d.qpos[0])
        for st in range(9000):
            env.vx_cmd = material_command_speed(st, float(d.qpos[0]), "start_ramp_100_v1")
            R = d.xmat[env.body_root].reshape(3, 3)
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            ey = float(d.qpos[1]) - 4.0
            cmd = np.array([env.vx_cmd,
                            float(np.clip(1.0 * ey, -0.5, 0.5)),
                            float(np.clip(2.0 * ey + 1.5 * yaw, -0.8, 0.8))], dtype=np.float32)
            ang = d.qvel[3:6]
            t_obs = tob.build(ang=ang, quat=d.qpos[3:7], command=cmd,
                              elapsed=st * 0.02, leg_pos=d.qpos[leg_q],
                              leg_vel=d.qvel[leg_v], wheel_vel=d.qvel[wheel_v],
                              action=last_a)
            a_t = sess.run([oname], {iname: t_obs})[0].astype(np.float32).reshape(-1)
            last_a = a_t

            # 力矩级映射：先算教师 PD 想输出的力矩，再反解出本环境需要的 aux 目标偏移
            q_des_t = ACTION_DEFAULT_Q_T + 0.5 * a_t[:12]
            q_now = d.qpos[leg_q]
            dq_now = d.qvel[leg_v]
            tau_t = KP_T * (q_des_t - q_now) - KD_T * dq_now
            aux = np.clip((tau_t + KD * dq_now + KP * (q_now - Q_DEFAULT)) / (AUX_SCALE * KP),
                          -6.0, 6.0)
            w_des = float(np.clip(np.mean(35.0 * a_t[12:]), -35.0, 35.0))
            base_speed = max(env.vx_cmd / WHEEL_R, 0.5)
            wa = float(np.clip(w_des / base_speed - 1.0, -1.0, 0.5))
            act30 = np.concatenate([Z_TROT, aux, [-8.0, wa]]).astype(np.float32)

            obs_all.append(obs); act_all.append(act30)
            ep_all.append(seed); x_all.append(float(d.qpos[0]))
            obs, _r, done, info = env.step(act30)
            max_x = max(max_x, float(d.qpos[0]))
            if done:
                break
        print(f"[ep {k+1}/{args.episodes}] seed={seed} steps={st+1} max_x={max_x:.2f} reason={info.get('reason')}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=np.stack(obs_all), act=np.stack(act_all),
                        ep=np.asarray(ep_all, np.int64), x=np.asarray(x_all, np.float32))
    print(f"saved {len(obs_all)} steps -> {out}")


if __name__ == "__main__":
    main()

