"""对比教师 vs BC 在轮胎段（x≈23.9-25.6）的物理轨迹，定位失败点。"""
import sys
from pathlib import Path

import numpy as np
import mujoco
import torch

sys.path.insert(0, r"C:\Users\Administrator\AppData\Local\Temp\opencode")
sys.path.insert(0, r"E:\desktop-rl-apl\aptrl_go2w")
sys.path.insert(0, r"E:\desktop-rl-apl\rl_finetune_v5")

from exact_edge_v4_expert import ExactEdgeObsExpert
from eval_v4_multiseed_now import load_merged_params
from common.eval_protocol import material_command_speed
from stage3_rl.env import Go2WEnv
from finetune_full_course import build_from_checkpoint, EXTERO, PROPRIO

TRACK = r"E:\desktop-rl-apl\aptrl_go2w\results\diagnostics\robocon-material-preview-20261119-v5\scene.xml"
DEC = r"E:\desktop-rl-apl\aptrl_go2w\stage2_tvae\candidates\20261113T153754Z\tvae_decoders.npz"


def make_env(seed):
    return Go2WEnv(seed=seed, fixed_track=TRACK, eval_mode=True, episode_len=9000,
                   eval_protocol="pose-v1", decoder_path=DEC,
                   material_aux_limit=6.0, material_wheel_control="residual_v1")


def geom_z(env, names):
    return [float(env.data.geom_xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, n)][2])
            for n in names]


def dump(tag, env, rows, step, info):
    d = env.data
    x = float(d.qpos[0]); y = float(d.qpos[1]); z = float(d.qpos[2])
    quat = d.qpos[3:7]
    roll = float(np.arctan2(2*(quat[0]*quat[1]+quat[2]*quat[3]), 1-2*(quat[1]**2+quat[2]**2)))
    pitch = float(np.arcsin(2*(quat[0]*quat[2]-quat[3]*quat[1])))
    wz = geom_z(env, ["FL_foot_0", "FR_foot_0", "RL_foot_0", "RR_foot_0"])
    rows.append((step, x, y, z, roll, pitch, wz[0], wz[1]))
    return info


def run(policy, seed=8816, max_steps=3000):
    env = make_env(seed); d = env.data
    env.reset()
    if hasattr(policy, "reset_context"):
        policy.reset_context()
    rows = []
    for st in range(max_steps):
        x = float(d.qpos[0])
        env.vx_cmd = material_command_speed(st, x, "start_ramp_100_v1")
        obs = env._get_obs().astype(np.float32)
        a = policy.act(obs, st)
        if 23.9 <= x <= 26.2:
            dump("", env, rows, st, None)
        obs2, _r, done, info = env.step(a)
        if done:
            print(f"  done step={st} x={float(d.qpos[0]):.2f} reason={info}")
            break
    return rows


class Teacher:
    def __init__(self):
        self.ex = ExactEdgeObsExpert(load_merged_params())

    def act(self, obs, st):
        return self.ex.act(obs[None], True)


class NetPolicy:
    def __init__(self, ckpt):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck, arch, past, self.enc, self.actor, self.nm, self.ns = build_from_checkpoint(Path(ckpt), self.dev)

    def reset_context(self):
        self.actor.reset_context()

    def act(self, obs, st):
        ob_n = np.clip((obs - self.nm) / self.ns, -10, 10).astype(np.float32)
        ob_t = torch.from_numpy(ob_n[None]).to(self.dev)
        with torch.no_grad():
            ao = torch.cat([ob_t[:, :PROPRIO], self.enc(ob_t[:, PROPRIO:PROPRIO + EXTERO]),
                            ob_t[:, PROPRIO + EXTERO:]], dim=1)
            a = self.actor.dist(ao).mean
        return a.cpu().numpy()[0].astype(np.float64)


if __name__ == "__main__":
    which = sys.argv[1]
    ck = sys.argv[2] if len(sys.argv) > 2 else None
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 8816
    pol = Teacher() if which == "teacher" else NetPolicy(ck)
    print(f"== {which} seed={seed} ==")
    rows = run(pol, seed)
    for r in rows[::5]:
        print("st=%4d x=%.2f y=%.2f z=%.2f roll=%+.2f pitch=%+.2f FLz=%.3f FRz=%.3f" % r)
    if rows:
        print("window x: %.2f -> %.2f" % (rows[0][1], rows[-1][1]))
