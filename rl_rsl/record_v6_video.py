"""录制 v6 全程通关视频（神经网络策略，确定性执行）。"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, r"E:\desktop-rl-apl\aptrl_go2w")
from common.eval_protocol import material_command_speed
from common.material_route_monitor import MaterialRouteMonitor, frame_overlap_top
from stage3_rl.env import Go2WEnv
from stage3_rl.evaluate import PolicyRunner, VideoRecorder

REPO = Path(r"E:\desktop-rl-apl\aptrl_go2w")
TRACK = REPO / "results/diagnostics/robocon-material-preview-20261119-v6/scene.xml"
DEC = REPO / "stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz"
CKPT = Path(r"E:\desktop-rl-apl\rl_rsl\runs\v5_bc_best\model_bc.pt")
DESK = Path(os.environ["USERPROFILE"]) / "Desktop"


def record(seed, out_path):
    env = Go2WEnv(seed=seed, fixed_track=str(TRACK), eval_mode=True, episode_len=9000,
                  eval_protocol="pose-v1", decoder_path=str(DEC),
                  material_aux_limit=6.0, material_wheel_control="residual_v1")
    m, d = env.model, env.data
    wh = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot_0") for l in ("FL", "FR", "RL", "RR")]
    runner = PolicyRunner(CKPT, device="cpu", decoder_path=DEC)
    rec = VideoRecorder(env, str(out_path), fps=50, height=360, width=640)
    env.reset()
    rt = MaterialRouteMonitor()

    def upd():
        rt.update(float(d.qpos[0]), float(d.qpos[1]), d.geom_xpos[wh, :2],
                  [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "",
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "") for c in d.contact],
                  frame_overlap_top(m, d, 44.98, 45.02))

    upd(); rec.start(); rec(env)
    info = {}
    done = False
    st = 0
    for st in range(9000):
        x = float(d.qpos[0])
        env.vx_cmd = material_command_speed(st, x, runner.material_command_profile)
        obs = env._get_obs()
        a = np.asarray(runner.act(obs[None], True), dtype=float).reshape(-1)
        obs, _, done, info = env.step(a)
        upd(); rec(env)
        if done or rt.violations or x >= 100.0:
            break
    ok = (float(d.qpos[0]) >= 100.0 and rt.success()) or bool(info.get("success", False))
    if ok:
        rec.finish(True)
    else:
        try:
            rec.proc.stdin.close(); rec.proc.stderr.read(); rec.proc.stderr.close(); rec.proc.wait()
        finally:
            rec.active = False
    print(f"seed={seed} ok={ok} x={float(d.qpos[0]):.2f} steps={st} reason={info.get('reason')} "
          f"viol={sorted(rt.violations)} completed={rt.completed} -> {out_path}", flush=True)
    return ok


if __name__ == "__main__":
    seeds = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["8800"])]
    for seed in seeds:
        out = DESK / f"go2w_v6_SUCCESS_seed{seed}.mp4"
        record(seed, out)
