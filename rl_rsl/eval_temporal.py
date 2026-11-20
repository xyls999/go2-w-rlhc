"""评测 temporal_residual_v4 checkpoint（PolicyRunner 兼容）：可从起点或从 21.5m 起跑。"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(r"E:\desktop-rl-apl\aptrl_go2w")
sys.path.insert(0, str(REPO))
from common.eval_protocol import material_command_speed
from common.material_route_monitor import MaterialRouteMonitor, frame_overlap_top
from stage3_rl.env import Go2WEnv
from stage3_rl.evaluate import PolicyRunner

TRACK = REPO / "results/diagnostics/robocon-material-preview-20261119-v5/scene.xml"
DEC = REPO / "stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz"

ck = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 8800
start_x = float(sys.argv[3]) if len(sys.argv) > 3 else None
import mujoco

kw = dict(seed=seed, fixed_track=str(TRACK), eval_mode=True, episode_len=9000,
          eval_protocol="pose-v1", decoder_path=str(DEC),
          material_aux_limit=6.0, material_wheel_control="residual_v1")
if start_x is not None:
    kw["material_eval_start"] = start_x
env = Go2WEnv(**kw)
m, d = env.model, env.data
wheels = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot_0") for l in ("FL", "FR", "RL", "RR")]
runner = PolicyRunner(ck, device="cpu", decoder_path=str(DEC))
obs = env.reset()
runner.actor.reset_context()
rt = MaterialRouteMonitor()
rt.update(float(d.qpos[0]), float(d.qpos[1]), d.geom_xpos[wheels, :2],
          [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "",
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "") for c in d.contact],
          frame_overlap_top(m, d, 44.98, 45.02))
rt.violations.discard("invalid_start")
max_x = float(d.qpos[0])
steps = 0
for st in range(9000):
    x = float(d.qpos[0])
    env.vx_cmd = material_command_speed(st, x, "start_ramp_100_v1")
    obs = env._get_obs()
    a = runner.act(obs[None], True)
    obs, _r, done, info = env.step(a)
    steps += 1
    max_x = max(max_x, float(d.qpos[0]))
    if done:
        break
print(f"ck={Path(ck).parent.name}/{Path(ck).name} seed={seed} start={start_x} steps={steps} "
      f"max_x={max_x:.2f} final_x={float(d.qpos[0]):.2f} y={float(d.qpos[1]):.2f} reason={info.get('reason')}")
