"""自检：奖励函数行为 + 微调产物能被官方 PolicyRunner 加载并跑动。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path(r"E:\desktop-rl-apl\aptrl_go2w")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

from course_reward import RewardWeights, lane_reward, jump_reward, energy_reward  # noqa: E402
from stage3_rl.env import Go2WEnv  # noqa: E402
from stage3_rl.evaluate import PolicyRunner  # noqa: E402

TRACK = REPO / "results/diagnostics/robocon-material-preview-20261119-v5/scene.xml"
DEC = REPO / "stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz"
CKPT = ROOT / "runs/v5_ft/model_smoke.pt"

w = RewardWeights()
print("== lane reward ==")
for y in (4.00, 4.10, 4.30, 4.60, 4.85):
    print(f"  y={y:.2f} on_lane={lane_reward(y, 20.0, w):8.3f}  in_brick={lane_reward(y, 28.0, w):8.3f}")
assert lane_reward(4.0, 28.0, w) == 0.0
assert abs(lane_reward(4.6, 28.0, w) - 3.0 * lane_reward(4.6, 20.0, w)) < 1e-9 < abs(lane_reward(4.6, 20.0, w))

print("== jump reward ==")
for vz in (0.0, 0.3, 0.8):
    print(f"  vz={vz:.1f} flat={jump_reward(vz, 10.0, w):7.3f} obstacle={jump_reward(vz, 32.0, w):7.3f}")
assert jump_reward(0.0, 10.0, w) == 0.0
assert jump_reward(0.5, 10.0, w) < jump_reward(0.5, 32.0, w) < 0

print("== energy reward ==")
print("  small:", round(energy_reward([1.0] * 12, [1.0] * 4, 0.0, w), 5))
print("  large:", round(energy_reward([5.0] * 12, [5.0] * 4, 0.5, w), 5))
assert energy_reward([5.0] * 12, [5.0] * 4, 0.5, w) < energy_reward([1.0] * 12, [1.0] * 4, 0.0, w) < 0

print("== policy runner load ==")
assert CKPT.exists(), f"missing {CKPT}; run --smoke first"
runner = PolicyRunner(str(CKPT), device="cpu", decoder_path=str(DEC))
env = Go2WEnv(seed=8800, fixed_track=str(TRACK), eval_mode=True, episode_len=200,
              eval_protocol="pose-v1", decoder_path=str(DEC),
              material_aux_limit=6.0, material_wheel_control="residual_v1")
obs = env.reset()
runner.actor.reset_context()
for _ in range(200):
    a = runner.act(obs[None], True)
    obs, _, done, info = env.step(a)
    if done:
        break
print(f"  ran 200 steps, x={float(env.data.qpos[0]):.3f}, reason={info.get('reason')}")
print("ALL CHECKS PASSED")
