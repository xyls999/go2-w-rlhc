"""全程 RL 微调使用的奖励函数（可单测的纯函数）。

设计目标（对应用户要求）：
  1) 基础项保持最少：前进跟踪、姿态稳定、机身高度、动作平滑、力矩惩罚；
  2) 走线约束：deviation 超过 deadband 后平方惩罚，砖块段 3 倍权重（堵住绕行）；
  3) 跳跃惩罚：只罚向上竖直速度，平地加倍（"能走就别跳"）;
  4) 能耗惩罚：腿部力矩 + 轮力矩 + 轮速动作。

所有量都来自环境真值（训练期允许），推理阶段策略只看到 426 维 obs。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

LANE_CENTER_Y = 4.0
BLOCK_X_RANGE = (27.4, 29.6)      # 砖块段：绕行主要发生在这里
FLAT_X_RANGES = ((0.0, 23.5), (47.0, 104.0))  # 平地：不需要跳


@dataclass
class RewardWeights:
    # 1.1 基础项
    progress: float = 1.0
    upright: float = 0.6
    height: float = 8.0
    height_ref: float = 0.36
    height_deadband: float = 0.05
    smooth: float = 0.02
    torque: float = 1.5e-4
    # 1.2 走线
    lane: float = 20.0
    lane_deadband: float = 0.12
    lane_block_mult: float = 3.0
    # 1.3 跳跃
    jump: float = 0.8
    jump_flat_mult: float = 2.5
    # 轮胎段抑制跳跃惩罚（必须允许跳才能翻越胎顶）
    jump_suppress_ranges: tuple = ((23.5, 27.2),)
    # 胎顶攀爬奖励：x 在轮胎段时，机身高度超过 0.30 的部分给正奖励
    climb: float = 1.5
    climb_x_range: tuple = (24.4, 27.2)
    climb_z_base: float = 0.30
    # 胎前磨蹭惩罚：低速且不给轮速时按步扣分
    tire_stall: float = 0.6
    tire_x_range: tuple = (24.4, 27.2)
    # 1.4 能耗
    energy: float = 1.0e-4
    wheel_torque_ratio: float = 0.3
    wheel_action: float = 0.2
    # 1.5 终止
    success_bonus: float = 200.0
    failure_penalty: float = -50.0


def _mean_sq(values) -> float:
    values = list(values)
    return sum(float(v) * float(v) for v in values) / max(1, len(values))


def progress_reward(vx: float, vx_cmd: float, w: RewardWeights) -> float:
    """按指令速度前进给正奖励；倒走 / 停住不得分。"""
    return w.progress * max(0.0, min(float(vx), float(vx_cmd)))


def upright_reward(roll: float, pitch: float, w: RewardWeights) -> float:
    """滚转 + 俯仰的姿态惩罚。"""
    return -w.upright * (float(roll) ** 2 + 0.5 * float(pitch) ** 2)


def height_reward(z: float, w: RewardWeights) -> float:
    """机身高度偏离参考值的软惩罚（带 deadband，避免正常起伏被罚）。"""
    excess = max(0.0, abs(float(z) - w.height_ref) - w.height_deadband)
    return -w.height * excess * excess


def smoothness_reward(action, prev_action, w: RewardWeights) -> float:
    """相邻两步动作差（抑制抖动）。"""
    if prev_action is None:
        return 0.0
    diffs = [float(a) - float(b) for a, b in zip(action, prev_action)]
    return -w.smooth * _mean_sq(diffs)


def torque_reward(tau_leg, w: RewardWeights) -> float:
    """关节力矩均方惩罚（基础能耗项）。"""
    return -w.torque * _mean_sq(tau_leg)


def _in_ranges(x: float, ranges) -> bool:
    return any(lo <= x <= hi for lo, hi in ranges)


def lane_reward(y: float, x: float, w: RewardWeights) -> float:
    """走线约束：偏离中线超过 deadband 后平方惩罚；砖块段加倍。

    这一项专门针对「砖块段绕到 y≈4.8 从旁边过去」的行为。
    """
    excess = max(0.0, abs(float(y) - LANE_CENTER_Y) - w.lane_deadband)
    if excess <= 0.0:
        return 0.0
    mult = w.lane_block_mult if BLOCK_X_RANGE[0] <= x <= BLOCK_X_RANGE[1] else 1.0
    return -w.lane * mult * excess * excess


def jump_reward(vz: float, x: float, w: RewardWeights) -> float:
    """跳跃惩罚：只罚向上的竖直速度；平地权重更高；轮胎段完全放开。"""
    if any(lo <= x <= hi for lo, hi in w.jump_suppress_ranges):
        return 0.0
    up = max(0.0, float(vz))
    if up <= 0.0:
        return 0.0
    mult = w.jump_flat_mult if _in_ranges(x, FLAT_X_RANGES) else 1.0
    return -w.jump * mult * up * up


def energy_reward(tau_leg, tau_wheel, wheel_action: float, w: RewardWeights) -> float:
    """能耗/轮速浪费惩罚：腿部力矩 + 轮毂力矩 + 轮速动作幅值。"""
    leg = _mean_sq(tau_leg)
    wheel = _mean_sq(tau_wheel)
    wa = float(wheel_action)
    return -w.energy * (leg + w.wheel_torque_ratio * wheel + w.wheel_action * wa * wa)


def climb_reward(z: float, x: float, w: RewardWeights, vx: float = 1.0) -> float:
    """胎顶攀爬奖励：仅在轮胎段且真正在前进时给，避免"站着白拿"。"""
    if not (w.climb_x_range[0] <= x <= w.climb_x_range[1]):
        return 0.0
    if x < 24.8 or float(vx) <= 0.2:      # 必须已经贴近轮胎且在走，站着不给
        return 0.0
    return w.climb * max(0.0, float(z) - w.climb_z_base)


def tire_stall_reward(vx: float, wheel_action: float, x: float, w: RewardWeights) -> float:
    """轮胎段停滞惩罚：只要在轮胎区不往前走就扣分（不看轮子转不转，防空转绕过）。"""
    if not (w.tire_x_range[0] <= x <= w.tire_x_range[1]):
        return 0.0
    if abs(float(vx)) < 0.25:
        return -w.tire_stall
    return 0.0


def terminal_reward(success: bool, w: RewardWeights) -> float:
    return w.success_bonus if success else w.failure_penalty


def step_reward(*, vx, vx_cmd, roll, pitch, z, vz, y, x,
                tau_leg, tau_wheel, wheel_action,
                action, prev_action, w: RewardWeights) -> dict:
    """返回分项字典（便于日志诊断），`total` 为总奖励。"""
    parts = {
        "progress": progress_reward(vx, vx_cmd, w),
        "upright": upright_reward(roll, pitch, w),
        "height": height_reward(z, w),
        "smooth": smoothness_reward(action, prev_action, w),
        "torque": torque_reward(tau_leg, w),
        "lane": lane_reward(y, x, w),
        "jump": jump_reward(vz, x, w),
        "climb": climb_reward(z, x, w, vx),
        "stall": tire_stall_reward(vx, wheel_action, x, w),
        "energy": energy_reward(tau_leg, tau_wheel, wheel_action, w),
    }
    parts["total"] = sum(parts.values())
    return parts


def weights_dict(w: RewardWeights) -> dict:
    return asdict(w)


