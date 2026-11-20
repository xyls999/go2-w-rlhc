"""参考轨迹（手动/脚本教师示范）奖励：先靠它拿到完成率，再按轮次逐步退火移除。

参考数据来自 `v4_fullcourse_demos_v2_gclock.npz`（脚本教师在赛道上跑出的全程成功回合）：
  - obs: (T, 426) 原始观测
  - act: (T, 30)  教师动作
  - x:   (T,)     机身 x
  - ep_seed: (T,) 回合编号

从 obs 可无损恢复 y 与 yaw：
  ey  = -obs[16]              （环境令 vy_cmd = -ey）
  yaw = -(obs[17] + 2*ey)/1.5 （环境令 wyaw_cmd = -2*ey - 1.5*yaw）

参考轨迹按 x 分箱（默认 5 cm 一格），每格取所有示范回合的中位数，
得到 (y_ref, yaw_ref) 的时间无关参考；动作参考同样按 x 分箱取中位数。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

BIN = 0.05
X_MAX = 101.0
YAW_INDEX = 17
VY_INDEX = 16


def _recover_y_yaw(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ey = -obs[:, VY_INDEX]
    yaw = -((obs[:, YAW_INDEX] + 2.0 * ey) / 1.5)
    return 4.0 + ey, yaw


@dataclass
class ReferenceTrajectory:
    x_centers: np.ndarray      # (N,)
    y_ref: np.ndarray          # (N,)
    yaw_ref: np.ndarray        # (N,)
    act_ref: np.ndarray        # (N, 30)
    n_episodes: int

    @classmethod
    def from_demos(cls, npz_path: str | Path) -> "ReferenceTrajectory":
        ds = np.load(npz_path)
        obs, act, x, ep = ds["obs"], ds["act"], ds["x"], ds["ep_seed"]
        y, yaw = _recover_y_yaw(obs.astype(np.float64))
        nb = int(X_MAX / BIN) + 1
        buckets: list[list[int]] = [[] for _ in range(nb)]
        for i in range(len(x)):
            b = int(min(max(x[i] / BIN, 0), nb - 1))
            buckets[b].append(i)
        xc, yr, ywr, ar = [], [], [], []
        for b, idxs in enumerate(buckets):
            if len(idxs) < 5:
                continue
            idxs = np.asarray(idxs)
            xc.append((b + 0.5) * BIN)
            yr.append(float(np.median(y[idxs])))
            ywr.append(float(np.median(yaw[idxs])))
            ar.append(np.median(act[idxs], axis=0))
        return cls(x_centers=np.asarray(xc, np.float32),
                   y_ref=np.asarray(yr, np.float32),
                   yaw_ref=np.asarray(ywr, np.float32),
                   act_ref=np.asarray(ar, np.float32),
                   n_episodes=int(len(np.unique(ep))))

    def lookup(self, x: float):
        """返回该 x 所在的参考点（就近一格）与"参考是否可用"。"""
        if x < self.x_centers[0] or x > self.x_centers[-1]:
            return None
        k = int(np.clip(round(x / BIN - 0.5), 0, len(self.x_centers) - 1))
        return self.y_ref[k], self.yaw_ref[k], self.act_ref[k]


def reference_reward(ref: ReferenceTrajectory, x: float, y: float, yaw: float,
                     weight: float, act=None, act_weight: float = 0.0) -> float:
    """轨迹跟随奖励（负的加权偏差）。

    weight 由训练脚本按轮次退火：阶段一开始较大（先拿到完成率），后面降到 0，
    让策略完全由任务奖励（走线约束 / 跳跃 / 能耗 / 前进）驱动。
    """
    if weight <= 0.0:
        return 0.0
    got = ref.lookup(x)
    if got is None:
        return 0.0
    y_r, yaw_r, act_r = got
    r = -weight * ((y - y_r) ** 2 + 0.5 * (yaw - yaw_r) ** 2)
    if act is not None and act_weight > 0.0:
        r -= weight * act_weight * float(np.mean((np.asarray(act[:28]) - act_r[:28]) ** 2))
    return r


def anneal_weight(w0: float, it: int, total: int) -> float:
    """线性退火：it 从 1 开始，最后一轮降到 0。"""
    if total <= 0:
        return 0.0
    frac = max(0.0, 1.0 - (it - 1) / float(total))
    return w0 * frac
