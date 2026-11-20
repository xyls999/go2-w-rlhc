# 全程 RL 微调（v5 赛道）——奖励函数与训练脚本

本目录是为「在冻结/已批准赛道上做全程 RL 微调」准备的可读实现，独立于 `aptrl_go2w` 仓库原有代码，不改动原有环境与测试。

- `course_reward.py`：新设计的奖励函数（基础项 + 走线约束 + 跳跃惩罚 + 能耗惩罚）
- `finetune_full_course.py`：从现有 checkpoint 出发的全程 PPO 微调脚本（单进程多环境 + GPU 更新）

## 1. 奖励函数

逐步奖励 `r_t`（物理 100 Hz 策略步，dt = 0.01 s）：

### 1.1 基础项（保留原有最少的几样）

| 项 | 公式 | 权重 | 作用 |
|---|---|---|---|
| 前进跟踪 | `r_prog = w_p * clip(vx, 0, vx_cmd)` | `w_p = 1.0` | 按指令速度前进；倒走不得分 |
| 姿态稳定 | `r_up = -w_u * (roll² + 0.5 * pitch²)` | `w_u = 0.6` | 抑制侧倾/俯仰 |
| 机身高度 | `r_h = -w_h * max(0, |z − z_ref| − 0.05)²` | `w_h = 8.0`，`z_ref = 0.36` | 防止趴地或跳起 |
| 动作平滑 | `r_s = -w_s * mean((a_t − a_{t−1})²)` | `w_s = 0.02` | 抑制抖动 |
| 力矩惩罚 | `r_tau = -w_t * mean(tau_leg²)` | `w_t = 1.5e-4` | 抑制发力过猛 |

### 1.2 走线约束（针对砖块段绕路）

```
r_lane = -w_lane * max(0, |y − 4.0| − deadband)²
w_lane = 20.0 (砖块段 27.4 ≤ x ≤ 29.6 时 ×3)
deadband = 0.12 m
```

含义：偏离中线 12 cm 以内不罚（普通小跑需要横向余量），超出后**平方增长**；砖块段权重 3 倍，专门堵住「绕到 y≈4.8 从旁边过去」这条路。原先脚本教师正是靠绕到 4.72~4.88 才躲开碎砖，这里用奖励把它按回中线。

### 1.3 跳跃惩罚（少跳、别乱跳）

```
r_jump = -w_j * max(0, vz)²          （vz = 机身竖直速度）
w_j = 0.8（平地 x < 24 或 x > 47 时 ×2.5）
```

台阶/独木桥/轮胎确实需要起跳，所以只罚**向上**的竖直速度，且平地加倍——把「能走就别跳」写进奖励。

### 1.4 能耗/轮速浪费惩罚

```
r_energy = -w_e * (mean(tau_leg²) + 0.3 * mean(tau_wheel²) + 0.2 * a_wheel²)
w_e = 1.0e-4
```
`a_wheel` 是动作第 30 维（轮速残差），避免用大轮速硬拽。

### 1.5 终止项

```
成功（x ≥ 100 且路线监控通过）: +200
失败（碰撞/姿态/超时）        : −50
```

### 1.6 总奖励

```
r_t = r_prog + r_up + r_h + r_s + r_tau + r_lane + r_jump + r_energy + r_terminal
```

## 2. 训练脚本

```powershell
$env:PYTHONPATH="E:\desktop-rl-apl\aptrl_go2w"
C:\Users\Administrator\miniconda3\python.exe E:\desktop-rl-apl\rl_finetune_v5\finetune_full_course.py `
  --init E:\desktop-rl-apl\aptrl_go2w\stage3_rl\experiments\20261119_temporal_v4_nav_r1\model_best.pt `
  --iters 200 --envs 6 --out E:\desktop-rl-apl\rl_finetune_v5\runs\v5_ft_r1
```

先冒烟（200 步、1 次更新，验证奖励与反传都通）：

```powershell
... finetune_full_course.py --smoke
```

产物：`model_XX.pt`（可直接被 `stage3_rl/evaluate.py` 的 `PolicyRunner` 加载）、`metrics.jsonl`（每轮奖励分项与通过率）。

## 3. 已知边界

- 从当前候选（纯观测教师蒸馏版，未见过种子约 0/16）出发做 PPO 微调，起步会很低；先看奖励分项是否按预期变化，再谈通过率。
- 这条赛道是固定几何，走线项把砖块段的「绕行」堵死之后，需要网络自己学会在碎砖上小步通过；如果不能，就说明该段必须继续用教师数据补样本（DAgger）。
