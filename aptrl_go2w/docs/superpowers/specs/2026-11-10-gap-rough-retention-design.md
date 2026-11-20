# 0.2m 单沟与原赛道粗坡混合保留训练设计

## 背景与目的

`docs/superpowers/specs/2026-11-09-go2w-progressive-gap-curriculum-design.md` 已获批准。首个有效 25 轮单沟 pilot 的局部门槛从 v21 的 0/10 提升到 5/10，但原赛道 seed 9002–9004 在 x≈17–18m 的 rough_slope 段分别比 v21 提前约 8.6–9.5m 终止；该 pilot 未晋级。逐回合证据：`stage3_rl/experiments/20261109T135158Z/stage-decisions.jsonl`、`results/diagnostics/20261109T135158Z/pilot-single0p2-full/episodes.jsonl`。目标是在不降低原赛道难度的前提下尝试保留原粗坡能力，同时继续练 0.2m 单沟。

用户已确认本方案的核心：4 个 PPO worker 中固定 2 个练单沟、2 个练原赛道粗坡片段，只跑 25 轮，从原始 v21 热启动，不使用人工起跳动作。选择固定 worker 而非每回合切换，是为了不在每个 reset 重建大型 MuJoCo 高度场并降低缓存/摩擦状态风险；代价是每个 worker 的地形类型固定，故必须记录实际分配和每类样本数。

## 训练数据流

- 新训练 CLI 模式 `gap_rough_retain` 仅作为 learner 的 worker 分配模式，不是正式赛道或一个可直接构造的单环境地形。要求 `--workers 4 --epw 1 --gap-width 0.2 --train-track <原 NPZ> --probe-track <同一原 NPZ> --eval-protocol pose-v1`；只允许 4 个 worker，以固定 2:2 试验条件。
- worker 0、2 使用现有 `single_gap`，原高度场的训练专用首沟缩到 0.2m，局部起点 x≈24.5m、终点 x≥28.3m；worker 1、3 使用新增 `rough_slope_retain`，读取**同一份原 NPZ 高度场，不修改几何**，局部起点 x≈15.0m（前段起跑）、终点 x≥23.5m（未碰到下一沟）。两者都用相同的 PPO 网络、奖励、动作空间、失败判据和执行器限制；只改变训练回合的起终点/采样分配。
- 每个 worker 独立 MuJoCo 模型，reset 时从名义摩擦重新施加随机系数；绝不在单个 worker 内切换地形。`run_config.json` 记录 worker index、实际阶段、2:2 比例、NPZ/初始化检查点/源码哈希、随机种子、25 轮预算。新目录不可复用；旧无效 pilot 和 v21 不覆盖。
- 初始化检查点固定为 `stage3_rl/runs_v21/model_latest.pt`（SHA-256 `a9a6dab12c2417e15b105ebe7d7eb0f8c4c5ba07d01ee8fa237c88c42a42e43c`），不从仅达 5/10 且粗坡退步的单沟 pilot 继续训练。

## 预注册测量与决策

已用原 v21、相同的十个 seed 9000–9009 测 `rough_slope_retain`：**6/10** 通过，具体成功 seed 为 9001、9003、9004、9005、9008、9009；10 个实际初始姿态不同、原高度场不变。证据：`results/diagnostics/20261113T131910Z/v21-rough-retain/gate.json`。初始方案的绝对 9/10 门槛高于 v21 自身，故暂停训练并诊断；独立尝试 x≈12.0/13.5m 起跑分别只有 7/10、6/10（一次性诊断，不作为晋级门槛）。用户现已批准把**局部保留条件**改为对照这六个 v21 原本成功的 seed，候选必须六个全部继续成功且十个 seed 总通过数不低于 6/10。既有 0.2m 单沟 v21 门槛为 0/10，原赛道 `pose-v1` 10-seed 基线成功 0/10、中位最远 x=16.804418713455846m。

训练 25 轮后，以同一 checkpoint 分别评估：

1. 单沟 0.2m 十个 seed，必须 ≥8/10 安全落地且到 x≥28.3m；
2. 原粗坡局部十个 seed，v21 曾通过的 9001、9003、9004、9005、9008、9009 必须仍全部无失败到 x≥23.5m，且总通过数至少 6/10；这是对已会技能的保留，不是降低最终赛道验收；
3. 原 104m 赛道同一十个 `pose-v1` seed，中位最远 x 不低于 v21，任何配对 seed 的最远 x 不得比 v21 提前超过 5m；不得把十个回合或局部成功称为最终 >90% 验收。

所有条件同时满足才晋级下一难度。若任一不满足，保留日志和独立权重，明确停止；不自动延长训练或放宽 0.5m/0.4m 原沟、`x>=100m`、`|y-4|<1.9m`，也不覆盖原 `results/eval_summary.json`、`results/eval_best_traj.npz` 或旧视频。正式目标仍需独立 100 回合至少 91 成功及同 checkpoint 的成功录像；本次 25 轮不能保证达标。

## 验证顺序

粗坡局部高度场/起终点/seed 差异已有测试；接下来先写失败测试，再实现纯 worker 分配函数在 4 worker 时严格 `[single_gap,rough_slope_retain,single_gap,rough_slope_retain]`、错误配置在启动前拒绝；run_config 的逐-worker 分配；两个局部门槛和原赛道的报告严格引用同一候选检查点，并验证粗坡六个基线成功 seed 不得丢失。跑完整测试套件并复核原 NPZ、v21 哈希，再启动唯一一组 25 轮短 pilot，按上述固定条件写入新 `stage-decisions.jsonl`。
