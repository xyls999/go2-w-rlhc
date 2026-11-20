# APT-RL Go2-W 越障复现报告（骨架，训练完成后填充结果）

## 任务

复现 Science Robotics 2026 论文 **APT-RL（Action-Pretrained Transformer
Reinforcement Learning，arXiv:2607.13579）** 的完整训练流程，在本地仿真环境中
训练 Go2-W 四足轮腿机器人，最终在修复版
`go2w_race_track_seed50001.npz` 赛道上完成单方向越障任务（完成率 > 90%），
并产出演示视频。

## 论文流程 → 本地实现映射

| 论文阶段 | 论文设定 | 本地实现 | 文件 |
|---|---|---|---|
| 1. 冲量尺度 TO（§4.1） | 2D 矢状面 SRBD、Bezier GRF、周期优化；trot/bound 各 90k 条 | 旧数据集 `stage1_to/dataset/` 保留未覆盖，但审计发现其周期闭合、接触标签和可达性并不可靠。最终新候选数据采用周期/可达筛选：trot 907 条、bound 442 条；trot/bound 速度均只覆盖 -0.5、0.5、1.0、1.5m/s。不是论文 180k 轨迹的等价复现 | `stage1_to/trajectory_opt.py` |
| 2. TVAE 表征学习（§4.2） | 2层/8头/ff64/GeLU，latent 16，3帧重建 + 0.1·KL，冻结编码器训练力矩解码器 | 架构相近但不称“完全一致”；新候选使用按完整轨迹拆分的 held-out 验证，Go2-W 本体感知 78 维。其验证误差只证明对筛选后数据的拟合，不证明闭环技能可执行 | `stage2_tvae/train_tvae.py` |
| 3. RL（§4.3） | Isaac Gym PPO、29维动作、表 S3 奖励、课程训练 | MuJoCo 500Hz 物理 + 100Hz策略 + Go2-W PD；奖励并非表 S3 的逐项等权重忠实移植：速度项权重有差异，部分公式/项不同。旧 v21 与旧 decoder 配对；不能把 v21 checkpoint 当作新 decoder 的有效初始化 | `stage3_rl/env.py` |
| 4. 感知蒸馏（§4.4） | 教师 255 维外感知 → 32 维；学生 CNN-GRU 蒸馏用于真机部署 | 实现教师外感知编码器（255→128→64→32，训练中与 PPO 联合优化）；**跳过学生蒸馏**——本任务交付物为仿真视频，无需真机零样本迁移 | `stage3_rl/ppo_train.py` |
| Go2-W 适配 | 论文用 KAIST HOUND | 轮子速度控制器（τ_wheel = kw(ω_cmd − ω)，ω_cmd = v_cmd/0.086，仅触地时）——轮腿机器人在坡道/越障需要轮驱动牵引力，腿部运动仍由 APT 解码器+辅助动作学习 | `stage3_rl/env.py` |

## npz 赛道设计逻辑问题与修复（用户授权修改）

原 `go2w_race_track_seed50001.npz`（104m，13 段）的生成器存在缺陷：

1. **`stairs_cliff`（段10）/ `slope_cliff`（段11）：没有悬崖**。段名含
   "cliff" 但高度剖面是对称金字塔（升→降），从未生成垂直落差。
   修复：保留登顶台阶/坡道，将下降段替换为真垂直悬崖
   （stairs_cliff 落差 1.65m→0.80m，slope_cliff 1.22m→0.62m——1.65m 自由
   落体冲击超出四足机器人物理极限，取论文 HOUND 实测 90cm 落差范围）。
2. **`crawl`（段9）净高 0.345m 过苛**。Go2-W 躯干高 0.114m + 蹲伏腿部，
   余量 <5cm。修复：横杆下沿抬至 0.40m。
3. `stool`（段8）网格为"桌面+薄壁"分离结构 → 合并为实心箱体（语义一致）。
4. 其余段（gap 0.5m 深渊、continuous_gap 搓衣板、楼梯谷地/高地、1.9m 独木桥、
   粗糙地形）保持 npz 原始几何不变。

## 复现边界与诚实声明

- 仿真后端：Windows 本地 **MuJoCo 3.17**（论文用 Isaac Gym Preview；
  Windows 无原生 Isaac Gym，MuJoCo 通过 MjSpec 高度场网格精确加载 npz）。
- 论文的 180k 条 TO 轨迹不等于本地旧数据 1650 条；论文 S8 的约 1% 结果针对编码器表征损失，不是“1% 数据即可完成越障”的证据。
- 旧 TO 数据与旧 TVAE 继续保留。新筛选数据/TVAE 通过 held-out 单周期物理检查，trot latent actor-bootstrap 也通过了 0.5/1.0m/s 平地 1000 步闭环 gate；但这不等于越障技能或赛道验收。
- 奖励函数与表 S3 存在已审计差异；“19项全部按论文权重实现”的旧表述不准确，见 `docs/superpowers/research/2026-11-13-apt-rl-learning-failure-audit.md`。
- 跳过学生感知蒸馏（真机部署组件）。
- 训练命令速度域 [0.5, 2.5] m/s 单方向（任务要求"仅往一个方向走"），
  论文为 -1~7 m/s 全向。
- 成功判据：躯干 x ≥ 100m（赛道终点前 4m）、全程 |y−4.0| < 1.9m、
  无机身/腿部触地终止。

## 结果（尚未完成复现）

- 原始 v21 及此前 gap/rough pilot 未达到固定赛道验收；最近一次 25-update pilot 的原赛道 gate 为 0/10，不晋级。
- 2026-11-13 创建独立 TO/TVAE 候选：最终筛选数据位于 `stage1_to/dataset_periodic_20261113T153646Z/`；TVAE/decoder 位于 `stage2_tvae/candidates/20261113T153754Z/`。held-out 单周期 MuJoCo 平地检查通过 181/181 trot 与 88/88 bound 循环；这仅是从参考状态开始的开环单周期检查，不代表可从正常起始状态自主走路或越障。
- 行为克隆 actor-bootstrap checkpoint `stage3_rl/experiments/20261113T160601Z/actor-bootstrap/bootstrap-v1/model_bootstrap.pt` 与新 decoder 哈希匹配。普通平地 reset、未见 seeds 的 0.5/1.0m/s Gate A 为各 10/10、1000 步；这只验证平地行走。
- 第一轮 matched-decoder 25-update mixed pilot 在同三固定赛道 seeds 上 median max-x 从 0.514m 到 0.616m，仍 0/3 成功（body_contact），仅按预设进度门槛晋级。后续 mixed、固定1.5m/s flat、ramped-command pilots 均未通过各自门槛；未续训失败进程。
- 命令响应诊断：该策略在原赛道以 1.0m/s 命令可跑满1000步，固定1.5m/s 从起点早期 body-contact。外加四个 calf auxiliary 输出 -1 的诊断 checkpoint 可使相同三 seed 到约 x=26.7m 后 `attitude` 终止；随后25-update mixed pilot退化，仍0/3。此为诊断/候选，不是成功策略。
- 目标赛道 100 回合最终验收：尚未执行；成功视频：尚未生成。当前最近候选仍在约 0.5–26.8m 失败，远未到 x≥100m；任何小型 pilot 或手动动作偏置都不是最终验收。
- 复现判据仍为：原赛道、`pose-v1`、100 回合至少 91 次成功，并核验同一 checkpoint 的成功视频。

## 交付物

- `stage1_to/` 2D TO + 原始数据和版本化筛选数据
- `stage2_tvae/` 原始权重和版本化 TVAE 候选
- `aptrl_go2w/stage3_rl/` RL 环境 + PPO + 训练权重
- `aptrl_go2w/results/eval_summary.json` 评估结果
- `aptrl_go2w/video/` 演示视频
