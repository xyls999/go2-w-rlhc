# APT-RL 学习失败审计：不能把当前结果直接归咎于论文或单一奖励系数

## 范围与结论

用户要求解释 2+2 pilot 失败、奖励是否不对，以及原论文如何做到。此次仅对照源码、原文和已有权重做诊断；没有修改生产环境、奖励、赛道、数据集、权重，也没有继续训练。执行保持 inline。

**结论：已确认当前实现的奖励、TO 先验及 PPO 配置存在实质性偏差，不能再称为“与论文完全一致”。奖励确实有问题，但尚无单因素消融证明它独自解释失败。应先修复/验证学习链条，再考虑训练量。**

## 一手来源

- 原文 HTML（已直接获取正文及补充材料）：https://arxiv.org/html/2607.13579v1 。重点：Materials and Methods 中的 TO / representation learning / RL；Supplementary 中的 Table S3、Table S4、dataset-size ablation、feedforward-torque ablation。
- 原文表 S3：https://arxiv.org/html/2607.13579v1#Sx3.T3 。原文表 S4：https://arxiv.org/html/2607.13579v1#Sx3.T4 。
- 本地 `E:/desktop-rl-apl/paper_text.txt` 第 303–418、454–513、612–649、1531–1660 行包含译文和原始英文表格提取；与在线原文交叉检查。
- 本地实现：`stage1_to/trajectory_opt.py`，`stage2_tvae/train_tvae.py`，`stage3_rl/env.py`，`stage3_rl/ppo_train.py`。
- 可复算诊断：`results/diagnostics/20261113T134122Z_gaprough/learning-diagnosis.json`、`decoder-audit.json`、逐步 `reward-trace-*.jsonl` 及对应脚本。

## 1. 已证实：奖励不是论文表 S3 的忠实实现

| 项目 | 原文 | 当前 `env.py` |
|---|---|---|
| 线速度跟踪 | 权重 1.5、条件归一化、平均速度 | 权重 2.5、`max(abs(vx_cmd),1)` 归一化、当前速度（624 行） |
| 动作平滑 | 一阶和二阶动作差分 | 仅一阶（629–633 行） |
| 关节加速度 | `-1e-5 * sum(qddot^2)` | 额外 `/100`，且 `prev_dq` 只在 reset 赋值，之后未逐步更新（382、635–636 行） |
| 步态多样性 | `1-epoch/50000`、有限历史窗口 | `1-train_iter/250`、指数均值；pilot 更新索引从零重启（639–641 行；PPO main） |
| 足端间隙 | 0.25m 目标、速度加权、权重 -1.3 | 0.12m 单侧铰链惩罚、无相同速度权重、系数 -0.5（647–648 行） |
| 足端擦地 | 权重 -0.5、原文接触判据及水平速度 | 权重 -0.02、离地且低于 0.03m、三维足端速度（650–651 行） |
| 足端碰撞 | 表 S3 有独立 -5 项 | 未实现该独立 reward 项；腿部碰撞终止不能代替足端碰撞项 |
| 名义姿态 | 权重 -0.005、按命令水平速度归一化 | -0.015、除以 `1+vx_cmd`（667 行） |
| 机身晃动 / 同时接触 | -0.4 / -0.25，含原文条件 | -0.15 / -0.025，判据也不同（后续各项） |

这些差异不都等于 bug：例如较小的 Go2-W 不能机械照搬 HOUND 的 0.25m 抬脚目标。但它们需要被明确标为适配，并通过消融验证，不能称为原样复现。`prev_dq` 不更新则是确定的实现错误，不是合理的体型适配。

### 明确的局部课程奖励错位

`env.py:585–595` 失败终止时：

```python
rew -= 1.5
prog = max(0.0, min(1.0, (self.max_x - APPROACH) / SEG_LEN))
pb = 3.0 * prog
rew += pb
```

`APPROACH=2`、`SEG_LEN=8`，所以 x≥10 后已经拿满 `+3`。两个局部起点 x≈15/24.5 都在此范围内，局部失败的这两项净增量恒为 **+1.5**，不是预期的纯失败惩罚。三个抽查候选单沟回合（9000/9001/9009）真实终止 terms 均为 `termination=-1.5, progress_bonus=3.0`。注意：这是**终止附加项**为正，不表示包括所有项的终止步总 reward 总为正，更不能据此证明“撞沟是最优策略”；失败还丢失后续回报。

### 已量化的奖励占比（确定性只读重放，不代表训练期全部分布）

- 候选单沟 seed 9009 最远 28.184569887130113m，仍因 body_contact 失败。全回合每步均值：速度 +1.4612780643291574，足端接触速度 -0.4566089880974273，擦地 -0.32898519977600454，机身晃动 -0.09062220945615819；不能只凭“有惩罚”断言它压死了必要的起跳。
- v21 单沟 seed 9001 的 `bound` 选择占比 0.8620689655172413，仍失败；故“只要强制选 bound”不是已经获得证据的解决方案。选择解码器标签也不等于生成了有效起跳。
- 八个回合的逐步记录都显示 `prev_dq` 全程未更新；这验证了加速度项的历史状态错误。

## 2. 已证实：TO 先验存在结构问题，尚未验证为可执行技能

原文 §4.1 的核心不是“生成任意数组再拟合”，而是周期且物理有效、包含力矩的二维运动先验。原文优化包含 z、pitch、pitch-rate 周期误差，trot 的最大 pitch 正则系数为 0。

本地代码与数据：

1. `assemble_traj` 的 trot 分支令 `cF=cH=1`，前/后左右腿分别复制同一状态（354 行及后续映射）。实际 **914 条 trot 轨迹、27,499 帧的接触模式只有 `[1,1,1,1]`**，并非四腿对角交替接触的 trot 标签。
2. `solve_periodic` 明确不优化 z，代价也移除了 z 闭合项（285–298 行）；`gen_traj` 还对 trot/bound 都设 `lam2=1`（460 行），不等于原文。
3. 保存的 736 条 bound 轨迹最后一帧减第一帧 z 的中位数为 **+0.10687442786982651m**。最后保存帧是 T-dt，不能把它直接当成精确周期闭合误差；但足以说明需要做严格闭合审计。
4. 使用当前同一个生成器，额外构造一个**诊断样例**（不是改写数据集）：bound `t_st=.16,t_sw=.18,vz_scale=.8,vx=1.5,seed=42`，优化后精确积分一个周期的 `dz=+0.08666186735362735m`、`dpitch=+0.4624025264166949rad`、`dpitch_rate=+0.9565905327218879rad/s`。这证明优化器并不自动保证生成周期轨迹；没有证明所有训练样本都如此。
5. bound 数据中 **6.46180127924993%** 帧至少有一条腿的目标竖直跨度大于 0.213+0.2264m，按生成器自身平面 IK 假设，单是竖直方向就超出连杆总长。`leg_ik` 用 clip 处理不可达，而不是拒绝这些目标。不是完整三维机器人可达性证明，但明确提示数据约束缺失。

### 不是简单的“TVAE 完全没学到”

对原训练集随机抽取每步态 2048 个合法窗口，用保存的 encoder/decoder（eval 模式）只读预测：

- trot 取后验均值解码的 MAE **0.2866993248462677 N·m**，相对于每关节常数基线的 R² **0.9740398583047677**。
- bound MAE **0.1083516851067543 N·m**，R² **0.9982638141245599**。
- NumPy 导出与 PyTorch 解码最大绝对差约 0.00939/0.01920 N·m。

这些是 **held-in 拟合诊断**，不是测试集泛化或真实运动验证。它们不支持“神经网络没拟合数据”这一简单解释；更应审查被拟合的数据，以及 RL 是否选用了有用的 latent。

只读 rollout 的候选三个单沟样本里，解码器平均绝对力矩约 **1.18–1.25 N·m**，辅助偏置力矩 `KP*0.2*a_aux` 约 **7.64–8.87 N·m**。这是两个未合成项的幅值比较，不是接触做功/动力学贡献分解，不能单独证明先验无效；但与论文强调的“有效前馈力矩先验”差距值得优先核验。

## 3. 已证实：另一个训练环境扰动索引错误

`env.py:397–403` 明确采用 free-joint `[linear WORLD(0:3), angular BODY(3:6)]`。然而训练 kick（575–576 行）写入 `qvel[3]`、`qvel[4]`，注释却说 vx、vy。这实际加的是角速度而不是水平线速度。它只在训练启用，造成与所称“水平推扰”不同的分布。此次只记录，没有修复；不能把它定为此次失败的唯一原因。

## 4. 样本量与保持能力：本 pilot 不能等价于论文训练

原文 Table S4：horizon=50、batch=204800、minibatch=40960、gamma=.99。本 pilot：4 workers×1 env×21 chunks×50=**4200 samples/update**；25 更新总计 **105000 samples**，单沟 worker 只占 **52500**。整个 pilot 的新采样量仅为论文**一次 batch 的 0.5126953125**。小预算是事先约定的防浪费探针，不是完整复现训练，因此 25 次更新本身不能用来否定论文，也不能反过来保证加练一定成功。

PPO 也有未标为忠实复现的差异：本地 gamma=.997；minibatch40960大于本地实际batch，故每个epoch实际只有一个minibatch；未看到论文描述的 latent Gaussian KL 项和辅助/步态均值正则。原文数值不能只复制字段而忽略对应的样本规模。

2+2 仅复习 x≈15–23.5 的 rough_slope，不覆盖从起点到 x≈15 的状态分布。候选退步的 9001/9004 在 x≈10.89/11.57（rough_flat），9006 在 x≈1.10（起步）就失败。因此“粗坡局部 8/10”与“全程若干 seed 严重退步”并不矛盾；**局部保留不是全程技能保留保证**。改变策略后的分布漂移/干扰是合理解释，但具体由哪些参数/更新造成，尚无消融证据。

## 5. 原论文实际上依赖什么

1. 优化生成规模大、物理有效的状态—力矩轨迹（18万条、约556万帧），学习共享潜空间和独立 gait decoder。
2. PPO 联合学习 latent、auxiliary action 和 gait，使用有效前馈力矩先验，不只是“速度奖励让机器人自己猜跳跃”。
3. 两个平地级别之后接十级障碍课程，覆盖七类地形；使用规模大得多的并行采样和匹配的超参数。
4. 论文 S10 的**作者报告**：在其 high-step、速度0–6m/s、难度2–5的评估中，移除前馈力矩成功率 2.9%，恢复后94.6%。这是特定实验条件下的消融，不是本项目固定104m赛道验收，也不是我们独立验证的结果。
5. S8 的 1% 只比较表征编码器训练损失。原文：“using only a small fraction (1%) of the dataset leads to slower convergence and noticeably higher total loss.” 它没有证明“任意质量的1%轨迹能保持相同下游越障率”。`REPORT.md` 对此的等价替代论据不成立，应撤回；“完全一致”“19项按论文权重全部实现”也应修订。

换平台、换轮腿形态、增加轮速控制器，原论文的绝对奖励/动作与身体尺度不能自动保持等价。当前失败首先说明本地学习链条和适配尚未被验证，不能据此指责论文不实，也不能据此证明论文所有泛化主张。

## 6. 后续实现与验证（2026-11-13）

下列内容是审计后的新工作，不属于上文“此次只读”的原始审计范围：

- 在 `stage3_rl/env.py` 修复三个已确认实现错误，并以测试锁定：失败终止不再有绝对位置进度奖；关节加速度使用相邻策略步 `dq`；训练推扰写入 free-joint 世界线性速度 x/y，而非角速度。
- 在 `stage1_to/trajectory_opt.py` 修正 bound 前/后足的摆动相位/落足区间；正确生成 trot 的交替对角触地，而非全程四脚接地。增加高度/俯仰周期闭合、俯仰角速度、模型连杆可达性筛选；不可达目标不再静默 clip 后进入新候选集。已通过测试覆盖错误样本拒收、有效 bound 保留及 trot 接触模式。
- 旧 `stage1_to/dataset/` 和旧 TVAE/decoder 文件没有覆盖。最终数据版本为 `stage1_to/dataset_periodic_20261113T153646Z/`：trot 907 条/27,744 帧，bound 442 条/29,489 帧；bound 与 trot 都只覆盖 -0.5、0.5、1.0、1.5m/s。每条都通过生成器闭合/可达性筛选；1.5m/s bound 使用较短周期。完整采样/哈希在该目录 `dataset_manifest.json`。
- 最终 TVAE 候选 `stage2_tvae/candidates/20261113T153754Z/` 使用 trajectory-held-out split：trot/bound 验证窗口 5,137/5,788，torque MAE 0.2529/0.1076 N·m，normalized state reconstruction MSE 0.0060。checkpoint SHA-256 `cfe5d6ebd3d180c3dfb1a012fdfa18f8206e23438854ab654ebba5bcf1438adf`；decoder SHA-256 `5d7e8d8612874e511863eb5149378fef11f339a3c340e316d5a54f7c974d3b17`。前面若干候选因后续物理审查发现错误而被保留但弃用，勿用于训练。
- `results/diagnostics/20261113T134122Z_gaprough/validate_prior_mujoco.py` 将 held-out 单周期 latent 序列从各自参考状态注入平地 MuJoCo：trot 181/181、bound 88/88 没有终止接触。此证据只覆盖筛选后数据在理想初始化下的一个开环步态周期，不含从任务起点学会启动、横向控制、越障或持久策略。
- 匹配 decoder 的 25-update、4-worker flat pilot `stage3_rl/experiments/20261113T154146Z/prior-v5-flat-pilot/` 已按门槛停止。相同 decoder、相同 pose-v1 seeds 9043–9045：`model_initial.pt` 为 0/3、max-x 0.524/0.540/0.553m；`model_latest.pt` 为 0/3、max-x 0.520/0.540/0.547m；全部 body_contact。初始策略与训练后均停在起点附近，没有候选晋级，未追加训练。结果见 `results/diagnostics/20261113T134122Z_gaprough/prior-v5-flat-pilot/{initial,latest}-evaluation-v2/`。
- 单独的“正常 reset 起步 + 固定 held-out latent 序列”诊断 `prior-normal-reset-probe.json`：trot 0.5/1.0m/s 各完成其 29/31 步参考周期，bound 1.5m/s 于第40/57步 body_contact。这比参考状态注入更接近起步，但仍只是开环单周期，并未测试长期控制。它支持继续区分 trot 起步、bound 转换与 actor 探索，不能据此把失败归因于单个组件。
- PPO、串/并行评估现可显式指定 decoder，并将哈希写入 run config/checkpoint；加载时拒绝不匹配 decoder。以上历史完整测试日志 `test-suite-postprior.stdout.log` 为74项；后续 actor-bootstrap/ramp/auxiliary-bias 增加测试后，最新完整套件为103项，日志 `results/diagnostics/20261113T134122Z_gaprough/test-suite-aux-bias.stdout.log`。未执行最终100回合验收，未生成成功视频；复现目标仍未达成。

### 仍需优先解决

尽管筛选后的开环周期在 MuJoCo 中稳定，随机/未训练 actor 仍会在约 25 步内 body-contact；25 次 PPO 更新也未使起点进度改善。因此瓶颈已从“静态 TO 周期/可达性”缩小到“从正常 reset 状态启动并由 actor 选择/跟踪 latent/aux/gait”的链条，但尚无单因素归因。下一步应针对 actor bootstrap、latent 与 auxiliary 的分工/尺度、起步初态和终止奖励做预先设门槛的诊断设计；这会改变训练架构，不能通过盲目增加迭代解决。新 decoder 不可配旧 v21 actor；不得从旧的未晋级 gap/rough 或上述失败 pilot 续训。

## 建议的下一步（部分已执行，最终门槛未变）

优先级是 **实现正确性 → 先验有效性 → 分布/奖励消融 → 扩大预算**：

1. 对 `prev_dq`、扰动索引、局部终止奖励写失败测试，明确修复与适配边界；逐项对表 S3/S4 做真实等价性测试，而不是只测文件/参数存在。
2. 审核并重建真实 trot 接触时序、周期闭合与可达性约束；预训练验证应同时报告拟合误差及在 Go2-W 动力学中的技能可用性。不要直接覆盖旧数据或权重。
3. 在保持最终标准不变的前提下，诊断训练期各 stage 的成功样本、reward/advantage、起跳相位、离地/落地、动作饱和及 decoder/aux 分工；当前日志不足以声称训练从未采到成功跨沟。
4. 保留回放应覆盖实际退步的起步/rough_flat及后续全程，不把粗坡局部通过当成全局保持证明。新方案和预算须先确认。
5. 最后才做有预注册门槛的对照训练。不要同时改十几个奖励并加长训练，否则无法归因。

## 验证与限制

- `diagnosis-verification.log` 逐一核对了八次带诊断记录的 rollout，与原 gate 的 max_x、steps、reason、success **全部精确一致**，说明本次记录未改变所测轨迹。
- 同一文件确认九个受保护的原赛道、检查点、数据集及生产代码哈希未变。
- Decoder 审计首轮因双 OpenMP runtime (`OMP Error #15`) 中止，未生成有效结果；没有使用 `KMP_DUPLICATE_LIB_OK` 绕过。改为单进程 `MKL_THREADING_LAYER=SEQUENTIAL OMP_NUM_THREADS=1` 后同一诊断脚本成功；首轮与成功轮日志均保留。成功轮有 PyTorch nested-tensor 配置警告，不是训练结果。
- 先前58项通过只说明现有测试覆盖的行为通过，**不证明奖励等价、TO物理正确或任务已学会**。本审计没有修复生产行为，因此不能宣称上述新发现的bug已修复。

## Actor bootstrap follow-up (2026-11-13)

- 已按用户确认实现 `stage3_rl/actor_bootstrap.py` 与 `stage3_rl/validate_bootstrap.py`：从普通 reset 收集、trajectory-held-out 克隆和 decoder 匹配检查。第一批演示因第一帧 observation 仍带默认1.5m/s命令被标为 `INVALID.md` 并保留；修正后 `demonstrations-v2/` 在0.5/1.0m/s各20/20完成1000步。
- Bootstrap checkpoint `model_bootstrap.pt` 在未见平地 seeds 0.5/1.0m/s Gate A 各10/10完成1000步、分别至少4/8m；验证损失 latent MSE 0.007106、aux MSE 0.0001181、gait BCE 0.00002530。这仅是平地步行。
- 同三 `pose-v1` 固定赛道 seeds 9043–9045，bootstrap actor原样0/3、median max-x 0.514m；首个25-update mixed pilot达到0/3但median增至0.616m，失败仍为body_contact且变晚，故只通过预设的弱进度Gate B。第二个mixed pilot、固定1.5m/s flat pilot、训练期命令ramp pilot均失败各自门槛；所有进程均停在25 updates，没有自动续训。
- 诊断显示速度敏感：v1 actor固定轨道命令1.0m/s时3/3可运行1000步、位移约9.3–9.7m；1.5m/s从起点早期body_contact。固定1.5m/s训练与1.0→1.5 ramp训练都未使同条件固定赛道性能改善。外部ramp只是实验诊断，不改变官方评估命令或用于声称pose-v1成功。
- 额外人工辅助输出bias诊断：actor的四个calf auxiliary输出各减1.0后，三seed在官方固定1.5m/s赛道都从起点运行至约x=26.7m，随后attitude终止；它不是经过学习的策略。用该偏置初始化的又一个25-update mixed pilot退化到三个seed均未到x=30m，且出现更早/不同失败。单独把bound latent在gap前切入也未改善。
- 候选报告汇总 `results/diagnostics/20261113T134122Z_gaprough/actor-bootstrap-followup-diagnosis.json`；Gate B细节 `actor-bootstrap-pilot-v1-gateB.json`。最新源码完整测试 **103项通过**，日志 `test-suite-aux-bias.stdout.log`。固定赛道91/100目标仍未达到；没有最终视频。
