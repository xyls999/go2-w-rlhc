# APT-RL Go2-W：从已验证步态先验初始化 PPO Actor

## 状态

**用户已批准；基础 actor-bootstrap 已实现。** 本规格落实 actor-bootstrap 方向。设计只使用版本化候选数据/decoder，不改变最终赛道验收标准。

## 目标

解决新的 25-update flat PPO pilot 中，初始策略和训练后策略都在正常固定赛道起点附近 body-contact（均 0/3）的启动失败。先用在 MuJoCo 平地已验证的 trot latent 序列训练一个行为克隆 actor，再以该策略作为新 PPO pilot 的初始化。只有稳定性门槛通过后才启动 PPO。

## 现有证据与限制

- 版本：TO 数据 `stage1_to/dataset_periodic_20261113T153646Z/`；TVAE checkpoint/decoder `stage2_tvae/candidates/20261113T153754Z/`。
- trajectory-held-out、从各参考状态初始化的平地循环检查：trot 181/181、bound 88/88 无终止接触。该测量不是从标准 reset 起步。
- 固定 held-out trot latent 循环从普通平地 MuJoCo reset 出发，0.5m/s 和 1.0m/s 分别运行 1000/1000 步，无终止；1.5m/s 在第35步 body-contact。见 `results/diagnostics/20261113T134122Z_gaprough/prior-normal-reset-probe.json`。
- 上一轮从随机 actor 开始的 25-update flat pilot 未改善，已停止。旧 v21 与新 decoder 不配对，禁止作为新 decoder 的初始化。

## 设计

### 1. 示范数据采集

新增独立 bootstrap 工具，不改写旧数据、旧 checkpoint 或生产 decoder。使用上面的固定 TVAE/decoder 版本，在 `Go2WEnv` 普通 flat reset 下重放有效的 trot latent 周期；速度仅取 0.5、1.0m/s。保存每个策略步的原始 426 维 observation 与 teacher action，并保留 seed、速度、轨迹 ID、teacher source hash 和终止信息。

Teacher action 组成：TVAE encoder 对相位对齐参考窗口输出的 posterior mean（16 维 latent）；12 维 auxiliary 固定为零；trot gait logit 固定选择 trot。teacher 周期在模拟中循环，采集仅在实际 MuJoCo rollout 中完成；若发生非脚部终止则该演示标为失败，不进入克隆训练。每个输出写入新的空 run 目录。

### 2. Actor 克隆

在独立训练器中，用 actor 的确定性输出拟合 teacher 16 维 latent 与 gait logit，auxiliary 输出受零值目标约束；共同训练 `ExteroEncoder` 与 `Actor`，不更新 TVAE/decoder。按完整模拟 episode/seed 划分 train/validation，禁止相邻帧随机切分导致泄漏。记录 latent/gait/aux 的分项验证误差及 seed 级结果。

输出标准 PPO 可读的初始化 checkpoint（encoder、actor、critic、observation normalizer、`iter=0`、decoder path/hash、bootstrap/data provenance）。critic 可随机初始化；该 checkpoint 必须通过现有 decoder 哈希配对检查。

### 3. 预先约定的晋级门槛

**门槛 A（不训练 PPO）：** 对每个速度分别用至少 10 个未参与克隆的 reset seeds 做 1000-step 平地 MuJoCo closed-loop rollout，actor 只使用冻结/克隆输出。每个速度至少 9/10 回合完整运行 1000 步，无 body-contact/fall/off-lane；相对起点位移至少 4m（0.5m/s）及 8m（1.0m/s），全程在 lane 内。未通过即停止，不开始 PPO；先报告分项损失、失败 seed 和接触信息。

**门槛 B（小型 PPO pilot）：** 仅门槛 A 通过后，从 bootstrap checkpoint 启动一个新的、最多 25-update、4-worker、batch-target 4096、2 epochs 的 matched-decoder pilot。保存 `model_initial.pt`，在相同 decoder、相同 `pose-v1` seeds 下比较初始 checkpoint 与第25-update checkpoint 的 3 回合原赛道进度。只有 median max-x 严格增加且没有新增/提前的失败模式才晋级；否则立即停止，不自动延长。

最终验收仍是原赛道、`pose-v1`、100 回合至少 91/100 成功，并核验同一 checkpoint 的成功视频。

## 非目标与保护

- 不从 v21、旧 TVAE、未晋级 25-update gap/rough pilot 或失败的 prior-v5 pilot 续训。
- 不调奖励、不放宽 lane/gap/成功距离、不改变赛道或最终 91/100 标准。
- 不宣称真机部署；所有结果仅称 MuJoCo 仿真。
- 不覆盖既有数据、权重、评估、视频；失败输出保留并标记未晋级。

## 验证要求

- 单测：teacher action 维度/范围、完整 episode split 无重叠、空目录保护、checkpoint provenance、decoder mismatch 拒绝、确定性 seed 排序。
- 集成：用固定 decoder 采集至少一个成功周期；保存并重新加载 bootstrap checkpoint；验证 checkpoint 的 decoder hash 与 source manifest 完全相同。
- 物理 gate：按门槛 A 对未见 seeds 做完整 1000-step rollout。
- PPO gate：按门槛 B 同 seed 对比初始/训练后 checkpoint；失败即停。
- 每次运行保存命令、代码/数据/decoder/checkpoint SHA-256、seed 列表、episode JSONL 和汇总；不将验收评估与训练 pilot 混称。

## 自审

- 该方案只以平地 trot（0.5/1.0m/s）做 actor 启动示范；bound 从普通 reset 的开环测试仍会失败，因此不伪称 bootstrap 已解决跳跃或越障。
- 1000-step 两档稳定性与行进距离是明确可观测门槛；如果行为克隆不能从普通 reset 泛化，PPO 不会启动，避免再次消耗训练预算。
- Teacher 序列是相位驱动的开环参考，不等同于状态反馈专家；门槛 A 专门验证克隆策略从真实 observation 闭环执行，结果可能失败，应诚实停止。

## Evidence-driven follow-up amendment (2026-11-13)

The first bootstrap and mixed-stage PPO pilot passed their predeclared progress gate, but a second mixed-stage pilot failed its newly declared gate. On identical fixed-track `pose-v1` seeds 9043–9045, the v1 policy ran 1000 steps on the original track with an explicit 1.0m/s command (all three, about 9.3–9.7m displacement), while the fixed evaluator command of 1.5m/s produced early body contact around x=0.6m. The flat bootstrap demonstrations covered only 0.5/1.0m/s, so high-command behavior is the next isolated variable to test; this is evidence of command-conditioned brittleness, not proof of a unique causal factor.

A train-only `--command-speed` override was added, restricted to `train_stage=flat`, and unsupported speeds or evaluation use are rejected. It does not alter the official fixed-track evaluator or its 1.5m/s command. The fixed-1.5m/s 25-update flat pilot failed Gate C: same fixed-track seeds remained 0/3 and failed earlier. The failed pilot is retained, not extended.

Follow-up speed-response diagnostics found a narrow operating region: the bootstrap actor completed 1000 fixed-track steps at 0.8 and 1.0m/s on all three seeds; at 1.1m/s only one of three did so, and at 1.2–1.5m/s none did. A controlled 1.0-to-1.5m/s ramp over 500 steps let the v1 actor complete 1000 fixed-track steps on all three seeds (12.59–12.78m displacement), while the direct 1.5m/s reset did not. This is a diagnostic trajectory only; it does not change `pose-v1` or count as Gate A/B/C.

A train-only flat command ramp from 1.0 to 1.5m/s over 500 policy steps was implemented and tested; it does not change the official evaluator. The ramped-training 25-update pilot failed Gate D: on the official fixed 1.5m/s protocol, the same seeds regressed to about x=0.5m and failed earlier. The command ramp helped only as an explicit evaluation-time diagnostic intervention, so it is not used to claim protocol success.

An output-only auxiliary bias sweep then found a repeatable startup intervention: subtracting 1.0 from the four calf auxiliary action outputs (indices 18, 21, 24, 27; 0.2rad joint-target shift through the existing scale) let the matched v1 policy survive 1000 fixed-track steps at 1.5m/s on all three seeds. A versioned checkpoint with that bias progressed all three official pose-v1 seeds to x≈26.7m before `attitude` termination, compared with x≈0.5–0.7m before the bias. This is an initialization diagnostic, not a learned or successful controller.

Next bounded experiment: initialize a separate mixed-stage PPO pilot from the hash-matched, calf-bias checkpoint and run at most 25 updates. Official fixed-track evaluation remains unchanged at 1.5m/s from reset. New Gate E, declared before training, uses the same seeds 9043–9045: final median max-x >=30.0m, no seed may fail earlier than its calf-bias baseline, and no new failure reason may occur. If it fails, stop and diagnose; do not automatically retrain. Passing this progress gate still does not satisfy fixed-track success or authorize final evaluation/video.
