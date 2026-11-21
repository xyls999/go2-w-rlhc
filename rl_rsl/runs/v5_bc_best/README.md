# v5 当前最佳策略（BC + DAgger，确定性执行）

- 权重: rl_rsl/runs/v5_bc_best/model_bc.pt （来源 tire_bc_dagger6/model_bc_r1.pt）
- 架构: temporal_residual_v4（30 维动作，含相位/FSM 观测），1024/256
- 训练数据: 脚本教师全程示范 + 3 轮 DAgger（21-31m）
- 32 seed 确定性评测（从 pose-v1 0.5m 起跑）:
  - >=25.4m（过轮胎）: 30/32
  - >=26.4m: 20/32
  - >=27.6m（过轮胎段）: 12/32
  - >=29.8m（砖块段）: 8/32
  - >=30.4m: 0-3/32
  - >=100m : 0/32
- 失败原因: body_contact 14 / attitude 10 / brick_bypass 6 / off_lane 2
