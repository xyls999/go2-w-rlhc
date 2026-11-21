# v5 当前最佳策略（BC+DAgger，确定性执行）

- 权重: rl_rsl/runs/v5_bc_best/model_bc.pt （来源 tire_bc_dagger4/model_bc_r1.pt）
- 架构: temporal_residual_v4（30 维动作 + 相位观测），1024/256
- 32 seed 确定性评测:
  - >=25.4m: 25+/32
  - >=26.4m: 20/32
  - >=27.6m: 10/32
  - >=29.8m: 1/32
  - >=100m : 0/32
- 主要失败: body_contact 14（砖块/框下）, brick_bypass 9（横向偏出 27.4-29.6）
