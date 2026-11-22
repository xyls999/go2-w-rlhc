# v5 当前最佳策略（全程 BC + 松弛 FSM）

- 权重: rl_rsl/runs/v5_bc_best/model_bc.pt （= bc_full4 权重，解码器元信息已对齐配对 TVAE）
- 配对解码器: stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz
  sha256 = 5d7e8d8612874e511863eb5149378fef11f339a3c340e316d5a54f7c974d3b17
- PolicyRunner(ckpt, decoder_path=该 npz) 可正常加载（已验证）
- 32 seed 确定性评测（pose-v1, 0.5m 起跑）:
  - >=25.4m: 32/32
  - >=26.4m: 32/32
  - >=27.6m: 30/32
  - >=29.8m: 18/32
  - >=30.4m: 18/32
  - >=33.5m: 0/32
  - >=100m : 0/32
- 剩余卡点（已逐帧定位）: 30.4-30.7m 第一级台阶前的"爬行对齐"。
  教师用 4Hz 恒定 aux≈0.80 的爬行步态推进到 30.65 再起跳；
  策略在同一状态只输出 aux≈0.4，蹭不动。原因是逐帧 BC 对该周期性步态做了相位平均。
- 已加入的机制（当前 best 中表为零、不改变行为）:
  - hop_table/hop_wheel + hop_mask：按 (p_idx, hop_el) 查表的起跳相位替换
  - creep_table/creep_wheel + creep_mask：按 4Hz 时钟桶查表的爬行相位替换
  - rl_rsl/fit_hop_table.py：用教师数据闭式拟合上述表
