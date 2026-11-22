# v6 赛道当前最佳策略（全程 BC，确定性执行）

- 权重: rl_rsl/runs/v5_bc_best/model_bc.pt （来源 bc_v6_3）
- 赛道: results/diagnostics/robocon-material-preview-20261119-v6/scene.xml
  sha256 = 8d555e2265117624683d98e55c8a44e56522e2e8134657f9dea12bba2aa2d12f
- 32 seed 评测: 16/32 完成 100m（中位 98.5m，均值 69.2m）
  - >=27.6m: 26/32；>=33.5m: 20/32；>=40m: 20/32
- 失败: body_contact 6（轮胎/砖块）、timeout 6（95-97m 收尾及 41m）、attitude 4
- v6 与 v5 差异: 30-44m 平地化、独木桥 4m 宽 2.5mm 高、桥侧缺口 6.2m、钻框顶杆 0.63m
