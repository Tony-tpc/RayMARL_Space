# 保留的实验结果

更新：2026-10-25。按实际完成量整理，历史协议与新主流程分别记录。

2026-10-26 提交整理：[24轮2×2结果](SAMPLING_GATE_2X2.md)、[D24后gate/always八轮结果](ROLLBACK_COMPARISON.md)。本次删除73个已结束测试目录内的3,255个可重建文件（160.38 MiB），历史一次性续跑与性能核验入口已校验归档至 `D:/Codex/2026-10-26/raymarl-v9-critic/cleanup/`。所有模型、实验轨迹与评价依赖保留；修改前备份位于 `D:/Codex/2026-10-26/raymarl-v9-critic/backup/`。

| 实验 | 实际完成 | PPO 分支时段 / 提交轨迹时段 | 结果 |
|---|---:|---:|---|
| D24 回退对照，2026-10-25 | **两组各 8 轮，最终 MC 与模式评估完成** | 每组 12,288 / 6,144 | [最终对照](D:/Codex/2026-10-25/critic-actor-d-576-uc-actor/outputs/rotation-rollback-8rounds/comparison.md)、[核验](D:/Codex/2026-10-25/critic-actor-d-576-uc-actor/outputs/rotation-rollback-8rounds/final-verification.json) |
| 采样 × 门槛 2×2，2026-10-24 | A/B/C/D 各 **24 轮** | 每组 36,864 / 18,432 | [汇总](../../outputs/sampling-gate/20261024-seed11-2x2/summary.json)、[HTML](../../outputs/sampling-gate/20261024-seed11-2x2/report.html) |
| SOC 模式优化续跑，2026-10-23 | 总计 **72 轮、288 批** | **110,592 / 55,296** | [报告](../../outputs/ausgrid-hybrid-storage/20261023-soc-mode-72rounds-optimized/report.json)、[状态](../../outputs/ausgrid-hybrid-storage/20261023-soc-mode-72rounds-optimized/status.json) |
| SOC 模式，2026-10-23 | 10 轮 | 15,360 / 7,680 | [报告](../../outputs/ausgrid-hybrid-storage/20261023-soc-mode-10rounds/report.json) |
| 阶段合批，2026-10-22 | 10 轮 | 15,360 / 7,680 | [报告](../../outputs/ausgrid-hybrid-storage/20261022-phase-batch-10rounds/report.json)、[图表](../../outputs/ausgrid-hybrid-storage/20261022-phase-batch-10rounds/comparison10/report.html) |
| 历史旧协议，2026-10-20 | **72 轮、576 批** | 36,864 / 18,432 | [说明](YEAR_STUDY.md)、[原始报告](../../outputs/ausgrid-year-study/20261020/report.json) |
| 冻结联合策略 Critic，2026-10-21 | 3 组 × 3 种子 | 17,520 训练 / 5,904 验证状态 | [说明](FROZEN_CRITIC_ABLATION.md)、[报告](../../outputs/ausgrid-frozen-critic/20261021/report.json) |

SOC 72 轮由原 `20261023-soc-mode-72rounds` 的前 24 轮与 optimized 的后 48 轮组成，不能算成两个独立训练种子。前段目录继续保留。阶段合批 10 轮曾中断恢复，实际采集与协议有效预算不同，原报告保留这一差异。

原 2×2 检验的是采样顺序和 z 系数；新的八轮对照检验同一 D24 起点下候选保留规则，不包含模式头单独训练。训练种子仍只有一个。 八轮结果：gate / always 分别保留 4 / 8 轮，概率加权模式遗憾为 0.150637 / 0.145516，均仅略优于各自均匀参照；UC 完整 MC RMSE 升至 47.715 / 48.875。取消回退尚未解决模式头状态区分问题。冻结模式评价所需 `outputs/diagnostics/20261024-frozen72-order-mode/mode/` 完整保留。

更早的两次 10 轮结果仍保留报告、独立最终 checkpoint、Actor、数据、图表和冻结源码：[首次三模式](../../outputs/ausgrid-hybrid-storage/20261021-10rounds/report.json)、[长窗口](../../outputs/ausgrid-hybrid-storage/20261022-long-horizon-10rounds/report.json)。首次三模式的固定状态文件和长窗口报告仍被分析入口引用。两目录内部重复 Ray 运行文件已归档。

`20261022-phase-batch-72rounds` 仅完成 1 轮后停止，没有最终训练 checkpoint；[原状态](../../outputs/ausgrid-hybrid-storage/20261022-phase-batch-72rounds/status.json)、首轮报告、数据和冻结来源保留，其 Ray 运行日志另行归档。

本次共归档并从项目目录移出 352 个旧运行文件（414.33 MiB），以及 7 个已被当前主流程替代的一次性调度/测试文件。每个压缩归档逐文件校验 SHA-256 后才删除原件。[清理记录](D:\Codex\2026-10-25\critic-actor-d-576-uc-actor\outputs\rotation-rollback-8rounds/cleanup/summary.json)、[完整运行文件清单](D:\Codex\2026-10-25\critic-actor-d-576-uc-actor\outputs\rotation-rollback-8rounds/cleanup/archived-runtime.json)、[旧代码清单](D:\Codex\2026-10-25\critic-actor-d-576-uc-actor\outputs\rotation-rollback-8rounds/cleanup/removed-code.json) 支持恢复。旧代码另有完整修改前备份。

2026-10-23 的旧保留清单 [retained_runs.json](results/retained_runs.json) 仅代表当日快照，不再作为当前全目录哈希。数据集、环境验证基线、最新 2×2、最长训练及全部必要依赖维持原样。
