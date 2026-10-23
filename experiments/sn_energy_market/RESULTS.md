# 保留的实验结果

整理日期：2026-10-23。保留最近三次已完成储能训练、最长已完成训练、最新离线 Critic 对照，以及已停止长跑的真实进度。完整清单、删除数量和保留文件 SHA-256 见 [retained_runs.json](results/retained_runs.json)。

| 实验 | 实际完成 | 学习批 | PPO 分支时段 / 主轨迹时段 | 结果 |
|---|---:|---:|---:|---|
| 阶段合批，2026-10-22 | 10 轮 | 40 | 15,360 / 7,680 | [原始报告](../../outputs/ausgrid-hybrid-storage/20261022-phase-batch-10rounds/report.json)、[对比图表](../../outputs/ausgrid-hybrid-storage/20261022-phase-batch-10rounds/comparison10/report.html) |
| 长窗口，2026-10-22 | 10 轮 | 80 | 15,360 / 7,680 | [原始报告](../../outputs/ausgrid-hybrid-storage/20261022-long-horizon-10rounds/report.json)、[对比图表](../../outputs/ausgrid-hybrid-storage/20261022-long-horizon-10rounds/comparison10/report.html) |
| 首次三模式储能，2026-10-21 | 10 轮 | 80 | 5,120 / 2,560 | [原始报告](../../outputs/ausgrid-hybrid-storage/20261021-10rounds/report.json)、[对比图表](../../outputs/ausgrid-hybrid-storage/20261021-10rounds/comparison/report.html) |
| 历史最长完整训练，2026-10-20 | **72 轮** | **576** | **36,864 / 18,432** | [保留报告](YEAR_STUDY.md)、[完整原始结果](../../outputs/ausgrid-year-study/20261020/report.json)、[关键曲线](results/year_study_20261020/) |
| 冻结联合策略 Critic，2026-10-21 | 3 组 × 3 种子 | 每模型 20 epoch | 17,520 训练状态 / 5,904 验证状态 | [保留报告](FROZEN_CRITIC_ABLATION.md)、[原始结果](../../outputs/ausgrid-frozen-critic/20261021/report.json)、[日期指标与图表](results/frozen_critic_20261021/) |

阶段合批 10 轮在第 8 轮中断后恢复：15,360 是有效协议分支预算；实际采集 15,968 个时段，完成学习的时段为 15,744，每个 Critic 实际执行 1,968 步，其中有效协议预算为 1,920。中断和恢复记录保留在原目录。

`20261022-phase-batch-72rounds` 在 2026-10-23 按用户要求停止，实际只完成 1 轮：1,536 个有效 PPO 分支时段、768 个主轨迹时段，停止发生于第 2 轮。它没有最终 `report.json`，内存训练状态未保存；[status.json](../../outputs/ausgrid-hybrid-storage/20261022-phase-batch-72rounds/status.json) 和已有输出保留，不能计为完成 72 轮。

所有保留主实验的报告、checkpoint、Actor、逐批日志、逐时段数据、图表和冻结源码维持原样。历史 72 轮使用旧策略及目标，复现需使用原源码；各版本协议见 [TRAINING.md](TRAINING.md)。

更早的在线 Critic、批次 Critic、稳定性、LayerNorm 独立报告及重复图表已移除。两个早期 `report.json` 留作历史引用链及当前入口依赖；LayerNorm 目录仅保留冻结 Critic 所需的 Actor、checkpoint、最终验证、实验配置和报告。完整环境验证、数据验证、数据集与其他算法保留。
