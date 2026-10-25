# 采样与候选保留实验支持

季度轮转已进入原生 `StackelbergEnvRunner`，本包只保留实验配置、被动审计、冻结评价和文件输出。`ExperimentRunner` 仅添加能源场景的物理游标报告，不再覆盖采样或恢复逻辑。

2026-10-24 的 A/B/C/D（月份顺序/季度轮转 × z=1.96/1）各 24 轮已完成，原协议与冻结源码保留在 `outputs/sampling-gate/20261024-seed11-2x2/`。历史复现应使用该目录冻结来源。

当前入口为 `experiments.sn_energy_market.rollback_comparison`：同一 D24 checkpoint 出发，gate 与 always 各 8 轮。完整命令、预算和指标见 [TRAINING.md](../TRAINING.md) 第 9 节。旧的一次性 2×2 启动、重试、并行协调、报告脚本及其协调器测试已归档，暂停续接与候选恢复由原生测试覆盖。
