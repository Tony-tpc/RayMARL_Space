# 代码库维护说明

当前工作分支为 `算法实现/双层训练与接口复用`，使用本地 Git 管理，不合并 `main`。代码、当前说明和必要回归测试纳入版本；数据、运行环境、实验日志、checkpoint、图表和过程报告位于版本库之外。

算法位于 `RLAlgo/src/raymarl/algorithms/`；环境位于 `environments/`，负责物理、状态、动作、出清与奖励；`experiments/sn_energy_market/` 负责公开月份特征、数据接入、训练配置、评价和审计。通用算法不导入能源场景。

正式入口为 `python -B -m experiments.sn_energy_market.run --output D:/Codex/运行目录 --rounds 36`。模型、预算和恢复契约见[训练协议](experiments/sn_energy_market/TRAINING.md)。不保留历史版本迁移、旧参数兼容入口或从实验目录导入正式实现的路径。

本次审查及清理清单位于 `D:/Codex/2026-10-29/v9-main-integration/`；两组36轮严格对照位于 `D:/Codex/2026-10-29/v9-month12-online36/`。原72轮最终checkpoint、M76起点、重要诊断报告、完整月监督数据和最近编码对照的关键状态保留。重复运行缓存、历史中间checkpoint和过程副本按清单清理。其他项目与 Python 运行环境不删除。
