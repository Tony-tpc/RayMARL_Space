# 代码库维护说明

工作分支为 `算法实现/双层训练与接口复用`，本地Git管理，不合并main。提交代码、当前说明和必要回归测试；数据、环境、checkpoint、实验图表和过程报告不入库。提交说明使用中文，目前未配置远程仓库。

通用算法只接受输入缩放参数，不导入能源场景。`environments`负责物理、状态、动作、出清和奖励；`experiments/sn_energy_market`负责聚合摘要、数据接入、训练配置和评价。正式入口为 `python -B -m experiments.sn_energy_market.run --output D:/Codex/运行目录 --rounds 72`，不保留旧月份输入、GRU切换或历史版本兼容入口。

当前审查与清理清单在 `D:/Codex/2026-10-30/v9-summary-integration`，72轮产物在 `D:/Codex/2026-10-30/v9-physical-summary-72`。保留原72轮、M76和两组CP24最终检查点，以及选定混合摘要的模型、固定数据、报告和关键失败证据。历史目录名 `v9-month12-online36` 沿用，但实际为各24轮。只清理已结束实验的缓存、冗余中间状态和旧过程副本；删除清单、保留文件哈希与统计写在D盘。数据集、Python环境、其他项目与回归测试不属于清理对象。
