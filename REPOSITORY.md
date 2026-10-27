# 代码库维护说明

当前工作分支为 `算法实现/双层训练与接口复用`，使用本地 Git 管理，不合并 `main`。代码、当前说明和必要测试纳入版本；数据、运行环境、实验日志、checkpoint、图表和验证报告存放于版本库之外。

算法位于 `RLAlgo/src/raymarl/algorithms/`；环境位于 `environments/`，负责状态、物理、动作约束、出清与奖励；`experiments/sn_energy_market/` 负责数据接入、固定训练配置、评价和运行审计。公共算法不导入能源场景。

正式训练入口为 `python -B -m experiments.sn_energy_market.run`。参数、四流语义、预算和恢复契约见 [训练协议](experiments/sn_energy_market/TRAINING.md)。数学模型见 [v9 建模](environments/sn_energy_market/docs/v9/modeling_complete_zh.md)。其他六种算法和其他环境独立维护。

本次运行产物根目录为 `D:/Codex/2026-10-27/raymarl-v9-always128-72/`。历史实验脚本、迁移入口、结果和原 outputs 已移到该目录的 archive；Git 历史保留。当前训练不得导入归档模块或读取历史 checkpoint。只允许从完全一致的当前学习契约恢复，新训练从随机参数、空 Adam、零 PopArt 统计和四条月初流开始。
