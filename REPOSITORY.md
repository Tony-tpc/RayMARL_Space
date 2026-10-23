# 代码库维护说明

算法放在 `RLAlgo/src/raymarl/algorithms/`，可复用组件放在 `common/`；环境负责状态、动作、出清、物理和结算；实验负责数据、训练、评价、审计和输出。

| 入口 | 用途 |
|---|---|
| [v9 建模](environments/sn_energy_market/docs/v9/modeling_complete_zh.md) / [HTML](environments/sn_energy_market/docs/v9/modeling_complete_zh.html) | 数学规格与离线阅读 |
| [环境审核](environments/sn_energy_market/docs/v9/implementation_audit_zh.md) | 实现对应、差异和本次精简 |
| [实验入口](experiments/sn_energy_market/README.md) | 模块职责与命令 |
| [训练协议](experiments/sn_energy_market/TRAINING.md) | 采样、候选隔离、目标、恢复与预算 |
| [保留结果](experiments/sn_energy_market/RESULTS.md) | 最近三次完整储能实验、历史最长 72 轮、冻结 Critic 对照 |
| [交接说明](experiments/sn_energy_market/HANDOFF.md) | 当前状态与待处理事项 |

当前固定预算配置默认 3 轮、12 个 384 步学习批，使用完整月评价、γ=0.99 和正终端库存估值。最新阶段合批 10 轮已经完成；计划 72 轮的新储能运行只完成 1 轮后停止，历史 72 轮、576 批完整结果继续保留。

2026-10-23 按“最近三次完整储能训练 + 最长完整训练 + 最新离线对照”整理报告和输出。主实验原始目录、checkpoint、数据、图表及冻结源码维持原样；被引用的旧基线报告和冻结 Critic 来源模型仅作为依赖保留。完整保留 / 删除清单见 [retained_runs.json](experiments/sn_energy_market/results/retained_runs.json)。旧说明的 Git 历史仍可查询。

环境删除全局异常包装及重复入口检查，原异常直接抛出。动作可行域、能量与资金守恒、FIFO、终端、阶段、快照及幂等提交继续校验。算法中用于恢复环境、网络、Adam 和 RNG 的 `finally` 保留其隔离职责。

完整环境验证基线 `outputs/snmarket-validation/full-current-33f27b7-20261018/`、数据验证、数据集、运行时及其他算法保留。该基线绑定历史来源；本次验证范围由[环境审核](environments/sn_energy_market/docs/v9/implementation_audit_zh.md)记录。
