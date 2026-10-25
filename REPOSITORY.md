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

当前固定预算配置默认 3 轮、12 个 384 步学习批，使用完整月评价、γ=0.99 和正终端库存估值。季度轮转已进入原生采样器，能源数据工厂定义季度分组。生产候选规则保持；独立 D24 两组各 8 轮对照检查 z=1 与始终保留候选的影响。

V9 always 在专用入口显式启用四流各 192 步的阶段混合；Critic 批量推理、片段边界和 checkpoint 支持位于通用算法。默认单流协议不变。运行记录、缓存与新增实验输出均在 `D:/Codex/2026-10-26/raymarl-v9-critic/`，详细预算、验证和对照见 [V9 always](experiments/sn_energy_market/V9_ALWAYS.md)。历史一次性入口 `resume_optimized_study.py` 与 `verify_execution_optimization.py` 已归档，复用的候选种子迁移保留为 `continue_storage_study.change_candidate_seeds()`。

最近完成的原 2×2 为四组各 24 轮，SOC 版本优化续跑总计 72 轮；历史 72 轮旧协议结果与其来源依赖保留。旧一次性调度脚本与无用重复产物按本次清理清单归档，当前入口和结果由实验目录统一导航。

环境删除全局异常包装及重复入口检查，原异常直接抛出。动作可行域、能量与资金守恒、FIFO、终端、阶段、快照及幂等提交继续校验。算法中用于恢复环境、网络、Adam 和 RNG 的 `finally` 保留其隔离职责。

完整环境验证基线 `outputs/snmarket-validation/full-current-33f27b7-20261018/`、数据验证、数据集、运行时及其他算法保留。该基线绑定历史来源；本次验证范围由[环境审核](environments/sn_energy_market/docs/v9/implementation_audit_zh.md)记录。
