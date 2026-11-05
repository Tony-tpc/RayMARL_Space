# 代码库维护说明

当前工作分支为 `冻结GAE训练`，基于 `eec004423ce381f43034ade19330e03c56fdfc70`，不合并 main 或改写原算法实现分支。提交代码、当前说明和必要回归测试；数据、Python 环境、checkpoint、实验图表和过程报告不入库。提交说明遵循 `AGENTS.md` 的四部分要求。

正式训练只使用 `python -B -m experiments.sn_energy_market.run --output D:/Codex/运行目录 --rounds 72`。`settings.py` 保存共享先验和库存估值，`validate_dataset.py`、`validate_v9.py` 与 `smoke_test.py` 分别负责数据、环境参考和短场景验证。旧通用 `train.py` 训练/恢复命令及 `training(hidden_dim=...)` 模型参数别名已删除，不保留转发入口。

`environments` 负责物理、状态、动作、出清和奖励；能源场景、数据接入、训练配置和评价属于 `experiments/sn_energy_market`。通用算法不导入能源场景。

2026-10-31 清理记录位于 `D:/Codex/2026-10-31/repository-cleanup-two-commits`。`cleanup-plan.json` 逐项记录删除范围、归档成员和保留哈希；旧实验结论与脚本集中保存在 `historical-reports-and-scripts.zip`，归档内保留原相对路径。历史脚本按对应 Git 版本使用，当前源码不提供旧接口兼容层。

保留原始 72 轮、物理摘要 72 轮、选定 M76、两组 CP24 及本次模式头两组 CP73/CP74 检查点。删除结束实验的临时运行目录、冗余中间检查点、状态缓存与旧过程副本。数据集、Python 环境和当前原始评价证据保留。

本版冻结UC Actor GAE所用的采样估值，保留Consumer拟合后GAE、物理摘要Critic和Adam。未接入联合return-to-go或自然梯度。完整72轮原始基线位于 `D:/Codex/2026-10-30/v9-physical-summary-72`，对应 `8330a3f`；冻结GAE单种子24轮证据位于 `D:/Codex/2026-11-05/eec0044-frozen-gae-24-round`，不与旧实验混算。

2026-11-05审查与清理记录位于 `D:/Codex/2026-11-05/frozen-gae-integration-review`。移除无调用的离线 `CriticProbe`，不保留旧GRU探针入口。保留RLlib的局部状态更新、可选环境快照优化、物理约束和精确恢复检查；这些是当前接口与科研一致性要求，不属于旧版本兼容层。
