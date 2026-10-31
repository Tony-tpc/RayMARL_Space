# 代码库维护说明

工作分支为 `算法实现/双层训练与接口复用`，本地 Git 管理，不合并 main。提交代码、当前说明和必要回归测试；数据、Python 环境、checkpoint、实验图表和过程报告不入库。提交说明使用中文。

正式训练只使用 `python -B -m experiments.sn_energy_market.run --output D:/Codex/运行目录 --rounds 72`。`settings.py` 保存共享先验和库存估值，`validate_dataset.py`、`validate_v9.py` 与 `smoke_test.py` 分别负责数据、环境参考和短场景验证。旧通用 `train.py` 训练/恢复命令及 `training(hidden_dim=...)` 模型参数别名已删除，不保留转发入口。

`environments` 负责物理、状态、动作、出清和奖励；能源场景、数据接入、训练配置和评价属于 `experiments/sn_energy_market`。通用算法不导入能源场景。

2026-10-31 清理记录位于 `D:/Codex/2026-10-31/repository-cleanup-two-commits`。`cleanup-plan.json` 逐项记录删除范围、归档成员和保留哈希；旧实验结论与脚本集中保存在 `historical-reports-and-scripts.zip`，归档内保留原相对路径。历史脚本按对应 Git 版本使用，当前源码不提供旧接口兼容层。

保留原始 72 轮、物理摘要 72 轮、选定 M76、两组 CP24 及本次模式头两组 CP73/CP74 检查点。删除结束实验的临时运行目录、冗余中间检查点、状态缓存与旧过程副本。数据集、Python 环境和当前原始评价证据保留。

模式头修改基线提交为 `eec0044`：物理摘要 Critic、原始 PPO/GAE 模式梯度和 Adam，对应完整72轮检查点。当前版本在此基础上接入48小时条件模式回报和KL受限自然梯度，删除未采用的模式 Adam 分支。原环境、奖励、Actor观测与网络结构保持不变，额外仿真量独立记录。

原始72轮基线位于 `D:/Codex/2026-10-30/v9-physical-summary-72`；修复验证与第73/74轮检查点位于 `D:/Codex/2026-10-31/mode-credit-repair`。既有证据包含同源双臂两轮续训、独立模式分支检查及冻结测试年评价；单训练种子和两轮续训不代表收敛。
