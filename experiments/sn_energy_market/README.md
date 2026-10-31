# 能源市场训练

正式入口采用 V9 always 与条件模式回报更新：四季度同批，每流384步，共1536步。连续控制和Consumer Actor使用PPO/GAE；UC的静置、充电、放电模式使用训练环境中采集的条件回报和受KL限制的自然梯度。UC Actor为原68维、两层128维；UC Critic为68维加8维因果物理摘要、两层192维；Consumer网络保持两层128维。

```powershell
.\.venv\Scripts\python.exe -B -m experiments.sn_energy_market.run --output D:/Codex/运行目录 --rounds 72
```

这是唯一正式训练入口，使用指定轮数从零训练。运行前要求该输出目录的 `verification/ready.json` 与当前已提交的干净工作区一致。已有运行会拒绝重复启动，所有产物写入指定的D盘目录。

模式更新每轮额外探测24个训练状态，各可行模式重复4次、最多96个半小时时段；额外仿真量单独计数，不属于原PPO预算。市场、奖励、Actor观测和网络结构不变。通用 `StackelbergConfig` 仍默认关闭这项方法，当前能源实验在 `runtime.configuration()` 中显式启用。

新模式配置属于学习契约。旧PPO检查点必须先按旧配置恢复并核对完整状态，再显式迁移；不能直接绕过契约检查。已使用相同模式配置的检查点可原生恢复。短续训或单训练种子的改善不代表收敛。

| 文件 | 职责 |
|---|---|
| `run.py`、`runtime.py` | 指定轮数的原生生命周期、当前配置、种子、Ray与GPU运行时 |
| `settings.py` | 共享模型先验、终端库存估值 |
| `training_env.py`、`profile_dataset.py` | 两阶段接口、月份数据和季度分流 |
| `training_trace.py`、`training_audit.py` | 真实执行预算、候选隔离、独立目标检查和逐时段记录 |
| `validate_policy.py`、`evaluation.py` | 冻结完整月策略评价和原奖励单位MC误差 |
| `run_io.py` | 原子状态文件、运行锁和源码哈希 |
| `plot_batch_losses.py`、`audit_soc_mode.py` | 当前日志绘图、冻结SOC条件响应检查 |
| `validate_v9.py`、`validate_dataset.py`、`smoke_test.py` | 环境参考、数据及短场景验证 |

训练机制与预算见 [TRAINING.md](TRAINING.md)，目标及日志见 [BATCH_CRITIC_TARGETS.md](BATCH_CRITIC_TARGETS.md)，数据来源、单位和标定见 [AUSGRID.md](AUSGRID.md)。本目录不保存历史实验过程和结果。

模式头修改前的独立基线提交为 `eec0044`，保留原始 PPO/GAE 和 Adam 路径。当前版本删除未采用的模式 Adam 试验分支，相关历史证据保存在 D:/Codex 归档。
