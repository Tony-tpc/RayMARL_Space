# 能源市场训练

冻结GAE分支采用四季度同批，各384步，共1536步；γ=0.997、Actor λ=0.97、Critic λ=1；在线 PopArt、minibatch 32。UC Actor 使用原68维观测和128×128网络；UC Critic 使用额外8维因果物理摘要和192×192网络，Consumer 网络为128×128。UC使用采样时冻结的估值，Consumer使用拟合后估值；模式与连续控制仍由联合PPO概率和同一GAE更新。

```powershell
.\.venv\Scripts\python.exe -B -m experiments.sn_energy_market.run --output D:/Codex/运行目录 --rounds 72 --device cpu
```

这是唯一正式训练入口，使用指定轮数从零训练。运行前要求输出目录的 `verification/ready.json` 与当前已提交的干净工作区一致；已有运行会拒绝重复启动。完整72轮基线结果保存在 `D:/Codex/2026-10-30/v9-physical-summary-72`。

`--device` 显式选择 `cpu` 或 `cuda`，省略时保持原CUDA入口；CUDA不可用时直接报错，不静默改变设备。

| 文件 | 职责 |
|---|---|
| `run.py`、`runtime.py` | 原生训练生命周期、当前配置与运行时 |
| `settings.py` | 共享模型先验、终端库存估值 |
| `training_env.py`、`profile_dataset.py` | 两阶段接口、月份数据和季度分流 |
| `training_trace.py`、`training_audit.py` | 执行预算、候选隔离和逐时段记录 |
| `validate_policy.py`、`evaluation.py` | 冻结完整月策略评价及MC误差 |
| `run_io.py` | 原子状态文件、运行锁和源码哈希 |
| `plot_batch_losses.py`、`audit_soc_mode.py` | 日志绘图、冻结SOC条件响应检查 |
| `validate_v9.py`、`validate_dataset.py`、`smoke_test.py` | 环境参考、真实数据及短场景验证 |

训练预算见 [TRAINING.md](TRAINING.md)，数据合同见 [AUSGRID.md](AUSGRID.md)。历史实验与结果保存在 D:/Codex；旧通用训练 CLI 已移除。
