# 能源市场训练

固定正式入口采用 V9 always：同一学习批混合四季度，各384步，共1536步；γ=0.997、Actor λ=0.97、Critic λ=1；在线PopArt、minibatch 32；UC与所有消费者的Actor/Critic均为两层128维隐藏层。Critic 2 epoch，参与阶段的Actor 1 epoch。训练种子11，从零运行72轮。

```powershell
.\.venv\Scripts\python.exe -B -m experiments.sn_energy_market.run
```

入口要求审查完成、测速通过、当前分支已提交且工作区干净。检查依据位于运行目录的 `verification/ready.json`，产物全部写入 `D:/Codex/2026-10-27/raymarl-v9-always128-72/`。已有正式运行会拒绝重复启动。

| 文件 | 职责 |
|---|---|
| `run.py`、`runtime.py` | 固定72轮生命周期、当前配置、种子、Ray与GPU运行时 |
| `train.py` | 通用场景入口、smoke、数据合同与当前checkpoint评价 |
| `training_env.py`、`profile_dataset.py` | 两阶段接口、月份数据和季度分流 |
| `training_trace.py`、`training_audit.py` | 真实执行预算、候选隔离、独立目标检查和逐时段记录 |
| `validate_policy.py`、`evaluation.py` | 冻结完整月策略评价和原奖励单位MC误差 |
| `run_io.py` | 原子状态文件、运行锁和源码哈希 |
| `plot_batch_losses.py`、`audit_soc_mode.py` | 当前日志绘图、冻结SOC条件响应检查 |
| `validate_v9.py`、`validate_dataset.py` | 环境独立参考和数据验证 |

训练机制与预算见 [TRAINING.md](TRAINING.md)，目标及日志见 [BATCH_CRITIC_TARGETS.md](BATCH_CRITIC_TARGETS.md)，数据来源、单位和标定见 [AUSGRID.md](AUSGRID.md)。本目录不保存历史实验过程和结果。
