# 能源市场实验

当前固定预算入口为 [batch_critic_pilot.py](batch_critic_pilot.py)：月初连续采样，Actor GAE 与 Critic 多步目标分开，3 轮共 24 批，并记录真实逐批 loss。小规模结果见[报告](BATCH_CRITIC_PILOT.md)，实现口径见[目标与日志](BATCH_CRITIC_TARGETS.md)。测试集继续保留。

中长期实验复用同一配置执行 72 轮（576 批），在预先固定的轮次冻结验证；协议与结果见[中长期报告](YEAR_STUDY.md)。

| 模块 | 职责 |
|---|---|
| `train.py` | 通用合成／数据场景入口、预算、checkpoint、导出 |
| `batch_critic_pilot.py` | 固定预算配置与一次完整训练、验证、审计 |
| `evaluation.py` | 冻结窗口表现、完整验证月 MC、日期分层 RMSE |
| `training_audit.py` | 独立目标复算、物理账本及梯度／Adam 隔离检查 |
| `profile_dataset.py`、`training_env.py` | 月窗口、数据契约、两阶段训练适配 |
| `plot_batch_losses.py`、`summarize_batch_critic.py` | 读取已有结果、生成对照与曲线，不训练 |
| `validate_v9.py`、`validation_helpers.py`、`smoke_test.py` | 环境独立数值验证与固定行动 smoke |

算法属于 `RLAlgo/src/raymarl/algorithms/stackelberg`，状态转移和结算属于 `environments/sn_energy_market/src/snmarket`。完整参数见[训练协议](TRAINING.md)，真实数据接入见[Ausgrid 说明](AUSGRID.md)，数据选型依据见[调研](DATASETS.md)。

保留[上一轮对照报告](ONLINE_CRITIC_PILOT.md)、最新结果及对应原始输出。旧预拟合、旧连续采样实验和过程文档已删除，其源码与报告通过 Git 查看。现行入口不再导入历史实验。当前状态见[交接说明](HANDOFF.md)。

`MarketEnvFactory.get_contract()` 自动记录市场配置、种子切分与合成场景函数源码哈希。checkpoint 恢复保持学习目标和数据协议一致，设备与输出目录可以改变。

```powershell
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/train.py --smoke
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/train.py --outer-rounds 20 --output outputs/snmarket-training/main
```

下面两个入口使用固定或随机合法物理行动，独立验证环境机制。

```powershell
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/smoke_test.py --consumers 3 --horizon 288
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/smoke_test.py --consumers 8 --horizon 288 --mode random
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/validate_v9.py --plan-only
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/validate_v9.py --workers 4
```

`smoke_test.py` 完成一个时域并保存逐期账本，支持 CSV 和环境 YAML 配置。固定策略与随机策略都先显式生成合法物理动作，环境验证提交值。

`validate_v9.py` 的默认工作量为：

| 类别 | 数量 |
|---|---:|
| 静态生产结果与独立参考逐字段对照 | 100,000 |
| 非法输入与失败状态保持 | 10,000 |
| 常规完整回合 | 10,000 × 96 期 |
| 较长回合 | 100 × 672 期 |
| 跨月连续轨迹 | 20 × 3,360 期 |
| 动态物理时段合计 | 1,094,400 |

常规回合按 1、2、8、32、128 户分层；参数、时序、初始队列及行动分别由可复现种子生成。每期检查动作、市场、能量、服务、队列、DR、资金及观测。覆盖计数要求事件开关、零价、同价组、负 DR、两层成交、资源不足、到期能力不足、终止、跨月、追加数据和快照恢复实际发生。

结果保存在 `outputs/snmarket-validation/<运行ID>/`：`manifest.json` 绑定来源，`batches/` 保存原子分批结果，`failures/` 保存失败输入，`progress.json` 与 `report.json` 汇总进度和结论。

```powershell
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/validate_v9.py --resume outputs/snmarket-validation/<运行ID> --workers 4
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/validate_v9.py --replay outputs/snmarket-validation/<运行ID>/failures/<失败文件>.json
```

续跑要求代码、参考程序、配置及规格哈希一致。失败批次不计入通过数。修改实现后须重新启动验证，不拼接不同版本的结果。数学上保留的 U02、A01、A02、A03 在独立经济测试中复现，其状态不随守恒测试通过而改变。

环境接口、数据单位和生命周期见 [环境说明](../../environments/sn_energy_market/README.md)。
