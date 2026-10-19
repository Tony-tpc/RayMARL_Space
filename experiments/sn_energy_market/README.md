# 能源市场实验入口

本目录组织能源市场的场景、训练、评价和环境验证。算法实现位于 `RLAlgo/src/raymarl/algorithms/stackelberg`；市场执行、结算和状态转移位于 `environments/sn_energy_market/src/snmarket`。

`train.py` 运行双层训练：消费者适应、UC 候选生成、重新适应后的匹配评价、候选选择与冻结测试。完整调用和实验协议见 [训练说明](TRAINING.md)。

真实数据的候选比较、字段映射、时间划分和获取条件见 [数据选型调研](DATASETS.md)。当前训练入口仍使用合成场景；实测数据接入由本目录的数据转换与场景工厂负责。

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

