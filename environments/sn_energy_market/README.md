# v9 Stackelberg–Nash 六维调度环境

环境按照 [v9 完整建模](docs/v9/modeling_complete_zh.md)实现 UC 公告、消费者响应、两层市场、储能与任务执行、DR 采购和结算。纯 NumPy 内核不导入 Ray 或 Torch；本项目提供环境与非学习验证。

## 运行

在 RayMARL 根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .\environments\sn_energy_market --no-deps --no-build-isolation
.\.venv\Scripts\python.exe -B -m pytest -c environments/sn_energy_market/pyproject.toml environments/sn_energy_market/tests
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/smoke_test.py --consumers 3 --horizon 288
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/validate_v9.py --workers 4
```

完整压力入口默认执行 100,000 个静态独立对照、10,000 次非法输入，以及 1,094,400 个动态物理时段。结果、失败重放输入、分批检查点和来源哈希存放在 `outputs/snmarket-validation/`。可用 `--plan-only` 查看工作量，`--resume <输出目录>` 续跑相同代码与配置的未完成批次，`--replay <失败JSON>` 重现失败。程序不创建训练器或进行梯度更新。

## 动作与两阶段调用

UC 物理动作顺序为 PG 供给额度、主动弃电、DER 充电、PG 充电、ESS 放电额度、DR 采购上限。消费者物理动作顺序为削减、带符号移峰、DER 意向比例、DER 报价。向量和数据类均按物理单位解释。

```python
from snmarket import EnergyMarketEngine, map_uc_controls, map_consumer_controls

engine = EnergyMarketEngine()
uc_obs = engine.reset(seed=42)
# 显式输出层：[储能模式, 六个连续控制]；模式 0 闲置、1 充电、2 放电。
# 连续控制均映射到 [0,1]；此处选择放电额度的 50%。
uc_action = map_uc_controls(
    [2, 0.8, 0, 0, 0, 0.5, 0.5],
    engine.state, engine.profiles.der_kw[0], engine.config,
)
consumer_obs = engine.apply_uc_action(uc_action)
bounds = engine.local_bounds()
actions = {
    identity: map_consumer_controls([0.2, 0.5, 0.5, 0.5], local)
    for identity, local in bounds.items()
}
record = engine.apply_consumer_actions(actions)
assert engine.state.period_id == 1
```

输出层函数属于行动生成辅助工具。引擎不会调用它修改提交动作：非法联合动作报错，合法行动按公告执行。可行域随状态变化，RLlib 的 Box 表示外包络；直接从 Box 随机采样不保证满足联合约束。

运行接口只有 `UCAction`、`ConsumerAction`。动作对象按字段名构造，例如 `ConsumerAction(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=0.5, der_bid_per_kwh=0.3)`；裸向量按文档中的当前物理顺序解释。规则版本仍绑定配置、快照和验证记录，用于阻止不同物理定义的状态混用。

每个物理时段分为两次调用：UC 阶段只保存公告，消费者阶段统一执行和提交。成交使用必要层优先的 DER PAB→UC 比例分配→DER 原报价补配。两路充电与主动弃电按行动值执行；实际购电、放电限于公告。

## 状态、观测和数据

- UC 观测为 `float32[H+20]`，仅包含资源、匿名需求与任务汇总及六个上期反馈。
- Consumer 观测为 `float32[H+24]`，包含本人状态和公开公告。
- 内核用 float64 计算；可行域校验使用浮点比较容差，SOC/任务的机器精度尾差不改变决策。
- `central_state()` 是受信诊断接口。调用方应仅把聚合 UC 观测送给 UC 网络；环境没有实现通信层的隐私协议。
- `Profiles(loads_kw, der_kw, grid_cost_per_kwh, flex_kw, rigid_kw=..., start_datetime=..., resolution_minutes=...)` 保存不可变时序。功率为 kW，价格为元/kWh。刚性负荷不超过基线，柔性能力可以在零基线时用于历史回补。
- CSV 使用 `timestamp,der_kw,grid_cost_per_kwh,consumer_i_load_kw`，可显式提供 `consumer_i_rigid_kw`、`consumer_i_flex_kw`。未提供时，合成约定为柔性=基线×配置比例，刚性=max(0,基线−柔性)。
- 个人队列上限、DER 报价上限和消费者成本系数支持统一标量或按消费者顺序的元组。DER 实际报价上限取配置上限与本人当前 UC 单价的最小值。
- `market.uc_high_intervals_hours` 与 `uc_peak_intervals_hours` 配置高价/峰价小时半开区间；其余时刻使用平价。默认分别为 9—15 时、20—22 时，允许多段和小数小时，区间不得重叠。
- `Profiles(..., metadata={"source": ..., "value_kind": "realized", "seed": ..., "split": ...})` 保存来源信息并计入指纹；追加数据保留各段元信息。元信息只说明数据来源，预测误差发生时序需由具体场景提供。
- `ShiftTask(source_period, deadline_period, remaining_kwh, task_id=None)` 支持显式任务 ID；reset 为未命名任务生成确定 ID，新移出任务自动生成 ID，实际回补保留原 ID。

每条 `ExecutionRecord` 保留决策前状态、UC/Consumer 观测、个人条件边界及完整结算。`supply_diagnostics` 分列原始最大供给、PG 与 ESS 保留额度、充电分流和主动弃电。该记录属于诊断数据，不作为 UC 的公开输入；观测与条件边界以可比较的数据结构保存，支持快照逐项一致性验证。

默认步长15分钟、H=96、有限回合96步；该组合按规则关闭所有新移出，因为任务无法在回合结束前到期。需要测试或研究移峰时，使用更长回合或较短延期窗口；压力测试覆盖这些组合。

## 终止、连续运行和恢复

`system.episode_mode="finite"` 在 horizon_steps 真正终止，追加一次电池存量差与待办终端项。末端观测携带最终状态和最后有效外生帧/真实公告，并标记 terminal。

`episode_mode="continuous"` 保留 SOC、任务和月累计，剩余期数为−1。数据耗尽后仍保留已结算状态；下一次 UC 行动抛出 `ProfileExhaustedError`。在 UC 阶段通过 `engine.append_profiles(连续的新Profiles)` 补充时序后继续，不重复旧数据。耗尽时返回的载体使用最后有效外生帧，适配器标记 `bootstrap_observation_valid=False`；追加数据并重新读取 `uc_observation()` 后才得到有效下一观测。采样切片由调用方承载，不在物理引擎中伪造终止或终端奖励。

`get_state()/set_state()` 保存和恢复阶段、公告、初始/当前电量、队列、月账本、反馈、记录及 RNG。规则、配置、时序哈希不同的快照不能混用。transition ID 保证同输入重试不会重复付款；错误直接抛出原异常；批量验证入口保存失败输入和快照上下文。

## 验证和经济解释

当前逐项审核见[实现审核](docs/v9/implementation_audit_zh.md)：物理及结算公式一致，公共日历尚未完整编码到观测；训练实际使用 γ=0.99。

`tests/v9_oracle.py` 独立实现数学参考，`tests/v9_checks.py` 将生产结果逐字段对照。当前完整环境验证基线见[报告](docs/v9/full_validation_current_commit_zh.md)。

当前模型保留 U02 来源替代、A01 已有减载付费认领、A02 到期回补额外容量责任、A03 公告容量平坦区。测试分别验证其数值结果；这些经济性质与程序守恒、实现一致性分开报告。双层 RL 已实现并完成一次固定预算训练与验证，见[结果索引](../../experiments/sn_energy_market/RESULTS.md)；尚未建立动态均衡结论。
