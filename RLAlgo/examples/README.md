# 单智能体与 MARL：完整使用示例

`train_compare.py` 可以直接运行，环境实现位于工作区 `environments/energy_env.py`。示例采用简化电池调度任务，演示完整接口与运行流程，不复现 Ye 2021 的 EV、空调、P2P 定价或论文实验结果。运行示例需要完整 RayMARL 工作区；独立安装的算法 wheel 仅提供 `raymarl` 包。

## 文件与阅读顺序

1. [environments/energy_env.py](../../environments/energy_env.py)：先看 `BatteryCommunity` 的物理逻辑，再看 `SingleEnergyEnv` 与 `MultiEnergyEnv` 如何适配不同接口。
2. `train_compare.py`：先看 `build_single/build_multi` 的配置差异，再看共用的 `run_case` 训练、评估、保存恢复和推理流程。

```text
BatteryCommunity：SOC、电池功率、负荷/光伏、电价、费用与容量惩罚
    ├── SingleEnergyEnv：obs/action/reward 是单个值或数组
    └── MultiEnergyEnv：obs/action/reward 按 house_id 组织成字典

build_single/build_multi
    → algo.train()
    → algo.evaluate()
    → algo.save_to_path()
    → 新建相同配置的算法并 restore_from_path()
    → 再次 algo.train()
    → 加载导出的模型文件独立推理
```

## 运行

在 PowerShell 中进入项目目录，使用现有虚拟环境，无需安装新的依赖：

```powershell
Set-Location D:\Yang\RayMARL
$env:PYTHONUTF8 = '1'
```

默认同时运行单智能体 PPO 与双住户 MAAC，训练三次：

```powershell
.\.venv\Scripts\python.exe .\RLAlgo\examples\train_compare.py
```

分别运行：

```powershell
# 原生 PPO，一个住户
.\.venv\Scripts\python.exe .\RLAlgo\examples\train_compare.py --mode single

# MAAC，两个住户，局部 Actor 与联合注意力 Critic
.\.venv\Scripts\python.exe .\RLAlgo\examples\train_compare.py --mode marl --algorithm maac

# 开启过去八步历史；MAAC Actor/Critic 将按当前库的 Encoder 使用 LSTM
.\.venv\Scripts\python.exe .\RLAlgo\examples\train_compare.py --mode marl --algorithm maac --history 8

# MAPPO，沿用相同能源环境，配置负责提供集中 Value 输入
.\.venv\Scripts\python.exe .\RLAlgo\examples\train_compare.py --mode marl --algorithm mappo

# 各住户优化自身费用；社区容量惩罚仍使收益相互影响
.\.venv\Scripts\python.exe .\RLAlgo\examples\train_compare.py --mode marl --algorithm maac --reward individual
```

也支持 `masac`、`maddpg`、`isac`、`ippo`。动作都是连续 Box，因此六种算法都能直接使用。完整快速接口检查：

```powershell
.\.venv\Scripts\python.exe .\RLAlgo\examples\train_compare.py --mode both --algorithm all --iterations 1 --horizon 8 --batch 16 --history 4
```

默认 CPU、本地 EnvRunner、本地 Learner。`horizon=48`、每步半小时对应 24 小时任务；快速检查中的 `horizon=8` 只是四小时短任务。一轮 `train()` 的采样量由算法决定，不保证等于一个完整回合。

## 环境实现的关键约定

- 一个智能体代表一个住户；本示例每户只控制电池。单住户环境与多住户环境共享同一套电池状态更新公式。
- 动作 `+1` 请求最大充电功率，`-1` 请求最大放电功率。执行前根据 SOC 和效率限制功率，`infos` 记录请求量与实际执行量。
- 局部观测包含 SOC、当前负荷、光伏、买卖电价、周期时间编码及剩余任务时间。其他住户的私人 SOC 不进入这个局部输入。
- 历史可选开启：`current[8]`、`history[H,3]`、`history_mask[H]`。历史为该住户过去的负荷、光伏、买电价；有效记录在前，右侧零填充。`observe()` 只读取数据，`advance()` 才追加历史并更新物理状态。
- 所有住户先执行当前时刻动作，再统一计算社区净进口和容量惩罚，避免遍历顺序改变物理结果。
- 费用包含购售电净费用、电池使用费用、容量惩罚份额和任务末尾的初始库存缺口惩罚。这些示例系数是演示设定，不是论文参数。
- `cooperative` 模式把负社区总费用分配给每个住户；`individual` 模式分别使用各住户负费用，容量惩罚使个体收益存在耦合。
- `horizon` 是有限时域任务终点，观测包含剩余时间，所以末尾 `terminated=True`，`truncated=False`。同时返回有效的最终观测。

## 历史、保存恢复与结果的区别

单智能体使用原生 PPO。带历史时，`FlattenObservations` 将所有字段展开，供普通 MLP 使用；它没有自动变成 LSTM 模型。MARL 使用当前 RayMARL 模型，其 Actor Encoder 会根据 `history` 字段建立 LSTM。MAPPO 的集中 Value 接收包装后的联合展开观测。

完整算法 checkpoint 用于继续训练。单智能体另导出原生 RLModule checkpoint，包含标准 PPO 模型；MARL 导出 ActorBundle，仅包含局部 Actor。独立推理使用首次训练结束时导出的文件，与随后演示的恢复续训属于两个分支。

恢复使用相同配置，重新构建算法后调用 `restore_from_path`。这个示例不序列化模拟器内部的未完成回合，环境从新回合继续；不能用它验证同一个未完成回合的逐步恢复。

默认输出位于 `outputs/usage-example/<唯一运行ID>`，包含简短 `report.json`、一个完整训练 checkpoint，以及推理模型。示例关闭自动周期 checkpoint，只保留本次显式演示的文件。每次运行的目录不同，避免覆盖上一次结果。

共同奖励复制给每个住户，因此 RLlib 的多智能体回合总回报会汇总这些奖励，不能直接当作负社区费用；物理费用分项请看 `infos[house_id]['individual_cost']`。不同智能体数量、奖励模式和采样预算的回报不能直接用于算法优劣排名。

## 已验证范围

2026-10-07 使用当前项目 Python/Ray/PyTorch 运行：单智能体 PPO，以及 MAAC、MASAC、MADDPG、ISAC、MAPPO、IPPO。设置为一轮初始训练、8 步任务、batch=16、history=4，所有情况均完成评估、保存、新建算法恢复续训和独立推理，推理均运行完整的 8 步。

短跑只证明示例接口与生命周期可执行，不证明训练收敛或能源论文收益。机器可读结果保存在 `validation.json`；验证产生的临时模型和 Ray 会话文件已清理。
