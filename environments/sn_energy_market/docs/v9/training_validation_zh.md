# 双层训练实施与验证报告

日期：2026-10-19。依据：[完整建模文档](modeling_complete_zh.md)第 12—14 节，以及当前六维 UC、四维 Consumer 环境接口。

本轮完成原生 RLlib 双层 PPO 训练、重新响应后的候选选择、消费者单边偏离搜索、完整恢复和独立 Actor 推理。新增算法集中在 `RLAlgo/src/raymarl/algorithms/stackelberg/`，场景及实验入口位于 `experiments/sn_energy_market/`。当前环境机制沿用已验收实现。

## 1. 接口复用与模块边界

| 组成 | 实现 | 职责与复用方式 |
|---|---|---|
| 算法配置 | `StackelbergConfig` | 配置种子集合、预算、网络、损失、评价和停止参数；复用原生 `AlgorithmConfig` 生命周期 |
| 算法编排 | `Stackelberg` | 消费者更新、UC 候选生成、匹配重新响应、接受／回退；复用现有 `NativeAlgorithmMixin` |
| 策略及价值 | `StackelbergRLModule` | UC 原始高斯、消费者条件 Beta；每个参与者独立 Actor 和 Value |
| 参数更新 | `StackelbergTorchLearner` | 独立 Adam、PPO 截断、价值损失、梯度裁剪；由原生 `LearnerGroup` 调用 |
| 两阶段采样 | `StackelbergEnvRunner` | 每行一个物理时段、GAE、下一公告价值、快照及 RNG 恢复 |
| 场景契约 | `ScenarioFactory`、`TrainingEnvironment` | 通过结构协议提供固定规格、公告、推进、状态保存／恢复与时序追加 |
| 纯数值函数 | `generalized_advantages`、`paired_gain`、`select_candidate`、`actor_parameter_change`、`stopping_status` | 接收数组或记录并返回计算结果，可独立测试和复用 |
| 市场适配 | `MarketTrainingEnv`、`MarketEnvFactory` | 连接能源环境、构造本地边界和消费者训练侧价值输入、生成固定 seed 场景 |
| 独立执行 | `ActorBundle` | 只加载 Actor，通过同一输出接口产生原始 UC 控制值及消费者物理行动 |

新算法包不导入 `snmarket` 或 `experiments`。`common`、现有 MAAC／MASAC／MADDPG／ISAC／MAPPO／IPPO、公共注册和导出入口均无修改。环境 `src` 与配置无修改。具体算法通过显式包导入启用，公共框架没有增加算法名称判断或能源场景分支。

接口采用当前模型的一套定义。新增函数、类型均无版本后缀；生产链中没有旧五维 UC、旧报价、备用供电或旧快照回退解释。原有旧输入拒绝测试继续执行。

## 2. 文档规则—实现—验证对应

| 文档规则 | 主要实现 | 验证证据 |
|---|---|---|
| §13.2 UC 一次输出六维内部变量，显式映射六维物理行动 | `rl_module.draw`、`MarketTrainingEnv.publish` | 未裁剪高斯概率、原始变量和物理动作分别保存、公告不推进时间 |
| §13.3 条件动作及联合概率 | `draw`、`evaluate_actions` | 物理区间 Jacobian、固定历史前缀重算、零宽坐标梯度、随机合法域 |
| §13.5 UC 聚合输入与消费者集中价值 | `compute_values`、`publish` | Actor／Value 梯度边界；个人队列和月量仅进入消费者训练接口 |
| §14.1 两阶段时序 | `collect`、`bootstrap_values` | 下一消费者价值来自下一公告；采样切片恢复环境与 RNG，不留下旧策略公告 |
| §14.2 个体 PPO／GAE | `generalized_advantages`、`compute_loss_for_module` | 手算截断目标、真实终止、切片、每物理期一次折扣、冻结网络和优化器 |
| §14.3 UC 候选生成 | `training_step`、`_adapt` | 消费者网络固定，UC 单独更新；新轨迹由当前公告策略生成 |
| §14.4 匹配重新响应 | `_snapshot`、`_restore`、`select_candidate` | 同一消费者参数及 Adam 状态、相同预算和 seed；当前 UC 作为基准参与 |
| §14.5 单边偏离 | `search_deviations` | 每次只训练一户；搜索结束恢复策略、优化器和环境 |
| §14.6 联合停止 | `stopping_status` | 偏离、UC 改进、参数 RMS、区间半宽同时检查；缺少搜索或区间不能通过 |
| §12.1、§14.8 连续目标及评价 | `evaluate` | 固定窗口实际回报用于排序；改变 Critic 不改变候选或偏离结果 |
| §18.3 恢复与回退 | 原生 checkpoint 接口 | 采样轨迹逐数组相同；恢复后续训参数和轮次结果相同；契约不匹配提前拒绝 |
| §21 数据切分 | `MarketEnvFactory`、实验入口 | 训练／验证／测试 seed 不重叠；测试阶段冻结参数；失败恢复不覆盖已有实验元信息 |
| 独立推理与部署 | `export_actors`、`ActorBundle` | 导出后不构造 Value，确定性 UC 输出与训练模块逐元素相同 |

训练批保留原始变量、物理行动、条件边界、零宽掩码和行为概率。金额在环境中结算；采集器直接使用单期奖励，累计功率指标转为电量时仅乘一次时长。

## 3. 测试、构建与文档核验

最终完整测试 **306 项通过，0 失败、0 跳过**，耗时 42.671 秒：

| 范围 | 通过数 |
|---|---:|
| 当前环境、机制与接口 | 207 |
| 既有算法、公共契约及生命周期 | 46 |
| 新增策略、采样、候选评价与原生算法 | 36 |
| 新增实验适配和入口 | 17 |

原生生命周期测试实际完成 CPU 训练、GPU Learner／CPU Runner 更新、消费者冻结检查、偏离回退、保存、恢复续训和 Actor 导出。恢复后的后续参数与轮次结果逐项相同。消费者分布测试覆盖 1,536 组正常／到期／零宽边界样本，并核验物理行动合法性及概率重算。

测试中的 14 条警告来自既有生命周期测试重复注册同名 Gym 环境，不影响结果。新增代码通过 Ruff。wheel 构建成功，其中 9 个算法模块与工作区源码逐字节一致，未打包实验或模拟器实现。

建模 Markdown 与离线 HTML 已同步。HTML 包含 **544 处数学渲染、30 个编号公式、41 张表、7 个代码块**；桌面及移动端没有页面横向溢出、数学错误或外部请求。新增训练协议和实施状态已截图检查。

本轮重新执行环境的 207 项回归；百万时段环境压力验证沿用提交 `33f27b7` 的[既有报告](full_validation_current_commit_zh.md)。此次学习验证与那次环境压力统计分别记录。

## 4. 实际训练及恢复验收

最终代码使用四位消费者、96 个时段、16 期移峰期限运行两轮双层训练。每组采用两个独立 seed：训练 42—43、验证 10042—10043、测试 20042—20043。每轮包含现有 UC 和两个新候选；所有候选从同一消费者参考状态开始。

| 实际执行 | 数量 |
|---|---:|
| PPO 更新调用 | 36 |
| 学习采集物理时段 | 2,304 |
| 候选及偏离验证物理时段 | 4,992 |
| 最终冻结测试物理时段 | 192 |
| 本次训练和评价合计 | 7,488 |
| 从最终 checkpoint 恢复后再次冻结评价 | 192 |

每次更新采集 64 个物理时段，使用 2 个 PPO epoch、32 行 minibatch。每轮消费者初始适应 2 次，每个候选重新适应 2 次，UC 候选各更新 1 次，逐户偏离各更新 1 次。

两轮均接受候选 2。该结果说明候选生成、消费者重新适应、收益比较和策略组恢复实际发生。最终联合停止条件未通过。冻结测试的完整逐 seed 回报、实际现金、服务、电量、消费者成本及偏离改善量见[机器汇总](training_validation.json)和实验输出；短预算结果用于实现验收。

两轮选中候选相对各自当轮基准的平均 UC 验证收益改善分别为 74.8828 和 41.6182 元等值。最终两个冻结测试场景的每回合均值如下：

| 测试指标 | 每个 96 时段回合的均值 |
|---|---:|
| UC 折扣回报 | −221.8970 元等值 |
| UC 实际现金利润 | −47.0906 元 |
| 全体实际服务电量 | 583.9151 kWh |
| 原生未供电电量 | 114.7209 kWh |
| 全体消费者累计成本 | 414.0609 元等值 |

这些指标同时保留现金、奖励和服务三个维度，可直接用于后续扩大训练预算及基线比较。

恢复测试使用相同学习契约和独立输出目录。恢复后的两个测试场景记录与原最终测试逐字段相同，评价没有执行参数更新。完整网络和优化器 checkpoint、Actor 独立导出以及每轮 JSON 均已生成。

## 5. 本轮发现及修复

| 问题 | 修复及复验 |
|---|---|
| 原生构建需要默认 RLModule 规格 | 补充算法自己的模块规格接口，完成原生生命周期测试 |
| 旧快照权重序号导致候选回退不能完整覆盖 Learner | 按完整候选快照恢复权重、Adam 及序号；用恢复续训逐参数一致性验证 |
| 条件零宽坐标的概率／熵计算出现非有限中间值 | 零宽坐标确定执行、概率贡献为零，分布计算使用安全内部点；梯度测试通过 |
| 采样切片若保留下一公告会混用更新前的 UC 行动 | 暂存环境和 RNG，仅为 bootstrap 创建公告后恢复；分段和整段行动流一致 |
| 第二轮日志包含冻结模块的空 RLlib 统计 | 仅汇总本次更新模块的真实损失和梯度记录；两轮入口严格 JSON 输出通过 |
| 默认 Ray 工作目录改变调用方路径 | 算法本地日志直接写指定输出目录，测试断言调用方工作目录不变 |
| 截断评价把候选自身 Critic 误差当成真实收益 | 实际窗口回报用于选择，带尾值估计独立记录；只改 Value 的连续／有限截断回归通过 |
| 零／负外层轮数导致空结果被访问 | 训练入口提前校验，纯评价仍允许零训练轮数 |
| 恢复失败前覆盖了旧实验元信息 | 恢复成功后才写元信息；模拟失败恢复验证原证据保持 |

必要检查集中在输入契约、阶段、学习数值和恢复边界。正常调用路径直接执行，没有添加旧机制兼容分支。

## 6. 经济问题与研究范围

U02 来源替代、A01 已有减载付费认领、A02 到期回补额外容量责任、A03 公告额度平坦区继续按当前环境公式参与学习和评价。本轮没有改写奖励或市场机制来改变这些经济性质。

本次结果验证实现、梯度、状态和运行链路。正式策略表现需要更长训练、更多独立 seed、基线对照和更强偏离搜索。当前两条验证轨迹下的正态近似区间及有限候选搜索作为诊断，不构成 SNE 或学习收敛证明。连续窗口评价需要同时报告窗口长度和尾部误差；训练 GAE 的 bootstrap 与实际评价回报分别定义。

## 7. 可复现命令与证据

在项目根目录执行：

```powershell
$env:TMP="$PWD/.tmp/market-training"
$env:TEMP=$env:TMP
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -c RLAlgo/pyproject.toml RLAlgo/tests environments/sn_energy_market/tests experiments/sn_energy_market/tests -q -p no:cacheprovider --basetemp .tmp/pytest-training-release --junitxml outputs/snmarket-training/validation/all-tests.xml
```

完整训练验收：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 experiments/sn_energy_market/train.py --smoke --outer-rounds 2 --horizon 96 --deadline 16 --consumers 4 --rollout-steps 64 --inner-updates 2 --response-updates 2 --epochs 2 --minibatch-size 32 --hidden-dim 32 --candidates 2 --output outputs/snmarket-training/validation/release-training
```

冻结恢复评价沿用上述学习预算及维度，将输出改为 `outputs/snmarket-training/validation/release-evaluation`，并追加 `--evaluate-only --resume outputs/snmarket-training/validation/release-training/final-checkpoint`。日常最小示例使用 `train.py --smoke`，接口详情见[训练说明](../../../../experiments/sn_energy_market/TRAINING.md)。

[机器验证清单](training_validation.json)记录源码、保护模块、JUnit、训练产物、checkpoint、Actor、wheel 和 HTML 的 SHA-256。运行产物位于 `outputs/snmarket-training/validation/`；其记录中的父提交用于定位起点，最终代码由本次中文 Git 提交保存。
