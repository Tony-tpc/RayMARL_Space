# 能源市场双层训练与评价

训练直接执行当前建模文档第 12—14 节：UC 先发布六维物理调度，消费者根据公告生成四维行动；市场完成一个物理时段后，各参与者按自己的奖励学习。UC 更新较慢，每个候选都经过消费者重新适应和匹配场景评价。

## 1. 目录与调用

| 路径 | 职责 |
|---|---|
| `RLAlgo/src/raymarl/algorithms/stackelberg/` | 策略网络、条件分布、PPO Learner、两阶段采样、双层更新、快照及策略导出 |
| `experiments/sn_energy_market/training_env.py` | 将当前环境的公告和物理时段接口连接到采样器；构造训练侧消费者价值状态 |
| `experiments/sn_energy_market/train.py` | 场景切分、预算、运行目录、训练入口及最终评价 |
| `experiments/sn_energy_market/ausgrid.py` | 原始数据下载、计量质量审查、固定住户及年度数据转换 |
| `experiments/sn_energy_market/profile_dataset.py` | 通用数据清单、完整月窗口、训练集资源标定及场景工厂 |
| `environments/sn_energy_market/src/snmarket/` | 动作校验、出清、调度执行、任务队列、账本和奖励 |

最小完整运行：

```powershell
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/train.py --smoke --output outputs/snmarket-training/smoke
```

标准运行：

```powershell
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/train.py --outer-rounds 20 --output outputs/snmarket-training/main
```

恢复同一配置的运行：

```powershell
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/train.py --smoke --resume outputs/snmarket-training/smoke/final-checkpoint --output outputs/snmarket-training/continued
```

冻结策略评价使用 `--evaluate-only --resume <checkpoint>`，跳过参数更新和偏离训练。`--device cuda` 选择 GPU Learner；默认 CPU 和单个 Torch 计算线程，适合小网络与接口验证。Ray 生命周期由入口启动和关闭。

恢复时，模型、折扣目标、PPO 与候选响应设置、训练/验证/测试种子及场景规则必须匹配 checkpoint。设备和输出目录可调整，追加的外层轮数通过 `--outer-rounds` 指定。`MarketEnvFactory.get_contract()` 自动提供市场配置、种子归属、时序长度和合成曲线生成函数的源码 SHA-256；场景生成规则改变后会形成新的实验契约。

## 2. 默认场景与数据切分

标准场景沿用环境的设备、费率和成本配置，将有限期长度设为 288 个 15 分钟时段，移峰期限保留 96 期。这样前 192 期具有新移出空间，末段按文档关闭不能在终止前到期的新任务。`--config-dir` 使用指定环境配置；`--consumers`、`--horizon`、`--deadline` 可显式覆盖相应维度。

合成场景以日内负荷、DER 和 PG 价格曲线为基准，加入有界随机扰动。每个 seed 唯一确定一整条外生轨迹，候选之间使用相同轨迹。场景不改变各户成本系数、费率规则和设备额定参数，因此固定设置保持策略所需条件一致。这些合成轨迹用于实现验证和可复现实验，外生数据来源写入 `experiment.json`。

默认种子 `42` 的标准切分为：

| 集合 | 种子 | 用途 |
|---|---|---|
| 训练 | 42—49 | 消费者更新及 UC 候选生成 |
| 验证 | 10042—10049 | 匹配候选选择和偏离评价 |
| 测试 | 20042—20049 | 最终冻结策略的保留集评价 |

`--seed` 整体平移三组种子。测试集不参与参数更新和候选选择。最小运行每组使用两个种子、两位消费者、8 个时段、2 期移峰期限，从 17:00 开始以覆盖 DR 事件。

以上默认设置用于合成场景。选择 `--dataset datasets/ausgrid/processed/manifest.json` 后，使用固定 16 户的 Ausgrid 三年数据，按年度划分训练／验证／测试，各含 12 个完整月份。当前训练从月初启动，按月份顺序接续，真实月末才转入下一场景；验证和测试仍各有 12 个月初场景。步长为 30 分钟、期限为 48 步；默认 `gamma=0.9801`、`lambda_gae=0.9025`、采样长度为 64。PG、ESS、DR 和队列容量默认按训练年标定；显式提供 `--config-dir` 时保留所给容量。实测数据场景不接受截短月份的 `--horizon`，住户数量与清单保持一致。

数据场景每月月初采用既有明确初态。当前 UC／Consumer 策略从月初生成月内历史；更新网络后保留 SOC、任务、月累计 UC 用电、历史反馈和日历，普通批末保留 bootstrap，真实月末执行一次终端结算。`DatasetEnvFactory` 的显式月中规则暖启动仍供历史实验复核，当前默认入口及 `continuous_pilot.py` 不使用它。工厂合同绑定数据哈希、住户顺序、月窗口、配置、费率假设和相关代码，不依赖数据目录绝对路径。获取、清洗及时间口径详见 [Ausgrid 实验说明](AUSGRID.md)。以下命令仅验证数据与接口，不启动 Ray 或训练：

```powershell
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/train.py --dataset datasets/ausgrid/processed/manifest.json --validate-data --output outputs/ausgrid-validation/interface
```

## 3. 动作概率与信息输入

UC Actor 一次输出六维高斯参数，采集器保存未裁剪的原始变量 `z` 及其高斯 `log_prob`。策略输出层显式调用环境提供的纯函数 `map_uc_controls` 生成六维物理动作；随后环境校验和执行。PPO 重算原始 `z` 的概率，公告保持提交动作的意义。

实验入口通过 `model_settings()` 设置 UC 初始均值 `[0.95, 0.02, 0.5, 0.5, 0, 0.5]`、标准差 `[0.08, 0.05, 0.2, 0.2, 0.1, 0.2]` 和输出头权重尺度 `0.01`，使初始探索集中于较充分的 PG 公告、较少主动弃电及接近空闲的储能控制。六维均值和方差均继续参与训练；初始均值既不是物理下界，也不会覆盖网络后续输出。

Value 使用固定输出单位：UC 为 20、Consumer 为 10。网络输出乘该尺度后还原为环境奖励单位，价值损失按 `0.5 × mean(((V−target)/scale)²)` 拟合；GAE、bootstrap、实际奖励和候选评分继续使用原单位。报告中的 `value_rmse` 为原单位误差，`value_loss` 为缩放后的损失。

通用 Learner 的 `fit_values()` 接收调用方提供的固定回报目标，只更新相应 Value 及其已有 Adam 状态。完整终止轨迹的 MC 预热由 [修正后的小规模验证入口](CALIBRATED_PILOT.md)显式执行，`train.py` 不自动增加预热采样或优化预算。

消费者以自己的公告后观测生成削减、条件移峰、DER 来源比例和报价。采样记录保存本地 13 字段边界、条件区间和零宽掩码。条件区间使用本户队列、到期预留和柔性能力；零宽坐标执行确定值。

| 网络 | 输入 |
|---|---|
| UC Actor、UC Value | 文档固定顺序的 `H+20` 聚合观测 |
| 各消费者 Actor | 本户 `H+24` 公告后观测 |
| 各消费者集中 Value | UC 聚合观测与固定消费者顺序的全部公告后观测拼接 |

消费者集中价值输入仅由受信训练端构造。其中包含本期已发布的 UC 公告、逐户任务桶和月累计量；本期消费者行动与成交结果尚未发生。UC 网络接收的仍是聚合观测。固定 `signed_log1p` 输入转换和批内优势归一化随实验配置记录。

## 4. 一个物理时段与采样边界

`MarketTrainingEnv.reset(seed)` 返回 UC 决策前观测。`publish(z)` 冻结公告并返回消费者观测、本地边界、集中价值输入及映射记录，时间保持不变。`step(consumer_actions)` 执行出清和结算，时间推进一次。

UC 到消费者之间没有物理折扣；连续两个同类决策之间使用一次 `gamma`。下一消费者价值必须在下一次 UC 公告形成后计算。真正终止时追加环境定义的终端奖励并将 bootstrap 置零；采样切片保留合法下一状态的价值。

评价的 `returns` 使用实际窗口内累计折扣奖励。有限期默认完成整个回合；连续或显式截断评价使用固定长度匹配窗口。`bootstrap_estimated_returns` 另外报告加上 Critic 尾值的估计，不参与候选选择或偏离比较，从而使评价排序由实际奖励决定。连续目标的截断误差通过增加评价窗口检验；单期奖励绝对值有界于 `M` 时，长度 `L` 的尾部误差上界为 `gamma**L * M / (1-gamma)`。

快照包含当前阶段、公告、SOC、消费者队列、月累计量、完整环境记录和随机状态。`get_state()` / `set_state()` 支持在已发布公告的阶段恢复。连续适配器在数据耗尽时返回明确标志，可调用 `append_profiles()` 追加连续时序；不会循环旧数据。当前命令行实验仍采用有限月回合，采样切片和外层训练轮次之间连续保留状态。Runner 的 `trajectory_periods` 随选定分支恢复，不受 RLlib 聚合计数广播影响；Algorithm 的 `sampled_periods` 统计所有实际 PPO 分支预算。

## 5. 内外层更新与候选评价

每轮先冻结 UC，使用新采集的轨迹更新各消费者独立 Actor 和集中 Value。随后冻结消费者网络，采集 UC 数据产生上层候选。冻结消费者参数期间，输入公告仍随 UC 决策变化，因此消费者仍对不同公告生成相应行动。

消费者前置适应推进主轨迹，结束后保存公共快照。UC 提案从该快照开始，多个提案沿用逐次 UC 更新；提案采样位置不直接提交。候选集合包含现有 UC。每个响应分支恢复公共环境、观测、随机状态以及相同的消费者参数和 Adam，同时保留对应候选 UC 的参数及 Adam，获得相同响应预算。候选接受使用同 seed 验证场景的配对 UC 回报差正态近似区间（默认 `z=1.96`）：下界超过最小改善量才进入可接受集合，再选择平均改善最大的候选。选定策略组合及其响应分支终点一起提交；未选分支只计入计算预算。

验证、偏离搜索及 Critic 辅助采样与主轨迹隔离。`collect_value_data()` 只允许训练种子，成功或异常退出均恢复采样状态；随后 `fit_values()` 保留 Value 和其 Adam 更新，不能用旧 Learner 快照覆盖新拟合结果。连续实验也恢复 Value minibatch 打乱所消耗的随机状态。学习契约记录 `continuous_main_common_response_snapshot_selected_branch_v1`，旧重启协议 checkpoint 会被拒绝，历史实验产物保留。

当前固定小规模入口为 `continuous_pilot.py --stage 1` 和 `--stage 2`。第二次从第一次检查点接续；每次固定 3 轮、1,536 个全部分支 PPO 时段，主轨迹推进 768 步；每次额外用训练年 7 月的 1,488 步完整 MC 轨迹拟合 Value 32 epoch。两次累计主轨迹 1,536 步，穿过真实 7 月末。验证沿用四个月初 96 步窗口，并对四个完整验证月分日期层检查 Critic。入口只加载训练和验证 NPZ，测试集继续保留。详见 [连续采样小规模报告](CONTINUOUS_PILOT.md)。

消费者偏离搜索固定 UC 和其他消费者，仅更新被检查消费者。候选与偏离报告记录实际搜索预算和已发现的收益改善。最小运行的两条验证轨迹用于验收训练、回退、恢复和评价链路；正式比较应增加独立 seed 和偏离搜索预算，并报告区间近似与多候选选择条件。

每轮在候选生成前及最终选定策略组上分别搜索消费者偏离。最终策略组用于联合停止检查：消费者已发现改进、UC 候选改进、Actor 参数 RMS 变化和配对区间半宽同时达到阈值，并连续维持指定轮数，入口才提前停止。默认阈值为 `consumer_gain=0.1`、`uc_gain=0.1`、`policy_change=0.001`、`confidence_half_width=0.1`，连续轮数为 3；两类改进和区间半宽以元等值计，参数 RMS 无量纲。通过 `StackelbergConfig.training(stopping_tolerances=..., stopping_patience=...)` 配置，`stopping_patience=0` 使用固定外层预算。未搜索偏离或缺少区间时，联合停止条件不会通过。

算法通过 `ScenarioFactory`、`TrainingEnvironment` 结构接口连接场景，只要求公告、物理推进、状态恢复和固定输入规格。`paired_gain`、`select_candidate`、`generalized_advantages` 与 `stopping_status` 是独立函数。专用算法代码不导入 `snmarket` 或 `experiments`；现有六算法和 `common` 无需加入能源场景分支。

主要预算参数如下：

| 参数 | 含义 |
|---|---|
| `--outer-rounds` | 上层候选选择轮数 |
| `--inner-updates` | 每轮初始消费者适应更新数 |
| `--response-updates` | 每个候选的消费者重新适应更新数 |
| `--candidates` | 新 UC 候选数量，现有策略另行参与比较 |
| `--proposal-updates` | 每个新候选的 UC 更新预算 |
| `--deviation-updates` | 每个消费者单边偏离搜索预算 |
| `--rollout-steps` | 每批物理时段数 |
| `--epochs`、`--minibatch-size` | PPO 批内更新参数 |

## 6. 输出与可复现性

每次运行写入独立目录：

| 文件或目录 | 内容 |
|---|---|
| `experiment.json` | 市场配置、完整学习契约（含折扣、PPO、响应预算和候选选择参数）、种子切分、代码及建模文档 SHA-256、外生轨迹元信息 |
| `round-XXXX.json` | 每轮训练和候选选择结果 |
| `held_out_evaluation.json` | 保留测试集上的冻结策略表现 |
| `deviation_search.json` | 消费者单边偏离搜索结果 |
| `final-checkpoint/` | 可恢复完整训练状态 |
| `actors/` | 执行侧策略导出 |
| `result.json` | 最终输出路径和评价结果 |

金额与元等值奖励由环境统一结算。实验汇总现金不再乘 `delta_t`；功率指标转换成累计电量时乘一次时段小时数。折扣回报、实际现金和各户成本分别记录。

## 7. 验证入口

```powershell
.\.venv\Scripts\python.exe -B -m pytest experiments/sn_energy_market/tests -q
.\.venv\Scripts\python.exe -B -m pytest RLAlgo/tests -q
```

适配器测试覆盖公告时刻、下一公告的消费者价值状态、待执行公告快照、终端与连续数据耗尽、追加数据、种子匹配、信息边界、原始潜变量留存及账目单位。算法测试检查概率重算、GAE、独立参数更新、候选公平比较、快照恢复和导出执行。

独立推理复用相同输出接口：

```python
from raymarl.algorithms.stackelberg.inference import ActorBundle

actors = ActorBundle("outputs/snmarket-training/smoke/actors")
env = factory(seed)  # 同一市场设置下的实验适配器
observation = env.reset(seed)
decision = env.publish(actors.uc_controls(observation))
actions = {
    key: actors.consumer_action(key, decision.consumer_observations[key], decision.consumer_bounds[key])
    for key in env.consumer_ids
}
transition = env.step(actions)
```

`ActorBundle` 不创建 Value、优化器或模拟器；UC 输出原始控制值交给显式可行输出层，消费者输出当前合法域中的四维物理动作。
