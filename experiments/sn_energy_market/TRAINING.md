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

以上默认设置用于合成场景。选择 `--dataset datasets/ausgrid/processed/manifest.json` 后，使用固定 16 户的 Ausgrid 三年数据，按年度划分训练／验证／测试，各含 12 个完整月份。当前训练从月初启动，按月份顺序接续，真实月末才转入下一场景；验证和测试仍各有 12 个月初场景。步长为 30 分钟、期限为 48 步；默认 `gamma=0.99`、`lambda_gae=0.97`、采样长度为 192。PG、ESS、DR 和队列容量默认按训练年标定；显式提供 `--config-dir` 时保留所给容量。实测数据场景不接受截短月份的 `--horizon`，住户数量与清单保持一致。

数据场景每月月初采用既有明确初态。当前 UC／Consumer 策略从月初生成月内历史；更新网络后保留 SOC、任务、月累计 UC 用电、历史反馈和日历，普通批末保留 bootstrap，真实月末执行一次终端结算。`DatasetEnvFactory` 每个 seed 只对应一个完整月，不再提供月中规则暖启动。工厂合同绑定数据哈希、住户顺序、月窗口、配置、费率假设和相关代码，不依赖数据目录绝对路径。获取、清洗及时间口径详见 [Ausgrid 实验说明](AUSGRID.md)。以下命令仅验证数据与接口，不启动 Ray 或训练：

```powershell
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/train.py --dataset datasets/ausgrid/processed/manifest.json --validate-data --output outputs/ausgrid-validation/interface
```

## 3. 动作概率与信息输入

UC Actor 使用 RLlib `TorchCategorical`、`TorchDiagGaussian`、`TorchMultiDistribution`：先生成 PG、主动弃电和 DR 额度，再按资源掩码选择闲置／充电／放电，最后生成模式内功率。闲置三项储能功率均为零；充电分别生成 DER 和 PG 功率；放电只生成额度。DER 充电先取自身可行区间，PG 再取剩余充电空间，消除共同符号／强度门控，同时满足两路共享的物理上限。空电屏蔽放电，满电屏蔽充电。

采集器保存模式、六维原始控制、边界、掩码和联合行为概率。`ActiveDiagGaussian` 只为官方高斯增加未激活坐标的点质量处理，这些坐标不计入 logp 或梯度。PPO 使用存储的模式和前缀重算概率，不重新采样。`map_uc_controls([mode, *raw_latents], ...)` 映射到原 v9 六维物理动作，Consumer 分布保持原实现。详见[建模第 13.2 节](../../environments/sn_energy_market/docs/v9/modeling_complete_zh.md)。

实验入口通过 `model_settings()` 设置六个连续坐标初始均值 `[0.95, 0.02, 0.2, 0.2, 0.5, 0.5]`、标准差 `[0.08, 0.05, 0.2, 0.2, 0.2, 0.2]`、输出头权重尺度 `0.01`，模式偏置初始化为零。有效模式初始概率接近均匀，后续由 PPO 学习。第 5 个连续坐标只控制非负放电额度。

Value 使用固定输出单位：UC 为 20、Consumer 为 10。网络输出乘该尺度后还原为环境奖励单位，价值损失按 `0.5 × mean(((V−target)/scale)²)` 拟合；GAE、bootstrap、实际奖励和候选评分继续使用原单位。报告中的 `value_rmse` 为原单位误差，`value_loss` 为缩放后的损失。

Critic 的两层隐藏网络使用 `Linear → LayerNorm → Tanh`，UC Actor 改为三模式条件头。LayerNorm 按单个状态的隐藏维归一化，无跨批运行统计；已完全移除 UC 时间特征支路。逐批诊断保留预测／目标／实际多步奖励／bootstrap 分布、Tanh 饱和比例、裁剪前梯度范数及环境指标。

正常训练每批先更新全部 Value，再更新当前阶段选中的 Actor。`--critic-epochs` 独立控制 Value 轮数，普通训练默认为 4，`--smoke` 默认为 1；`--epochs` 控制 Actor 轮数。Actor 使用 `lambda_gae=0.97`，训练与全部评价共用 `gamma=0.99`；独立的 `--critic-lambda` 默认为 1.0，Critic 使用本批实际多步回报，批末以采样时的旧 Value bootstrap，真实终止尾值为零。Critic 目标及行为概率在采样后固定；Critic 拟合后，在保存的当前与下一决策输入上重算 GAE，对整个学习批归一化一次，随后在 Actor epoch 间固定。消费者下一价值输入保留原 UC 公告，重算不重新抽取公告，也不改变 RNG、采样位置或价值回归目标。

`critic_lambda` 写入学习契约，避免恢复时静默改变目标定义。每批全量固定目标的更新前后 loss、原单位 RMSE、实际参与更新的 Actor loss，以及候选回滚结果，实时写入运行目录下 `batch_losses.jsonl`。逐批绘图与字段定义见[目标分离与逐批日志](BATCH_CRITIC_TARGETS.md)。

当前训练只使用已采集批次进行在线更新。额外 MC 预拟合接口和历史脚本已删除；完整月 MC 仅在 `evaluation.py` 中用于冻结验证，验证数据不进入梯度更新。

消费者以自己的公告后观测生成削减、条件移峰、DER 来源比例和报价。采样记录保存本地 13 字段边界、条件区间和零宽掩码。条件区间使用本户队列、到期预留和柔性能力；零宽坐标执行确定值。

| 网络 | 输入 |
|---|---|
| UC Actor、UC Value | 文档固定顺序的 `H+20` 聚合观测 |
| 各消费者 Actor | 本户 `H+24` 公告后观测 |
| 各消费者集中 Value | UC 聚合观测与固定消费者顺序的全部公告后观测拼接 |

消费者集中价值输入仅由受信训练端构造。其中包含本期已发布的 UC 公告、逐户任务桶和月累计量；本期消费者行动与成交结果尚未发生。UC 网络接收的仍是聚合观测。固定 `signed_log1p` 输入转换和批内优势归一化随实验配置记录。

## 4. 一个物理时段与采样边界

`MarketTrainingEnv.reset(seed)` 返回 UC 决策前观测。`uc_bounds()` 提供决策前资源边界，`publish([mode, *z])` 冻结公告并返回消费者观测、本地边界、集中价值输入及映射记录，时间保持不变。`step(consumer_actions)` 执行出清和结算，时间推进一次。

UC 到消费者之间没有物理折扣；连续两个同类决策之间使用一次 `gamma`。下一消费者价值必须在下一次 UC 公告形成后计算。真正终止时追加环境定义的终端奖励并将 bootstrap 置零；采样切片保留合法下一状态的价值。

候选与偏离评价必须完成有限回合，`evaluation_steps=None`，只有 `score_complete=True` 的结果可参与配对选优。当前能源实验使用 `gamma=0.99`，Ausgrid 按完整月份比较；短窗口与 Critic 尾值只用于诊断。连续 Runner 保留采样、追加及诊断功能，双层候选选择入口要求有限场景。

`inventory_valued_config()` 启用固定终端库存估值 `v_E=eta_d * max(0, 0.35-throughput_cost)`，默认 0.3135 AUD/存储 kWh，显式正配置值保留。真实终点奖励追加一次 `v_E*(E_T-E_0)`，采样批末不结算。评分包含全部实际奖励及终端项；现金、初末电量、库存价值、两路充电与实际放电另列，库存价值不混入现金。该估值允许经济上合理的净放电，并非强制首末 SOC 相等。

快照包含当前阶段、公告、SOC、消费者队列、月累计量、完整环境记录和随机状态。`get_state()` / `set_state()` 支持在已发布公告的阶段恢复。连续适配器在数据耗尽时返回明确标志，可调用 `append_profiles()` 追加连续时序；不会循环旧数据。当前命令行实验仍采用有限月回合，采样切片和外层训练轮次之间连续保留状态。Runner 的 `trajectory_periods` 随选定分支恢复，不受 RLlib 聚合计数广播影响；Algorithm 的 `sampled_periods` 统计所有实际 PPO 分支预算。

## 5. 内外层更新与候选评价

每轮先冻结 UC Actor，更新消费者 Actor；随后冻结消费者 Actor，更新 UC Actor 产生候选。两类阶段都使用新采集的全体轨迹更新 UC 和所有消费者的 Critic。Actor 冻结时其梯度为 `None`，参数、Adam 动量和步数保持不变；Actor 阶段所有 Value 也同样冻结。实现通过原生 Learner 的 `critic`／`actor` 更新阶段选择损失，不改变六维 UC 动作、观测、物理模型或各自奖励。

消费者前置适应推进主轨迹，结束后保存公共快照。UC 提案从该快照开始，多个提案沿用逐次 UC 更新；提案采样位置不直接提交。候选集合包含现有 UC。每个响应分支恢复公共环境、观测、随机状态，以及相同的消费者 Actor、Value 和两套 Adam，明确丢弃提案期间消费者 Value／Adam 的进展；保留对应候选 UC 的 Actor、Value 和 Adam。响应期间继续更新所有 Critic，各候选获得相同响应预算。候选接受使用同 seed 验证场景的配对 UC 回报差正态近似区间（默认 `z=1.96`）：下界超过最小改善量才进入可接受集合，再选择平均改善最大的候选。选定策略组合及其响应分支终点一起提交；未选分支只计入计算预算。

验证与偏离搜索均在退出时恢复主轨迹；冻结评价还核对全部 Learner 和 Adam 状态。算法保留当前学习契约核对，源码或数据协议变化后不自动迁移旧 checkpoint。

当前固定小规模入口为 `batch_critic_pilot.py`，从 seed 42 的混合策略初值配置 3 轮：4,608 个全部分支 PPO 时段、2,304 个主轨迹时段、零额外 MC 训练采样。同阶段两个 192 步单位连续采样后合成 384 步学习批，三轮共 12 批；Actor 2 epoch、全部 Critic 4 epoch、minibatch 32。每个 Critic 每批执行 48 次、3 轮共 576 次优化，参与阶段的 Actor 每批执行 24 次，分支回滚后保留的步数另计。候选评分和冻结表现均运行四个完整验证月，测试集继续保留；新结果目录为 `outputs/ausgrid-hybrid-storage/20261022-phase-batch-pilot/`。历史来源模型和早期基线摘要作为依赖保留，清单见[结果索引](RESULTS.md)，其动作、折扣和窗口协议不同。新入口不再要求与旧 Actor 初值、旧评分完全相同。

已完成的中长期验证使用不加 LayerNorm 的旧结构及相同候选选择规则，显式传入 `run_pilot(..., outer_rounds=72)`：576 批、36,864 个全部分支 PPO 时段、18,432 个主轨迹时段；每批更新设置保持不变。第 0、3、12、24、48、72 轮冻结评价，第 3 轮与已保存的小规模结果精确核对。完整月 Critic 轨迹同时累计供电和现金指标，整月现金按真实天数折算为 AUD/日，月初 48 小时窗口仍单列。验证不参与额外梯度更新，测试集不加载。每个验证节点保存 checkpoint，自动保留最近两个，最终另行保存完整状态和执行 Actor。详见[中长期实验](YEAR_STUDY.md)。

当前入口只依赖 `train.py` 的通用设置与序列化、`evaluation.py` 的冻结评价、`training_audit.py` 的采样／梯度／Adam 审计。上一轮报告及原始输出作为对照保留，其运行源码由提交 `ea3d322` 定位；3 轮小规模结果绑定 `ed56de0` 的运行来源。每次新运行在开始时记录代码哈希，不改写历史结果。

消费者偏离搜索固定 UC 和其他消费者的 Actor，仅更新被检查消费者的 Actor，同时全部 Critic 继续跟踪当前联合策略；结束后恢复完整参考快照。候选与偏离报告记录实际搜索预算和已发现的收益改善。微型合成测试用于验收训练、回退、恢复和评价链路；结果报告明确搜索预算与区间近似，不据此宣称均衡。

每轮在候选生成前及最终选定策略组上分别搜索消费者偏离。最终策略组用于联合停止检查：消费者已发现改进、UC 候选改进、Actor 参数 RMS 变化和配对区间半宽同时达到阈值，并连续维持指定轮数，入口才提前停止。默认阈值为 `consumer_gain=0.1`、`uc_gain=0.1`、`policy_change=0.001`、`confidence_half_width=0.1`，连续轮数为 3；两类改进和区间半宽以元等值计，参数 RMS 无量纲。通过 `StackelbergConfig.training(stopping_tolerances=..., stopping_patience=...)` 配置，`stopping_patience=0` 使用固定外层预算。未搜索偏离或缺少区间时，联合停止条件不会通过。

算法通过 `ScenarioFactory`、`TrainingEnvironment` 结构接口连接场景，只要求公告、物理推进、状态恢复和固定输入规格。`paired_gain`、`select_candidate`、`generalized_advantages` 与 `stopping_status` 是独立函数。专用算法代码不导入 `snmarket` 或 `experiments`；现有六算法和 `common` 无需加入能源场景分支。

主要预算参数如下：

| 参数 | 含义 |
|---|---|
| `--outer-rounds` | 上层候选选择轮数 |
| `--inner-updates` | 每轮初始消费者适应的采样单位数，合并为一批学习 |
| `--response-updates` | 每个候选响应的采样单位数，合并为一批学习 |
| `--candidates` | 新 UC 候选数量，现有策略另行参与比较 |
| `--proposal-updates` | 每个新候选的 UC 采样单位数，合并为一批学习 |
| `--deviation-updates` | 每个消费者偏离搜索的采样单位数，合并为一批学习 |
| `--rollout-steps` | 采样单位长度；实际学习批长 = 它 × 该阶段采样单位数 |
| `--epochs`、`--critic-epochs`、`--minibatch-size` | Actor / Critic 轮数与批内切分 |
| `--critic-lambda` | 独立 Critic 目标参数，默认 1.0 |

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
decision = env.publish(actors.uc_controls(observation, env.uc_bounds()))
actions = {
    key: actors.consumer_action(key, decision.consumer_observations[key], decision.consumer_bounds[key])
    for key in env.consumer_ids
}
transition = env.step(actions)
```

`ActorBundle` 不创建 Value、优化器或模拟器；UC 返回 `[mode, *raw_latents]`，显式输出层生成六维物理动作；消费者输出四维物理动作。旧六维高斯 checkpoint 不能直接恢复到新网络，旧实验由当时源码复现。

## 8. 三模式储能修改的核验（2026-10-21）

319 项相关检查通过：310 项单元检查、4 项原生两阶段环境检查、5 项本地 Ray CPU 训练集成检查。覆盖空电／满电和来源边界、采集与 PPO 联合概率一致、未激活坐标零梯度、候选分支恢复、checkpoint 精确重放及 Actor 导出。CUDA 本次未测。

固定电价算例验证了库存结算：只看第一个时段时，消耗初始电量带来 0.33 的奖励优势；真实终点扣减 0.33 库存价值后，完整回合与同价 PG 供电的奖励一致，终端项只记一次。

冻结初始化策略另运行一个完整 Ausgrid 验证月（seed 10042，1,488 个半小时）：闲置 487 次、充电 502 次、放电 499 次，DER 充电 90.7314 kWh、PG 充电 253.4044 kWh、实际放电 315.4900 kWh。逐期重算 logp 无差异，未发生空电放电或满电充电的非法模式，真实终止一次且 bootstrap 为零。该核验没有更新策略，也没有加载测试集，数值用于核对新接口和完整月结算。原始记录见 [full-month.json](../../outputs/snmarket-training/hybrid-storage-verification-20261021/full-month.json)。

新策略的训练曲线由 `plot_batch_losses.py` 读取实际日志生成；完整月和版本间对比使用 `plot_hybrid_storage.py`、`plot_storage_round_comparison.py`。旧三轮专用汇总工具已删除。

本次按用户要求完整撤销供电／现金可行性回调、轮初评价缓存和全拒绝恢复路径。新 UC 候选不满足原有配对回报规则时直接选择候选 0，前置消费者适应及选中响应分支接续。供电和现金仍报告，但不参与新增硬筛选；未实施此前讨论的首批 Critic 保留方案。


## 9. 延长学习范围（2026-10-22，已完成的前一版实验）

现有 γ 改为 0.99、Actor λ 改为 0.97、数据批长改为 192；不增加超参数、网络参数或奖励项。候选、偏离评价和完整月 MC 使用 Runner 的同一 γ，现金和物理量继续未折扣汇总。24 小时后 TD 残差对优势的权重约 0.1431，旧 10 轮为 0.00727；192 步批首的 Critic 尾值系数约 0.1452。完整月折扣评分偏重前几天，月末库存项的月初权重很小，因此全月现金、失供和初末 SOC 仍须并列审查。

前一版 10 轮已完成，目录为 `outputs/ausgrid-hybrid-storage/20261022-long-horizon-10rounds/`：80 批、15,360 个 PPO 分支时段、7,680 个主轨迹时段、每个 Critic 1,920 次实际更新。样本和优化步数为 `20261021-10rounds` 的三倍。两版结果及冻结源码保留原样。当前新的阶段合批协议见下节。

无额外奖励的 SOC 方法及文献依据见 v9 第 26.5 节：本次落实长轨迹和 GAE 信用分配，SOC 容量归一化、单调动作头作为后续候选，未接入生产网络。


## 10. 同阶段合批与更新后 GAE（2026-10-22，10 轮已完成）

保留 γ=0.99、Actor λ=0.97、三模式、六维物理动作、网络和奖励。同阶段固定联合行为策略，`collect(192 * 2)` 连续采集 384 步，使用 RLlib shuffle/minibatch 完成 Critic 拟合，再重算并归一化 GAE，最后更新选定 Actor。第 192 步没有额外截断或更新；真实月末分段，384 步批末使用合法下一状态 bootstrap。

Critic 拟合目标保留采样时的旧尾值，拟合期间固定。Actor GAE 使用拟合后的 Critic，但始终使用原动作、行为 logp、条件边界及保存的下一 UC／消费者价值输入。重算使用 `no_grad`，不采样动作、不推进环境。整批优势仅归一化一次，Actor 的两个 epoch 共用同一份优势。该改动减少每次仅追逐一个短批均值的问题，并让当前 Actor 用到本批拟合结果；对应 10 轮实测结果由 [RESULTS.md](RESULTS.md) 汇总。

| 每 10 轮预算 | 前一版 | 本次实现 |
|---|---:|---:|
| 学习批数 × 每批时段 | 80 × 192 | 40 × 384 |
| PPO 全部分支时段 | 15,360 | 15,360 |
| 主轨迹时段 | 7,680 | 7,680 |
| 每个 Critic 实际 Adam 步数 | 1,920 | 1,920 |
| UC Actor 提案实际 Adam 步数 | 240 | 240 |

每轮四批分别为消费者适应、UC 提案、旧 UC 响应、新 UC 响应。每批 12 个 minibatch，Critic 4 epoch、Actor 2 epoch，回滚后保留步数另列。合并前后在 minibatch 32 可整除当前批长的条件下优化步数一致；其他自定义预算按实际整批切分计算。

候选恢复规则不变：相同公共消费者网络及两套 Adam、环境和 RNG，配上各候选 UC 状态；只提交选中响应分支。学习契约的 `update_protocol` 升级，旧逐段学习 checkpoint 不直接续训。新的固定输出目录为 `20261022-phase-batch-10rounds`，程序在目录已存在时拒绝覆盖。该目录已完成 10 轮；第 8 轮中断及预算内恢复的额外实际采样 / 优化量保留在原始报告中。计划 72 轮的新运行只完成 1 轮后按用户要求停止，见[结果索引](RESULTS.md)。
