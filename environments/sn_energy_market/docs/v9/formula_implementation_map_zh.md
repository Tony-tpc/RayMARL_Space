# v9 环境公式、实现与验证对应表

**核查日期：2026-10-18。** 本表以[完整建模文档](modeling_complete_zh.md)中的当前编号为准，逐项列出环境运行链和实际测试入口。环境实现覆盖式（1）—（29），并已补齐可配置分时时窗、稳定任务 ID、决策输入记录与资源容量分解。首次迁移的确定性测试集为 177 项（包含参数化情形）；统一接口后的新增检查见[兼容性审查](compatibility_audit_zh.md)。式（30）是原论文公式的文献转写；第 12—14 章的学习更新、最佳响应和均衡检验属于后续 RL 工作。

## 1. 如何复核

生产实现位于 `environments/sn_energy_market/src/snmarket/`。以下表格中的 `模块.函数` 均指该目录。测试位于 `environments/sn_energy_market/tests/`，可以直接按表内测试名定位。全部物理动作由调用者提交；环境只校验和执行，策略可行映射作为单独的纯数值函数提供。

验证分为三类：

1. **确定性回归**：手算例、非法输入、时序、账本、观测和状态恢复，使用 pytest 执行。
2. **独立数值参考**：[v9_oracle.py](../../tests/v9_oracle.py)不导入生产计算模块，按文档重新计算市场、物理、队列、DR 和费用；[v9_checks.py](../../tests/v9_checks.py)负责读取输入并逐字段比较。
3. **规模验证**：[validate_v9.py](../../../../experiments/sn_energy_market/validate_v9.py)与[validation_helpers.py](../../../../experiments/sn_energy_market/validation_helpers.py)生成合法物理动作，分别运行静态参考对照、非法输入、连续回合和跨月场景。最终成功数量、残差和覆盖计数以相应运行目录的 `manifest.json`、`report.json` 为准；计划数量不视为已完成数量。

静态参考比较覆盖每层 DER 首轮、UC 比例分配、DER 补配、逐户服务、任务、DR、账单及 UC 奖励。动态批量验证逐期核对不变量和直接算术关系，**没有逐期调用独立出清参考**；不能将动态期数同时报告为独立 oracle 对照次数。

## 2. 式（1）—（10）：UC 与主体接口

| 公式 | 文档定义及实施要点 | 生产实现 | 已有确定性验证 |
|---|---|---|---|
| （1） | 六维 UC 物理动作，依次为 PG 额度、主动弃 DER、DER 充电、PG 充电、放电额度、DR 上限。没有总充电这一额外行动坐标。 | `contracts.UCAction`；`action.coerce_uc_action` | `test_six_actions_preserved_with_full_oracle`；`test_invalid_uc_never_changes_state_or_announces` 验证六维及错误维度。 |
| （2） | PG、DER 来源分流、总充电空间、SOC、放电能力、充放电互斥、事件 DR 边界。非法输入报错，原状态保留。 | `uc.validate_uc_action`；策略侧 `action.map_uc_controls` | `test_random_policy_output_is_feasible_but_never_applied_implicitly`；`test_output_map_covers_target_actions_and_zero_domains`；`test_rectangular_action_space_does_not_replace_joint_validation`。 |
| （3） | DER 公告容量等于出力减主动弃电与 DER 充电；UC 公告容量等于 PG 额度加放电额度减 PG 充电。 | `uc.prepare_offer`；`contracts.PublicOffer` | `test_six_actions_preserved_with_full_oracle`；`test_document_full_worked_example_end_to_end`。 |
| （4） | PG 实际进口成本、实际使用 DER 的运行成本、外部金额及 UC 现金利润。DER 充电也计入实际使用 DER。 | `settlement.settle` 的 `grid_cost`、`der_operating_cost`、`external_cost`、`cash_profit` | `test_document_full_worked_example_end_to_end` 复现现金 −0.025；`test_current_u01_reward_includes_adequacy_beyond_historical_cash` 区分高 PG 价下的现金与奖励；静态参考逐项比较现金。 |
| （5） | 储能交流侧总吞吐退化、实际弃电成本、经营收益。经营收益仅为派生诊断。 | `settlement.settle` 的 `ess_cost`、`curtailment_cost`、`book_profit` | `test_zero_sale_keeps_firm_charging_and_curtailment`；`test_cost_units_per_energy_per_time_and_per_assessment`；完整算例经营收益 −0.060。 |
| （6） | 冻结基线加能力内到期回补，扣公告总容量和正价事件内实际付费 DR 抵扣；另加逐次到期残留考核。到期项不再乘步长。 | `incentive.procure` 生成 `supply_credit_kw`；`settlement.settle` 生成参照、缺口、缺口成本和到期成本 | `test_u03_normal_supply_beats_unpaid_induced_cut_in_document_case`；`test_u03_credit_requires_event_positive_price_and_actual_dr`；`test_a01_purchase_credit_does_not_claim_incremental_dr`；`test_a02_extra_due_reservation_remains_visible`；步长测试。 |
| （7） | UC 净请求 DR 服务估值，基础经济组系数为零；正值作为可配置偏好。 | `settlement.settle` 的 `dr_value` | `test_positive_dr_value_can_reopen_low_coefficient_incentive`；完整算例估值 0.200。 |
| （8） | UC 奖励为现金减储能、弃电、服务成本，加 DR 估值；真实终止另追加终端项。 | `settlement.settle` 的 `uc_reward`；`reward.calculate_rewards` | `test_document_full_worked_example_end_to_end` 复现 −1.860；`test_u02_der_to_uc_substitution_is_an_economic_result`；U03 系列测试；静态参考完整奖励比较。 |
| （9） | Consumer 观测为 23 个标量加 H+1 个个人截止桶，共 H+24，字段顺序固定。 | `observation.build_consumer_observation`、`consumer_fields`、`queue_buckets` | `test_every_current_observation_coordinate_against_hand_vector` 逐坐标核对两户完整观测；另有顺序、维度、原生 episode 和批量截止桶检查。 |
| （10） | Consumer 四维物理动作：削减、带符号移峰、DER 比例、DER 买方报价；不再包含 UC 买方报价。 | `contracts.ConsumerAction`；`action.coerce_consumer_action` | `test_invalid_consumer_preserves_pending_announcement`；`test_higher_der_bid_paid_at_own_bid_and_uc_at_fixed_tariff`。 |

## 3. 式（11）—（20）：消费者负荷、DR 与费用

| 公式 | 文档定义及实施要点 | 生产实现 | 已有确定性验证 |
|---|---|---|---|
| （11） | 到期回补请求为个人到期能量除步长与个人柔性能力的较小值。UC 观测汇总逐户结果，不能先汇总后取最小值。 | `action.build_local_bounds`；`observation.build_uc_observation` | `test_due_local_capacity_and_terminal_no_deferral`；`test_observation_dimensions_order_aggregate_due_and_no_private_uc_fields` 的“到期任务与柔性能力位于不同住户”反例。 |
| （12） | 正移峰创建未来任务；负移峰提出回补请求。同一带符号变量表达两种方向，回补成交后才减少队列。 | `action.prepare_order`；`dr.realize_service` | `test_document_full_worked_example_end_to_end` 同时包含新移出和历史回补请求；`test_net_dr_cancels_repayment_request_even_without_service`。 |
| （13） | 先为到期回补预留能力，再限制削减。削减不能侵占刚性负荷或必要到期回补能力。 | `action.build_local_bounds`、`validate_consumer_action` | `test_due_local_capacity_and_terminal_no_deferral`；非法削减测试；参考 `test_local_due_bounds_do_not_allow_cut_to_remove_due_obligation`。 |
| （14） | 移峰负方向受剩余柔性和历史债务限制；正方向受负荷、柔性和队列空间限制；到期时必须提出能力内负移峰；有限期关闭无法在时域内到期的新任务。 | `contracts.LocalBounds.shift_interval`；`action.validate_consumer_action`、`map_consumer_controls` | 到期与终端边界测试；`test_explicit_horizon_delay_mode_combinations` 对有限 H=4、有限 H=96、连续 H=24 各运行 96 期并逐期检查允许新移出条件。 |
| （15） | DER 比例在 [0,1]；报价在 DER 底价与个体报价上界、UC 当期零售价的较小值之间。 | `action.build_local_bounds`、`validate_consumer_action` | `test_invalid_consumer_preserves_pending_announcement`；`test_heterogeneous_coefficients_and_queue_caps_reach_individual_ledgers`；输出层可行性测试。 |
| （16） | 原生需求、总请求、必要层请求分别派生；历史回补增加总请求，最低到期回补进入必要层。 | `action.prepare_order`；`contracts.MarketOrder` | `test_necessary_layer_beats_high_bid_early_repayment`；完整算例；静态参考逐户核对三种请求。 |
| （17） | 有符号 DR 取事件内基线减请求，全体求和后取正。回补抵扣使用请求，不使用实际完成量。 | `incentive.procure` | `test_net_dr_cancels_repayment_request_even_without_service`；`test_shift_then_full_repayment_preserves_total_energy` 检查先正移出、后负回补；事件门控测试。 |
| （18） | 总补偿由单价、步长、采购上限与净 DR 决定，再按正贡献份额分配；零正贡献时为零。 | `incentive.procure`；`contracts.DRResult`、`DRIncentive` | 完整算例的 2:1:0 份额；`test_a01_purchase_credit_does_not_claim_incremental_dr`；净 DR 相消测试；静态参考逐户金额对照。 |
| （19） | DER 按买方报价、UC 按冻结零售价计费；原生失供赔偿按真实失供；净账单减去 DR 与失供返款。 | `settlement.settle`；`tariff.uc_price` | `test_higher_der_bid_paid_at_own_bid_and_uc_at_fixed_tariff`；`test_der_rescue_uses_original_bid_and_only_remaining_capacity`；`test_month_transition_freezes_price_then_resets_next_month`；`test_configured_half_open_windows_and_monthly_tiers`；异质赔偿测试。 |
| （20） | 二次削减/正移出不适，原生失供成本，期末待办与逾期持有成本，以及 Consumer 总成本负值奖励。负移峰不单列二次不适。 | `settlement.settle` 的各户成本；`reward.calculate_rewards` | 完整三户奖励 −0.800000、−0.785417、−0.943750；`test_cost_units_per_energy_per_time_and_per_assessment`；异质系数测试；静态参考逐项比较。 |

## 4. 式（21）—（29）：市场、物理与跨期闭合

| 公式 | 文档定义及实施要点 | 生产实现 | 已有确定性验证 |
|---|---|---|---|
| （21） | 每户服务不超总请求；DER、UC 总成交不超各自公告容量。 | `market.clear`、`allocate_der`、`proportional`；`EnergyMarketEngine._validate_results` | `test_six_actions_preserved_with_full_oracle`；必要层优先测试；全部静态参考对照和动态 `check_record`。 |
| （22） | 总服务等于总请求与两池公告容量合计的较小值。两层各执行 DER 首轮、UC 分配、DER 补配。 | `market.clear` 计算成交；`EnergyMarketEngine._validate_results` 统一校验容量利用 | `test_der_rescue_uses_original_bid_and_only_remaining_capacity`；`test_equal_bids_proportional_order_independent`；`test_necessary_layer_beats_high_bid_early_repayment`；动态逐期总服务恒等式。 |
| （23） | UC 成交先在公告放电额度内执行 ESS，其余加固定 PG 充电形成实际 PG 进口。 | `physics.execute` | `test_six_actions_preserved_with_full_oracle` 中放电与两路充电场景；`test_zero_sale_keeps_firm_charging_and_curtailment`。 |
| （24） | 未售 DER、主动弃 DER 和总弃电分别记录；总弃电也等于 DER 出力减 DER 充电及 DER 成交。 | `physics.execute` 的 `der_unsold_kw`、`der_planned_curtailment_kw`、`der_spill_kw` | `test_zero_sale_keeps_firm_charging_and_curtailment`；`test_u02_der_to_uc_substitution_is_an_economic_result`；静态参考与动态分解恒等式。 |
| （25） | SOC 按充电效率乘输入、放电效率除输出更新，无自放电。数值边界归一化仅处理浮点残差。 | `physics.execute`；`EnergyMarketEngine._validate_state` | `test_exact_full_discharge_reaches_valid_zero_soc`；完整算例 SOC=5.475；`test_legacy_config_not_silently_reinterpreted` 拒绝旧自放电字段；动态逐期 SOC 对照。 |
| （26） | PG+DER+ESS 等于用户服务+两路充电+弃电。 | `physics.execute` 的 `power_residual_kw`；`EnergyMarketEngine._validate_results` | 完整算例、零成交固定充电、六维行动场景及动态逐期功率守恒。 |
| （27） | 原生优先，再用剩余服务完成历史任务；实际回补更新 FIFO，正移出追加任务；请求不能直接当完成量。 | `dr.realize_service`；`consumer.next_consumer_state` | `test_actual_repay_fifo_earliest_deadline_and_native_priority`；`test_shift_then_full_repayment_preserves_total_energy` 两期累计服务能量等于原基线能量且队列归零；`test_new_task_identity_survives_partial_repayment_and_replay`；微小真实任务保留测试。 |
| （28） | 期末到期残留为入期到期能量减实际回补电量的正部；队列保持容量上界；能力缺口单独诊断。 | `dr.realize_service`；`EnergyMarketEngine._validate_state` | `test_due_local_capacity_and_terminal_no_deferral`；`test_a02_extra_due_reservation_remains_visible`；动态到期超能力、容量与逾期残留检查。 |
| （29） | 真正有限期终止追加初末电池差值估值及双方剩余任务责任；连续数据耗尽和采样边界均不作真实终止结算。 | `settlement.settle`；`EnergyMarketEngine.evaluate_consumer_actions`；`SnEnergyMarketEnv.step` | `test_cost_units_per_energy_per_time_and_per_assessment`；`test_final_period_rejects_late_new_shift_and_settles_terminal_once`；连续数据追加测试；原生 episode 终止协议测试。 |

式（30）的二次 DR 价值来自原论文，不调用生产环境。当前奖励使用式（7）的线性净请求 DR 估值；不能将原论文式（30）加入账本后仍声称与当前 v9 相同。

## 5. 无编号规则与实现边界

| 规则 | 实现与实际证据 |
|---|---|
| UC 聚合观测 H+20 | `observation.build_uc_observation` 依次生成 13 个当期标量、H+1 个聚合桶、6 个历史值。`test_every_current_observation_coordinate_against_hand_vector` 核对固定双户例的完整 UC 向量；另有逐户限幅汇总、负现金及 info 隔离测试。`test_private_queue_redistribution_preserves_uc_aggregates_and_other_consumer_input` 检查交换两户私有任务后，UC 聚合观测和第三户个人观测保持相同。 |
| 神经网络一次输出后的可行映射 | `action.map_uc_controls` 实现第 13.2 节的固定数值映射。测试覆盖正负储能方向、零域、目标行动和 1,000 个内部控制采样。这里没有实现神经网络、随机策略概率或 PPO 更新。 |
| 时间、月累计和冻结零售价 | `tariff.uc_price` 与 `consumer.next_consumer_state`。分时时窗、三组费率及累计阈值均来自配置；窗口使用小时半开区间并校验不重叠。跨月测试先用入期单价结算，再记录关账量并将下一月累计清零。`test_calendar_audit.py` 与 `test_append_cannot_switch_local_calendar_offset` 拒绝构造、恢复或追加时改变本地价表日历。 |
| 原子执行和幂等 | `evaluate_consumer_actions` 构造候选，`commit` 提交一次。`test_two_stage_clock_atomic_candidate_and_idempotent_concurrent_commit`、`test_injected_failure_and_tampered_candidate_preserve_state` 检验并发重放、失败回滚和候选完整性。 |
| 快照与状态一致性 | `get_state/set_state` 保存阶段、公告、队列、电量、月账本、历史反馈和随机状态。测试覆盖公告前后恢复、负现金恢复、字段篡改、公开返回对象隔离；`test_restored_history_order_is_chronological` 检查字典插入顺序变化后按真实物理期恢复最后记录与终端公告。 |
| 决策与执行记录 | `ExecutionRecord` 保存入期 `decision_state`、UC 观测、逐户 Consumer 观测、局部行动边界，以及动作、公告、出清、物理执行、账本和奖励。`test_record_keeps_complete_decision_inputs_and_capacity_decomposition` 对照真正送出的输入并检查返回对象隔离。完整内部记录与 Actor 公开反馈分别提供。 |
| 任务身份 | `ShiftTask.task_id` 随部分回补保持；初始缺省 ID 在 reset 时确定生成，新移出生成独立 ID。`test_initial_task_ids_are_deterministic_and_do_not_reorder_tasks`、`test_new_task_identity_survives_partial_repayment_and_replay` 及重复、无效 ID、碰撞测试覆盖初始输入到恢复重放。 |
| 数据来源 | `Profiles.metadata` 保存来源、实现值/预测值标记、种子和数据划分等元数据，纳入数据指纹；追加数据保留各段来源。`test_profile_metadata_is_detached_bound_and_preserved_on_append` 验证隔离、指纹变化及三段追加记录。 |
| 功率、电量和系数的一致缩放 | `test_consistent_power_energy_and_unit_coefficient_rescaling` 将生产三户例的功率、电量、任务、容量、阈值和 epsilon 同乘 1,000，电量及持有成本系数和报价除以 1,000，已有金额保持不变，验证奖励、现金不变以及 SOC、供给缺口按比例变化。这是数学重标检查，公共 API 字段仍采用 kW/kWh。 |
| 连续外生数据 | `Profiles.append`、`EnergyMarketEngine.append_profiles` 要求显式连续数据，耗尽报错且不推进状态。不会循环使用旧数据或重置日历。 |
| 两阶段适配 | `SnEnergyMarketEnv` 实际接入原生 `MultiAgentEpisode`，运行 UC/Consumer 交替动作与真实终止；没有启动 Algorithm、Learner 或训练器。 |
| 第 12—14 章博弈与学习 | 长期策略目标、Nash 最佳响应、UC 重新适应、PPO/GAE、价值网络、模型参数共享和收敛检验均不属于本轮已实现运行链。环境可提供所需观测、奖励和执行记录；这不等于学习或均衡已经实现。 |

## 6. 压力生成器覆盖及其统计解释

当前生成器独立于生产输出层和消费者边界函数，直接按物理约束构造动作。消费者规模按静态/普通回合序号分配为 N=1、2、8、32、128，计划占比分别为 20%、20%、40%、18%、2%；长期组使用 N=2、8、32、128，跨月组使用 N=2、8、32。

资源和参数覆盖：零 DER、零基线、零柔性、空电池、满电池、部分 SOC、充电/放电/空闲、DR 关闭/全天/部分时段、零 DR 价格、个体异质系数与队列上限、初始逾期和未来任务、精确同报价、0/1 边界与内部随机比例。分时时窗包括默认表、全平时段、多窗口及非整数小时边界；任务包括显式初始 ID、自动初始 ID、回补保留 ID 与新移出 ID。PG 单价取 0.05—1.70 元/kWh；充放电效率取 0.75—1.00；UC 公告考核系数取 0—3。随机动作不根据利润寻找最优行为。

需要按真实生成方式解释以下覆盖：

- **组合并非全因子。** `index % 4` 同时选择步长 5/15/30/60 分钟、H=1/4/24/96、初始 SOC 分组；普通动态回合的奇偶又决定有限/连续。因此 H=4、96 的普通动态回合落在连续模式，H=1、24 落在有限模式。规模计数不能证明全部 H×步长×模式×SOC 组合已遍历。
- **有限期 H≥T 的确切语义。** 当 H=T=96 时，所有 t≥0 都有 t+H>T−1，因此整个有限回合不能新移出任务。这是文档行动域的结果。定向扩展测试现已分别运行有限 H=4、有限 H=T=96、连续 H=24 各 96 期，逐期核对边界；这些轨迹使用零移峰行动验证时域边界，实际正移出与回补闭环另由两期能量测试覆盖。随机连续 H=96 的计数仍不能替代有限 H=T 的证据。
- **“full_service”计数的名称比判据更宽。** 当前批量计数使用原生失供为零，仍可能存在未完成历史回补。总请求是否全部满足应结合 `Allocation.unfilled_kw` 或请求与实际服务差额判断。
- **“negative_signed_DR”计数使用全体有符号 DR 和为负。** 它确认净增载时段，不等同于所有发生任一个体负 DR 的时段数量。
- **FIFO 数值对照。** 静态桥接按创建期、截止期逐来源核对能量，容许已完成减法留下的机器精度尾差。动态辅助检查忽略低于 1e−10 kWh 的队列条目结构，但总队列恒等式仍逐期核对，并另外检查 ID 非空、唯一及回补身份保持。真正 1e−13 kWh 未服务与半服务任务由专门测试要求保留。
- **观察向量验证强度。** 固定双户例已手算核对 UC 与两户 Consumer 的全部坐标；批量检查覆盖维度、有限值、时间、基线、截止桶和聚合到期功率。全部随机观测逐坐标独立复算仍可作为后续扩展。

## 7. 经济与单位验证中仍需区分的事项

| 问题 | 已有证据 | 本表审查时尚未建立的证据 |
|---|---|---|
| U01：高采购价下的服务选择 | 固定 1 kWh 请求、PG 单价 1.70 时，供电现金与奖励均为 −1.30，限供现金 −1.00、含考核奖励 −3.00。`test_u01_fixed_request_service_threshold` 进一步检查 PG 价 0.65、1.70、3.40、3.50，供电与限供奖励差精确为 3.40−PG 价，覆盖优势、临界相等和反转。 | 当前系数下的单期固定请求阈值已有验证；其他赔偿、考核、零售价、储能机会成本及跨期策略的最优边界属于参数和策略研究。 |
| U02：来源替代 | 明确回归 DER 直接售电收益 0.30、弃 DER 后以 PG/UC 电价供电收益 0.60。 | 未证明所有费率、未来 SOC 与长期策略下都存在同一优势；本模型保留该经济选择。 |
| U03：公告供给责任 | 原算例、正价事件抵扣门控、低考核加正 DR 估值反例、A01 既有减载付费认领、A02 额外到期预留均有确定性回归。 | 不包含消费者连续最佳响应搜索、UC 策略重新适应或长期最优限供证明。 |
| A04：真实供给不足与自主保留 | `test_a04_physical_shortage_and_voluntary_withholding_are_distinguishable` 对照物理供给上限 0 与 4 kW、公告同为零的两个场景；相同公告考核分别对应真实不足与自主保留。记录中的最大物理供给及容量减项另有恒等式验证。 | 容量分解说明供给从何处减少；各减项不能直接解释为奖励损失的独立边际贡献，后者还受缺口正部和 DR 抵扣影响。 |
| U04：基线、反弹与终端价值 | FIFO、连续状态、终端和月账本已验证；`test_shift_then_full_repayment_preserves_total_energy` 的两期移峰/回补例确认净请求 DR 先正后负、总服务电量仍为原基线的 2 kWh、最终待办归零。 | 已有能量回补闭环，不包含真实基线识别、正常供给配对福利或全周期节能的因果判断。随机终端系数取样只验证计算，不验证参数标定。 |
| 时间与功率单位 | 5/15/30/60 分钟生产测试区分按电量、按时间和逐次考核；新增生产三户例完成 kW/kWh→W/Wh 的一致重标，对报价、系数、状态、容量、阈值及 epsilon 同步换算后奖励和现金不变。 | 重标例为单期且没有到期残留和真实终端值发生；逾期、终端两类项已有分别的步长测试，尚未在同一个非零逾期/终端场景中重复 W/Wh 对照。 |
| 日历 | 具有 UTC 偏移的输入、连续时序、月切换、本地价表日历一致性已测试。 | 当前日历为固定偏移时间。真实含夏令时地区的时区规则或缺失/重复本地小时处理不在此数据格式内。 |
| 可行映射与策略 | 纯数值映射覆盖合法边界与采样可行性，环境拒绝非法联合动作。 | 尚未验证网络概率、梯度、分布参数、PPO 概率比及零宽域训练行为；这些属于后续 RL 实现。 |

定向测试覆盖现行 U01 服务阈值、A04 物理与公告供给区别、W/Wh 整体缩放、有限 H=T 轨迹和两期移峰能量闭环。真实数据校准、福利因果识别和均衡研究需使用相应的实验方法。

## 8. 无编号工程契约的核查结果

本轮审查发现的四项工程接口缺口已补齐。以下对应文档中的具体要求，并列明验证入口：

| 文档要求 | 当前实现 | 验证入口 |
|---|---|---|
| 第 10.7 节：TUTT 分时表与阈值由配置提供 | `MarketConfig.uc_high_intervals_hours`、`uc_peak_intervals_hours` 给出高、峰时段，其余为平时段；费率和月累计阈值也由配置提供。静态参考和批量辅助函数各自读取配置计算零售价。 | `test_default_tariff_calendar_is_unchanged`；`test_configured_half_open_windows_and_monthly_tiers`；`test_invalid_and_overlapping_tariff_windows_are_rejected`；`test_adjacent_tariff_windows_and_nested_serialization`。 |
| 第 2.3 节：FIFO 任务包含任务 ID | `ShiftTask.task_id` 在 reset、追加、部分回补、快照和重放中保持稳定；输入缺省值会确定生成，重复与无效 ID 拒绝，外部初始 ID 与新任务冲突时生成新的唯一 ID。 | `test_initial_task_ids_are_deterministic_and_do_not_reorder_tasks`；`test_duplicate_initial_ids_and_missing_snapshot_id_are_rejected`；`test_new_task_identity_survives_partial_repayment_and_replay`；`test_external_initial_identifier_cannot_collide_with_a_new_task`。 |
| 第 5.3、20.2 节：区分真实物理不足与 UC 自主保留等缺口来源 | `ExecutionRecord.supply_diagnostics` 保存最大物理供给、未公告 PG 能力、未公告放电能力、两路充电分流与主动弃电。最大物理供给减四类容量减项等于公告总容量。结合冻结参照与 DR 抵扣可进一步计算不可避免缺口及自主决策增加的缺口。 | `test_record_keeps_complete_decision_inputs_and_capacity_decomposition`；`test_a04_physical_shortage_and_voluntary_withholding_are_distinguishable`。 |
| 第 19—20 节：决策信息和执行结果可追溯 | 每条执行记录包含当期完整状态、实际 UC/Consumer 观测、局部边界和执行结果。Profiles 保留可选来源元数据，指纹绑定数值与元数据，连续追加保存分段来源。 | `test_record_keeps_complete_decision_inputs_and_capacity_decomposition`；`test_profile_metadata_is_detached_bound_and_preserved_on_append`；`test_append_cannot_switch_local_calendar_offset`。 |

对当前文档的环境运行要求逐项核查后，未发现仍待补齐的明确计算接口。第 6 节列出规模验证的取样边界，第 7 节列出经济条件与研究证据范围；第 12—14 章的概率、梯度、策略学习和均衡检验按用户要求留待后续实施。
