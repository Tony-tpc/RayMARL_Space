# Stackelberg–Nash 双资源 PAB 与需求响应：完整建模整合审查稿

版本：`v7-integrated-review-20261010`。性质：**完整归集的候选模型，含明确未闭合项；尚非可直接训练的修复定稿。**

本稿以用户指定的无括号 v7 文件为交易规则主线，合并 v5 中仍适用的状态、跨月、终端、原子提交、CTDE、经验采样、实验记录与目录约定；与 v7 冲突的 v5 规则不同时保留。它不是把两个版本全文首尾拼接，也不是把前次聊天的“取消 PAB”建议写入 v7。

- 原始来源快照：[指定 v7 原文](source_v7_supplied.md)。原始文件及下载目录均未修改。
- 对照来源：[v5 建模文档](../../../../Stackelberg_Nash_Pay_as_Bid_Modeling_ZH_v5.md)。
- 本轮结论：[整体审查与逐项证据](modeling_audit_zh.md)。
- 可复查证据：[独立核验脚本](verify_v7_mechanisms.py)、[核验结果与来源 SHA256](verification_results.json)。

**来源权限说明**：源文件中“此前批准”“应用户要求”等说法只是该文件的叙述。本次用户授权的是合并、审查；不据此认定所有候选机制已获批准实施。未获得独立 v6 文档，故不假定 v6 的未展示内容；也未使用下载目录的同名括号版本。

## 阅读索引与合并准则

| 内容 | 位置 | 合并结果 |
|---|---|---|
| 主体、动作、订单、双池、备用、服务、观测、DR、收益 | 第 0—12 节 | 以指定 v7 为主，保留问题所在公式以便审查 |
| 博弈、训练、软件接口、验证与文献 | 第 13—16 节 | 保留 v7，补充明确的算法边界和现有仓库职责 |
| 全量状态、日历、FIFO、边界与异常 | 第 17 节 | 合入 v5 仍有效的状态语义，并列出 v7 空动作域问题 |
| 资金、费用、真实现金与研究效用 | 第 18 节 | 完整定义口径，纠正非现金成本混入 cash 的命名 |
| 合法分布、轨迹、两阶段 RLlib、版本接口 | 第 19 节 | 合入 v5 工程约束，更新为 v7 动作与观测 |
| 数据及示例参数 | 第 20 节 | 仅为合成核验；不冒充论文标定 |
| 执行与验证流程、实现差异 | 第 21 节 | 明确“文档整合完成”与“环境尚未升级” |
| 决策清单与正式结论 | 第 22 节 | 未闭合项不被文字表述掩盖 |

本稿仅直接纠正可核实的引用、交叉引用、账本名称及提交顺序，其他经济规则的修改建议放在审查报告中，不静默替换。

| 编辑编号 | 明确修订 | 理由 |
|---|---|---|
| E01 | 式（37）改为原文的 $T^s Q^2/p^m-T^rQ$ | 原 PDF 式（10）的分母没有平方；v7 自己的式（38）保持不变 |
| E02 | 原 v7 式（41）的 $\Pi^{cash}$ 改记 $\Pi^{book}$，新增真正 $\Pi^{cash}$ 的拆分规则 | 其中包含弃电机会成本和可能的电池折旧，不能直接称实际现金 |
| E03 | 原伪代码中的 reward、终端成本和校验移到 commit 之前 | 延续 v5 的完整转移原子提交，不允许计算奖励失败后已改状态 |
| E04 | 2.2 的 SOC 引用由式（29）改到式（26） | 原式（29）是服务缺口恒等式 |
| E05 | 目录映射为 `environments/`、`RLAlgo/`、`experiments/` | 遵守已确定的 RayMARL 主目录和算法归属 |
| E06 | 增加显式状态、终端、成本与配置补充约定 | 作为整合稿约定注明，避免“沿用旧版”造成实现歧义 |

**当前结论**：固定报价、固定服务量下的普通转备用套利已修复；改变报价后的备用依赖仍存在；本人队列可见性已改善；DR 系统价值与真实净减载仍可能脱节；连续缺供可能把系统推进到无合法动作状态。详细证据见审查报告。第 0 节的“设计”表不能被视作各问题已全部解决的证明。

---


**主线来源：指定 v7.0 研究设计稿；以下公式编号延续原文，正文已作 E01—E06 明示整合。**

**与 v6 的关键区别**：消费者重新同时参与 DER、UC 的 Pay-as-Bid 竞价；UC 普通售电和 PG 备用补供采用**同一消费者本轮 UC 报价**，而不是按 PG 成本对备用另行定价；备用有限且必要用电优先；扩展消费者观测，直接在本地可行域采样；UC Reward 加回 Nie 等（2024）式 (10) 的“DR 响应平方收益减采购支出”结构，并修正因本研究重新定义 $p_t^m$ 带来的分母退化。

> **来源界限**：基本 UC–Consumers Stackelberg + Consumers–Consumers Nash、PG/DER/ESS、TUTT、DR 增益项的思想来自 Nie 等，IEEE TSG 2024 的 §II、式 (1)–(31)；双 PAB、两级优先排序、备用同价、五维消费者动作、FIFO 跨期队列、成本与奖励拆分、固定 DR 参考基准、事件内正响应核验均是**本研究的拓展建模**。它们并非原文的原封实现。

## 0. 修订范围与执行承诺

| 原讨论项目 | v7 正式设计 | 不再增加的步骤 |
|---|---|---|
| 问题 1：UC 将亏损普通成交转为备用 | 同一个消费者、同一个 UC 报价对普通/备用统一结算；UC 承担**真实备用额外成本**和**合同服务责任** | 不二次询价，不令 UC 事后改变报价 |
| 问题 2：消费者追求低价备用 | 备用不再另有低价；备用容量有上限；先保障必要负荷/到期任务，再分配其他需求 | 不对消费者预先分配 $F_{i,t}$ |
| 问题 3：本地任务与期限隐藏 | 基础 10 字段后加入个人 FIFO 到期桶、已公布备用合同额度；Actor 直接生成本地合法 DR 动作 | 不用事后 `clip` 去悄悄修改本地 DR 决策 |
| 问题 4：出清正确不等于 SNE | 保留独立消费者最佳响应和 UC 重新响应检验；不把单元测试或 PPO KL 小称为均衡证明 | 不修改物理时段操作流程 |
| 问题 5：UC 不积极采购 DR | 借鉴原文式 (10) 的平方 DR 增益结构；保留补偿支出；原式及适配式并列，独立展示资金口径 | 不虚构外部 PG 向 UC 付款 |

**单时段执行顺序**：

```text
加载外生状态和本人待办 → UC Actor 一次公布供给和 DR 采购计划
→ Consumer Actors 各一次：可行域内决定削减/移峰/资源比例/两类报价
→ 在一笔条件式订单中明确 UC/DER 需求 → DER PAB 出清 → UC PAB 出清
→ 计算基本 PG/ESS 执行 → 有限 PG 备用按必要性优先补供
→ 服务与真实回补入账 → DR 核验和补偿 → 支付与收益 → 单次状态提交
```

### 0.1 研究问题和研究边界

系统有一个掌控 PG 接口、共享 DER 与共享 ESS 的 UC，以及 $N$ 个具有抽象 DR 能力的消费者。UC 先形成供给与激励采购公告；消费者知道公告后各自行动；市场在**一次**联合出清程序中求出成交结果。消费者以自身电费、调整不适和服务损失最小化为目标；UC 以售电净收益、服务责任以及 DR 系统价值组成的目标最大化为目标。消费侧没有独立 DER/ESS，暂不包含配电网多节点潮流，仅考虑**单个聚合电力接口**。

用 $t\in\{0,\ldots,T-1\}$ 记物理时段，$\Delta t=0.25\;\mathrm h$（96 时段/天）；$k$ 为训练外层轮次。功率统一为 kW，电量为 kWh，能源价格为元/kWh，货币为元。

---

## 1. 符号、主体和决定权

| 变量 | 经济或物理含义 | 谁决定/计算 |
|---|---|---|
| $p_{i,t}^{*}$ | 消费者原始基线需求，kW | 外生数据，在本轮前冻结 |
| $\Delta p_{i,t}^{l}$ | 主动削减功率，kW | Consumer Actor |
| $\Delta p_{i,t}^{s}$ | 带符号移峰功率：正为移出、负为回补，kW | Consumer Actor |
| $\rho_{i,t}^{r}$ | 本轮优先投给 DER 的总需求比例 | Consumer Actor |
| $b_{i,t}^{u},b_{i,t}^{r}$ | 对 UC、DER 的单段买方报价，元/kWh | Consumer Actor |
| $d_{i,t}$ | DR 后总申报负荷，kW | 由 Actor 动作唯一确定 |
| $q_{i,t}^{u,0},q_{i,t}^{r,0}$ | 初始 UC、DER 意向需求，kW | 订单转换 |
| $q_{i,t}^{u,cond}$ | DER 实际成交后仍需进入 UC 的条件需求，kW | 市场机械计算 |
| $p_{i,t}^{r},p_{i,t}^{u}$ | DER、普通 UC PAB 实际成交，kW | 市场 |
| $p_{i,t}^{back}$ | 外部 PG 备用补供量，kW | 备用程序 |
| $\ell_{i,t}^{native}$ | 本轮原生需求的未服务功率，kW | 服务模块 |
| $r_{i,t}^{rep}$ | 本轮按 Actor 要求计划回补历史任务，kW | Consumer Actor，由 $\Delta p^s$ 定义 |
| $r_{i,t}^{served}$ | 真正得到电力服务、已完成的历史回补功率，kW | 服务账本 |
| $B_{i,t},\bar B_i$ | 历史未完成任务、其最大容许余额，kWh | 本地 FIFO 队列、静态约束 |
| $p_t^m$ | **可补偿 DR 采购上限**，kW，非原论文分母中的 DR 目标同一语义 | UC Actor |
| $q_{i,t}^{ver},q_{i,t}^{paid}$ | 核验的主动净响应、实际付费响应，kW | 激励模块 |
| $P^{DR,ref}$ | DR 平方收益归一化的固定外生基准，kW | 实验配置，**不属于 UC 动作** |

**符号区分**：$p_t^m$ 是 UC 愿意补偿的采购上限；$P^{DR,ref}>0$ 是固定尺度。$B_{i,t}$ 是个人待办电量；$B_t^{back}=\sum_i p_{i,t}^{back}$ 是全体备用功率，这两个 $B$ 的索引及单位不同。

---

## 2. UC 上层动作、资源与供电公告

### 2.1 UC Actor

独立动作写为：

$$
\boxed{a_t^u=\left(p_t^{b,plan},p_t^{a,plan},p_t^{d,plan},p_t^{e,plan},p_t^{o,plan},p_t^m\right).}\tag{1}
$$

$p_t^{b,plan}$：PG 基础进口计划；$p_t^{a,plan}$：DER 预留弃电；$p_t^{d,plan}$：DER→ESS 充电；$p_t^{e,plan}$：PG→ESS 充电；$p_t^{o,plan}$：ESS 放电；$p_t^m$：允许补偿的 DR 采购功率上限。

ESS 总充电量是派生量：

$$
\boxed{p_t^{c,plan}=p_t^{d,plan}+p_t^{e,plan}.}\tag{2}
$$

### 2.2 物理可行域

$$
\begin{aligned}
&0\le p_t^{b,plan}\le\bar p_t^{b,phys},\quad 0\le p_t^{a,plan},p_t^{d,plan},\\
&p_t^{a,plan}+p_t^{d,plan}\le p_t^g,\qquad 0\le p_t^{e,plan}\le p_t^{b,plan},\\
&0\le p_t^{c,plan}\le\bar p^c,\qquad 0\le p_t^{o,plan}\le\bar p^o,\\
&p_t^{c,plan}\,p_t^{o,plan}=0,\qquad 0\le p_t^m\le\chi_t\bar P^{DR,proc}.
\end{aligned}\tag{3}
$$

$\chi_t\in\{0,1\}$ 为事前公布的 DR 事件标记。ESS 放电/充电还须满足期末能量界，见式 (26)。UC 的可行计划应由 Actor 的有约束分布/可行参数化采样；若设备间存在联合非凸互斥，可用“充电／放电／空闲”模式加各模式的合法连续参数。

### 2.3 对 Consumer 公布的普通交易容量

$$
\boxed{
P_t^{u,avail}=p_t^{b,plan}+p_t^{o,plan}-p_t^{e,plan},\quad
P_t^{r,avail}=p_t^g-p_t^{a,plan}-p_t^{d,plan}.
}\tag{4}
$$

公告包含 $P_t^{u,avail},P_t^{r,avail},p_t^m,T_{i,t}^u,T_t^d,T_t^r$、报价合法上下限以及事前确定的备用合同上限 $\bar P_t^{back,contract}$；所有公告先于本轮消费者动作产生。外部 PG 物理上限为 $\bar p_t^{b,phys}$。$T_{i,t}^u=T^u(t,E_{i,t}^u)$ 保留原论文 TUTT **作为合法报价底价或参考价**，不是本研究双 PAB 的最终成交价。

---

## 3. Consumer 下层动作：五维、明确申报 UC 需求

### 3.1 Actor Action

$$
\boxed{a_{i,t}^c=\left(\Delta p_{i,t}^{l},\Delta p_{i,t}^{s},\rho_{i,t}^{r},b_{i,t}^{u},b_{i,t}^{r}\right).}\tag{5}
$$

$\rho_{i,t}^r\in[0,1]$ 表示想优先参加 DER 池的净需求比例；$b^u,b^r$ 为两类资源分别出价。消费者无需先决定两类**实际成交量**，但要显式记录自己希望购买的初始数量。

### 3.2 **直接在本地可行域采样**，而非截断动作

冻结本期负荷 $p_{i,t}^*$、最大本地 DR 功率 $\Delta p_{i,t}^{max}$、历史待办 $B_{i,t}$、任务容量 $\bar B_i$。本版进一步约定 **削减、移出和回补共享单一时段柔性功率预算**，防止同时削减和回补时超过设备调节能力。先从到期桶计算 $r_{i,t}^{min}=E_{i,t}^{due}/\Delta t$，只有 $r_{i,t}^{min}\le\min(\Delta p_{i,t}^{max},B_{i,t}/\Delta t)$ 的场景才具有本地可行解。削减动作的合法域为：

$$
\boxed{0\le\Delta p_{i,t}^{l}\le c_{i,t}^{max}=\min\{p_{i,t}^*,\Delta p_{i,t}^{max}-r_{i,t}^{min}\}.}\tag{6}
$$

采样完 $\Delta p^l$ 后，令：

$$
\begin{aligned}
s_{i,t}^{max}&=\min\left\{p_{i,t}^*-\Delta p_{i,t}^{l},\ \Delta p_{i,t}^{max}-\Delta p_{i,t}^{l},\ \frac{\bar B_i-B_{i,t}}{\Delta t}\right\},\\
r_{i,t}^{max}&=\min\left\{\Delta p_{i,t}^{max}-\Delta p_{i,t}^{l},\frac{B_{i,t}}{\Delta t}\right\}.
\end{aligned}\tag{7}
$$

上式中的 $r_{i,t}^{max}$ 已扣除了本轮选择的削减预算，保证削减与回补同时存在时仍有 $c_{i,t}+r_{i,t}^{rep}\le\Delta p_{i,t}^{max}$。设当前必须回补的已到期电量为 $E_{i,t}^{due}$（kWh）；在可满足到期服务的标准场景，最少回补功率为：

$$
\boxed{r_{i,t}^{min}=E_{i,t}^{due}/\Delta t.}\tag{8}
$$

当 $r_{i,t}^{min}=0$ 时，$\Delta p_{i,t}^{s}\in[-r_{i,t}^{max},s_{i,t}^{max}]$；若 $r_{i,t}^{min}>0$，则：

$$
\boxed{\Delta p_{i,t}^{s}\in[-r_{i,t}^{max},-r_{i,t}^{min}].}\tag{9}
$$

必须先满足 $r_{i,t}^{min}\le r_{i,t}^{max}$；式 (6) 的预算预留保证了有期限任务在本地动作空间内不会被同轮削减挤占。如果任务已逾期且在本地功率极限内无法一次回补，记为**任务约束不可行场景**，由场景生成或合同违约规则处理，而不是把一个无解区间喂给 Actor。若当前有硬性原生不可削减负荷，则式 (6) 的 $c^{max}$ 还应扣除该负荷；参数需由数据集提供。$\rho,b^u,b^r$ 同样分别在公开合法区间采样，例如：

$$
\boxed{0\le\rho_{i,t}^{r}\le1,\quad
T_{i,t}^{u}\le b_{i,t}^{u}\le\bar b^u,\quad
T_t^d\le b_{i,t}^{r}\le\bar b^r.}\tag{10}
$$

动作策略使用**状态相关的分布或带上下界的可逆变换**，训练中按真实采样分布计算 PPO `log_prob` 和熵。边界没有宽度时应采用确定性退化维度，不对零宽区间套用连续分布。

### 3.3 Actor 决定需求，不再由 `dr.py` 修改 DR 决策

定义可行的主动削减、新增移出和本期计划回补：

$$
\boxed{c_{i,t}=\Delta p_{i,t}^{l},\quad
s_{i,t}^{+}=[\Delta p_{i,t}^{s}]_+,\quad
r_{i,t}^{rep}=[-\Delta p_{i,t}^{s}]_+,\quad [x]_+=\max(x,0).}\tag{11}
$$

$$
\boxed{n_{i,t}=p_{i,t}^*-c_{i,t}-s_{i,t}^{+},\quad
d_{i,t}=n_{i,t}+r_{i,t}^{rep}=p_{i,t}^*-\Delta p_{i,t}^{l}-\Delta p_{i,t}^{s}.}\tag{12}
$$

$n$ 是本轮仍需服务的原生负荷，$d$ 是附加历史回补后的总购电需求。$d$ 可大于 $p^*$。**本地 Action 合法不等于联合市场一定能够成交**；系统级资源不足仍可能导致服务不足，这属于物理供给结果，不是本地 DR Action 越界。

### 3.4 两类初始电力申报 + 单次条件式 UC 需求

$$
\boxed{q_{i,t}^{r,0}=\rho_{i,t}^{r}d_{i,t},\qquad
q_{i,t}^{u,0}=(1-\rho_{i,t}^{r})d_{i,t}.}\tag{13}
$$

$0\le q^{r,0},q^{u,0}\le d$，二者之和为 $d$。一笔订单存储

`(consumer_id, demand_kw, der_initial_kw, uc_initial_kw, uc_bid, der_bid, essential_kw, voluntary_kw)`。

**关键约定**：订单在首次提交时就声明“DER 未获得的全部需求，自动以本轮已经提交的 $b_i^u$ 进入 UC 清算”。因此 DER 清算后：

$$
\boxed{q_{i,t}^{u,cond}=d_{i,t}-p_{i,t}^{r}\ge q_{i,t}^{u,0}.}\tag{14}
$$

$\sum_iq_{i,t}^{u,cond}$ 就是 UC **收到的真实条件式需求总功率**。这不是第二次 Consumer Action，也不改变 UC 报价。消费者对 DER 的初始请求上限是 $q_i^{r,0}$，没有用上的份额进入 UC。

---

## 4. 必要负荷、到期任务与优先权：不引入个体 $F_i$

### 4.1 两种不同意义的“优先”

这里的优先指**当资源不足时先服务什么需求**，而不是在交易前给消费者分配独占保底配额。设可在本轮截止的回补任务功率 $r_{i,t}^{due}\le r_{i,t}^{rep}$；本轮剩余当前原生负荷必须被计入必要服务。

$$
\boxed{m_{i,t}=n_{i,t}+r_{i,t}^{due},\qquad
v_{i,t}=r_{i,t}^{rep}-r_{i,t}^{due},\quad d_{i,t}=m_{i,t}+v_{i,t}.}\tag{15}
$$

$m_{i,t}$：**必要服务**＝消费者已选择合法 DR 后的原生负荷 + 本轮到期回补；$v_{i,t}$：可以继续排队的未到期计划回补。原生负荷中若有允许失供的低优先级用电，可进一步细分；初版只保留这两级。

### 4.2 单次双池 PAB 的确定性优先排序

分配函数 `clear_pool(orders, capacity)` 对所有订单按**服务优先级、买方报价**排序：

1. 必要服务订单 $m$ 优先于可继续延期的任务 $v$；
2. 同一优先级内，报价由高到低；同价格组按剩余请求量比例分配；
3. 严格不得超过该供电池的容量；允许边际部分成交。

具体两池顺序：

**DER**：消费者最多提交 $q_i^{r,0}$；先用该额度覆盖必要服务份额 $\min(q_i^{r,0},m_i)$，再覆盖剩余可选份额。结果为 $p_i^r=p_i^{r,ess}+p_i^{r,opt}$。

**UC**：DER 后的有效剩余订单为 $q_i^{u,cond}=d_i-p_i^r$；剩余必要需求为 $m_i-p_i^{r,ess}$，剩余可延期需求为 $v_i-p_i^{r,opt}$。同样先必要再可延期，同级按 $b_i^u$ 排序，UC 普通成交为 $p_i^u=p_i^{u,ess}+p_i^{u,opt}$。

即：

$$
\boxed{0\le p_{i,t}^{r}\le q_{i,t}^{r,0},\quad
0\le p_{i,t}^{u}\le d_{i,t}-p_{i,t}^{r},\quad
\sum_i p_{i,t}^{r}\le P_t^{r,avail},\quad
\sum_i p_{i,t}^{u}\le P_t^{u,avail}.}\tag{16}
$$

同一价格组按比例分配的基本形式为：

$$
\boxed{p_i=q_i\min\left(1,\frac{R}{\sum_{j\in G}q_j}\right),\qquad\sum_{j\in G}q_j>0.}\tag{17}
$$

该规则把优先级作为公告的合同规则，买方报价仍在相同服务等级内改变成交机会，故依然是**带服务优先级的 Pay-as-Bid 竞买机制**，不是单纯价格排序的无优先级 PAB。须在论文中明确这一修改并做优先级消融。

### 4.3 无 $F_i$ 的系统供给边界

对所有独立可行的 Consumer 动作，不可能在物理供给不足时保证全部 $d_i$ 成交。令最终总实际可供电能受供给计划、PG 物理进口及备用合同限制；若对某个联合动作 $\sum_i d_i$ 超过该能力，则必有缺供。即使 $\sum_i m_i$ 没超过总资源，**买方自己给 DER 设置的请求上限**或电力来源限制，也可能阻止总资源全部投入必要服务。此种情形允许报告跨池未利用资源和必要服务缺口，不作虚假保证。

*不分配个体 $F_i$*。标准实验应把“必要任务物理可完成”设为场景准入条件，并检查按既定双池订单规则是否实际完成；另单独报告资源紧缺场景的真实缺供和延期。

---

## 5. UC/DER 出清、结算价与 PG 备用

### 5.1 UC 双 PAB 付款规则

DER 付款按获分配电量乘自己的 DER 报价；UC **普通和备用都按本轮同一个 UC 报价**：

$$
\boxed{
C_{i,t}^{energy}=
\left[b_{i,t}^{r}p_{i,t}^{r}+b_{i,t}^{u}\left(p_{i,t}^{u}+p_{i,t}^{back}\right)\right]\Delta t.
}\tag{18}
$$

$TUTT$ 保留为 UC 合法报价底价与消费者月累计参考信息。$E_{i,t+1}^u=E_{i,t}^u+(p_{i,t}^{u}+p_{i,t}^{back})\Delta t$。式 (18) 是本研究的双 PAB **结算合同**，与原论文直接用 TUTT 结算 UC 电量的式 (7)、(23) 不同。

### 5.2 UC 根据真实普通成交落实 PG 与 ESS

定义普通 UC 已售出功率 $U_t=\sum_i p_{i,t}^{u}$，DER 已售 $D_t^r=\sum_i p_{i,t}^{r}$。延续 v5 已确认的低成交电池放电下调规则：

$$
\boxed{
 p_t^{o,exec}=\min\{p_t^{o,plan},\ U_t+p_t^{e,plan}\},\qquad
 g_t^{base}=U_t+p_t^{e,plan}-p_t^{o,exec}.
}\tag{19}
$$

ESS 充电按可行计划保持 $p_t^{d,exec}=p_t^{d,plan},\ p_t^{e,exec}=p_t^{e,plan}$；普通 DER 未售出部分进入弃电：

$$
\boxed{p_t^{a,exec}=p_t^g-p_t^{d,plan}-D_t^r.}\tag{20}
$$

当 PG 基础量校验为 $0\le g_t^{base}\le\bar p_t^{b,phys}$ 后，本时段 PG 可调用备用上限为：

$$
\boxed{P_t^{back,cap}=\min\left\{\bar P_t^{back,contract},\ \bar p_t^{b,phys}-g_t^{base}\right\}.}\tag{21}
$$

### 5.3 备用有限、必要负荷优先、无需议价

普通成交后的缺口 $u_{i,t}=d_{i,t}-p_{i,t}^{r}-p_{i,t}^{u}\ge0$。备用池先处理每人的未满足必要服务 $u_{i,t}^{ess}$，再处理未满足可延期任务 $u_{i,t}^{opt}$；在同一等级中，初版按缺口比例分配（**备用不按报价排序**），避免引入第三场拍卖。

$$
\boxed{p_{i,t}^{back}=p_{i,t}^{back,ess}+p_{i,t}^{back,opt},\quad
B_t^{back}=\sum_i p_{i,t}^{back}\le P_t^{back,cap}.}\tag{22}
$$

所有备用消费者仍按**自己的 $b_i^u$**支付，备用没有独立 $T_t^{back}$ 零售价。额外外部备用服务费 $h_t^{back}$ 与调用费 $a_t^{back}$ 由 UC 在真实成本账本支付：

$$
\boxed{C_t^{back,extra}=(h_t^{back}+a_t^{back}) B_t^{back}\Delta t.}\tag{23}
$$

如果当前剩余需要均已由普通成交覆盖，则 $B_t^{back}=0$。如果备用容量不足，不能再默默生成 PG 功率；结算真实缺口，并进入服务损失和待办队列。

### 5.4 电力执行守恒

$$
\boxed{p_t^{b,exec}=g_t^{base}+B_t^{back}\le\bar p_t^{b,phys}.}\tag{24}
$$

$$
\boxed{
 p_t^{b,exec}+p_t^{g}+p_t^{o,exec}=
 p_t^{a,exec}+p_t^{c,exec}+\sum_i\left(p_{i,t}^{r}+p_{i,t}^{u}+p_{i,t}^{back}\right).
}\tag{25}
$$

其中 $p_t^{c,exec}=p_t^{d,plan}+p_t^{e,plan}$。实际 ESS 能量为：

$$
\boxed{E_{t+1}^{ESS}=E_t^{ESS}+\eta_c p_t^{c,exec}\Delta t-\frac{p_t^{o,exec}\Delta t}{\eta_o},\quad E^{min}\le E_{t+1}^{ESS}\le E^{max}.}\tag{26}
$$

同一 PG 购电量在 UC 成本账本中只记一次；普通和备用归类只是销售及服务记录，**不改变实际购电的物理定义**。

---

## 6. 实际服务、移峰任务与消费者行动可行性

### 6.1 Actor 的回补动作和物理交付不同

本轮净需求由 Actor 直接决定，不做后验 `clip`；然而总供电有限时，合法的回补计划未必有电执行。这一差异不能从物理账本删除。最终交付功率与原生优先规则为：

$$
\begin{aligned}
p_{i,t}^{served}&=p_{i,t}^{r}+p_{i,t}^{u}+p_{i,t}^{back},\\
p_{i,t}^{native,served}&=\min\{n_{i,t},p_{i,t}^{served}\},\\
\ell_{i,t}^{native}&=n_{i,t}-p_{i,t}^{native,served},\\
r_{i,t}^{served}&=\min\{r_{i,t}^{rep},[p_{i,t}^{served}-n_{i,t}]_+\},\\
\ell_{i,t}^{rep}&=r_{i,t}^{rep}-r_{i,t}^{served}.
\end{aligned}\tag{27}
$$

合同优先级先保障必要用电；消费者内首先满足当前原生负荷，然后按历史任务截止期/FIFO 服务。待办电量只按**真正完成**的回补消除：

$$
\boxed{B_{i,t+1}=B_{i,t}+(s_{i,t}^{+}-r_{i,t}^{served})\Delta t.}\tag{28}
$$

若本轮能完整满足 $d_i$，式 (27) 给出 $r_{i,t}^{served}=r_{i,t}^{rep}$，可直接用 $B_{i,t+1}=B_{i,t}+\Delta p_{i,t}^{s}\Delta t$。只有真实缺供场景才需要单列未完成回补，**不是对 Agent Action 再做可行性修正**。

每笔正移峰产生 $(\mathrm{energy\_kwh},\mathrm{origin\_t},\mathrm{deadline\_t})$ 的任务，截止时间为 $t+H_i$，同一 $H_i$ 下 FIFO 与最早截止一致。逾期任务继续保留，按合同缴付违约/不适代价；跨日和跨月保留任务状态，回合中途截断不得人为清零。

### 6.2 服务缺口与真正物理响应

$$
\boxed{d_{i,t}=p_{i,t}^{served}+\ell_{i,t}^{native}+\ell_{i,t}^{rep}.}\tag{29}
$$

这条等式把**失去当前原生服务**与**历史回补未完成**分开记录。用户自己主动削减的负荷不属于失供；与市场抢不到电造成的失供不可混同。

---

## 7. Consumer Observation：公开合同 + 本人真实任务

### 7.1 保留原十字段的顺序

$$
\boxed{\bar o_{i,t}=\left[t,p_{i,t}^{*},E_{i,t}^{u},\Delta p_{i,t}^{max},T_{i,t}^{u},T_t^d,T_t^r,p_t^m,P_t^{u,avail},P_t^{r,avail}\right].}\tag{30}
$$

这些字段分别提供时间、基线负荷、累计 UC 电量、DR 能力、UC 合法底价、DER 底价、DR 单价、可补偿采购容量、UC 可售容量和 DER 可售容量。

### 7.2 追加消费者本人到期任务与公共备用额度

把本地 FIFO 队列按截止剩余时段展开：

$$
\boxed{\mathbf q_{i,t}^{due}=\left(q_{i,t}^{(0)},q_{i,t}^{(1)},\ldots,q_{i,t}^{(H)}\right),\quad B_{i,t}=\sum_{h=0}^{H}q_{i,t}^{(h)}.}\tag{31}
$$

$h=0$ 包括已经逾期和本时段到期的任务；$h\ge1$ 为 $t+h$ 截止的任务，均为 kWh。使用固定上限 $H$ 并补零。新观测：

$$
\boxed{o_{i,t}=\left(\bar o_{i,t},\ \mathbf q_{i,t}^{due},\ \bar P_t^{back,contract}\right)\in\mathbb R^{H+12}.}\tag{32}
$$

在 $t$ 仅用一个标量编码的约定下，维度是 $10+(H+1)+1=H+12$；展开周期编码时应按 schema 自动调整维度。若需显式告知消费者资源优先级，该规则为**公共静态合同配置**，无需每期动态拼接。

**信息边界**：消费者只能看到自己任务、自己的静态能力、公开供给与历史；其他消费者未提交的报价、真实当期缺口和本轮 PG 最终备用余量不在 Actor Observation 内。真正动态的备用剩余能力只能在出清后知道，不能事前泄漏进 Observation。集中 Critic 可以使用全局历史和全部当前真实环境状态。

### 7.3 UC 的观测与双阶段信息

UC Actor 只用该时刻能获取的 PG 价格、DER 预测、ESS、历史峰值/服务记录、公开聚合负荷预测与合同参数；不能先窥视本轮各消费者未来随机动作。UC 宣告后，消费者 Actor 读到 UC 公告。训练时可以为两个阶段建立不同的 centralized Critic，但不能把消费者当期已结算结果拼入其决策前 Actor 观测。

---

## 8. DR 核验、采购上限与补偿（仍保留 v5 的 Q3=B 与 Q5=B）

### 8.1 只奖励可验证的主动正响应

$$
\boxed{q_{i,t}^{ver}=\chi_t\left[c_{i,t}+s_{i,t}^{+}-r_{i,t}^{served}\right]_+.}\tag{33}
$$

它还可由实际负荷服务审计：

$$
\boxed{q_{i,t}^{ver}=\chi_t\left[p_{i,t}^{*}-(p_{i,t}^{served}+\ell_{i,t}^{native})\right]_+.}\tag{34}
$$

$\ell^{native}$ 加回服务基线，是为了**不把断电造成的少用电当成主动 DR**；$r^{served}$ 抵减正响应，防止消费者只做回补却领取净减载奖励。未来移峰任务仍须按真实队列偿还。

### 8.2 在 UC 公布的可补偿采购上限内付款

令 $Q_t^{ver}=\sum_iq_{i,t}^{ver}$，本版按核验响应比例分摊已采购份额：

$$
\boxed{q_{i,t}^{paid}=\begin{cases}
q_{i,t}^{ver}\min\left\{1,\frac{p_t^m}{Q_t^{ver}}\right\},&Q_t^{ver}>0,\\
0,&Q_t^{ver}=0.
\end{cases}}\tag{35}
$$

$$
\boxed{Q_t^{paid}=\sum_i q_{i,t}^{paid}=\min\{p_t^m,Q_t^{ver}\},\quad
I_{i,t}^{DR}=T_t^r q_{i,t}^{paid}\Delta t.}\tag{36}
$$

事件外 $\chi_t=0,p_t^m=0,T_t^r=0$。$p_t^m$ **不是**消费者物理 DR 上限：即使 $p_t^m=0$，消费者仍可自主调整自己的合法负荷，但没有 UC 激励采购收入。

---

## 9. 原论文式 (10) 与 v7 的 DR 平方价值：引用、适配和边界

### 9.1 原文到底写了什么？

Nie 等（2024），IEEE TSG，印刷页 **5625**，§II-B，式 **(10)**：

$$
\boxed{C_{t,\mathrm{original}}^{r}
=T^{s}\frac{\left(\sum_{i\in\mathcal I}\Delta p_{i,t}\right)^2}{p_{t,\mathrm{original}}^{m}}
-T^{r}\sum_{i\in\mathcal I}\Delta p_{i,t}.}\tag{37}
$$

原文称 $T^s$ 为 **DR benefit tariff**，$T^r$ 为 **DR flexibility purchase price**，$p_{t,\mathrm{original}}^m$ 为 UC **expected DR power**；原文另以式 (31) 约束消费者合计 DR 不超过预期 DR 功率。原文的 $C_t^r$ 是 UC 收益函数的 DR 分量，非外部电网实际汇入 UC 账户的已证实现金流水。原文未在式 (10) 中明确乘 $\Delta t$；其仿真参数及单位必须跟其自身时间尺度一致。

**E01 引用纠正**：PDF 第 4 页（印刷页 5625）的式（10）为“总响应平方除以目标功率”，不是“响应/目标之比再平方”。指定 v7 的式（37）误引，现已改正。原式在 $0\le Q\le p^m$ 条件下有 $Q^2/p^m\le p^m$，其沿可行路径趋于零与解除该约束后固定 $Q$、令 $p^m\to0$ 的发散不能混为一谈；$p^m=0$ 的定义仍需单列。新式（38）是研究者另设的有界状态下价值函数，不是原式的代数恒等改写。

### 9.2 为什么原式不能直接替换到 v7？

本研究已经决定：$p_t^m$ 是**可补偿采购容量上限**，而不再是原论文的“合计物理 DR 的目标上界”；由于 $p_t^m$ 是 UC 动作，若分母仍是 $p_t^m$，当 $p_t^m\to0^+$ 时收益函数可能虚增/发散，而 $p_t^m=0$ 时无定义。并且原文 $\sum_i\Delta p_i$ 允许含负的移峰响应，本研究需要按事件、主动响应和已采购服务分别核验。**这两个建模分歧不能靠给分母随便加 $10^{-6}$ 解决**。

### 9.3 v7 的正式适配式：保留原文平方收益结构

设 $P^{DR,ref}>0$ 是**从历史参考日或合同目标预先确定、全程固定的参考 DR 功率**，不是 UC 可修改的动作。定义 $\beta_t^{DR}\ge0$ 为“把归一化响应平方映射为本时段价值”的收益系数，单位 **元/时段**，具体数值须由仿真假设、可靠性价值或明确的 DR 合同标定。正式使用：

$$
\boxed{
G_t^{DR}=\beta_t^{DR}\left(\frac{Q_t^{paid}}{P^{DR,ref}}\right)^2,
\qquad C_t^{DR,purchase}=T_t^r Q_t^{paid}\Delta t.
}\tag{38}
$$

$$
\boxed{C_t^{DR,net}=G_t^{DR}-C_t^{DR,purchase}.}\tag{39}
$$

该适配同时继承：**总 DR 响应平方收益**、**按响应支付采购费用**和**UC 在 Reward 中权衡两者**，但与原式在归一化分母、DR 响应核验口径与时间单位上存在明示区别。因此论文中应称“借鉴原文式 (10) 的 DR 增益结构并进行单位化适配”，不能声称完全照搬原式。

这里选择 **$Q_t^{paid}$ 而不是 $Q_t^{ver}$** 进入收益，是为了使“仅对实际向 UC 交付、支付的服务获得增益”的经济口径一致：当 $p_t^m=0$，有 $Q_t^{paid}=0$，于是 $G_t^{DR}=C_t^{DR,purchase}=0$。即便消费者自行减载，也不凭空形成 UC 已采购的合同 DR 增益。

### 9.4 DR 采购为什么可能变得正向？

在某个固定消费者响应下，如果 $Q_t^{ver}\ge p_t^m$，则 $Q_t^{paid}=p_t^m$，边际 DR 净收益（忽略其他市场成本与消费者重新响应）为：

$$
\boxed{\frac{\partial C_t^{DR,net}}{\partial p_t^m}
=\frac{2\beta_t^{DR}p_t^m}{(P^{DR,ref})^2}-T_t^r\Delta t.}\tag{40}
$$

如果已采购响应很少且该导数为负，UC 仍可能选择 $p_t^m=0$；如果达到足够响应规模、系统价值参数大、或相应 DR 降低其他供电/服务成本，正采购可成为更好策略。由于原文使用的**平方是凸函数**，在只保留这两个项的简单静态比较中最优采购可能偏向 0 或上界，而非稳定的内部解；应通过敏感性实验报告该现象，而不是宣称加上增益就一定解决零采购。

### 9.5 必须分开记录真实现金利润和“研究目标价值”

$G_t^{DR}$ 在基础版本是借鉴原文的**系统服务价值/经济效用项**；如果没有外部结算合同，**不可把它伪装成 PG 的现金转账**。UC 的经营账本利润 $\Pi_t^{book}$ **不包含** $G_t^{DR}$；UC 的强化学习目标 $R_t^u$ 可以包含它。论文报告分别展示 `cash_profit`, `book_profit`, `noncash_cost`, `dr_system_benefit`, `dr_payment`, `uc_reward`。式（41）先给经营账本利润；真正现金口径见第 18 节。若将来采用外部 DR 服务合同，应改写 $G_t^{DR}$ 为真正合同价值函数并标注付款方，不能把该项再重复加一次。

---

## 10. 财务账本、服务责任与各方 Reward

### 10.1 UC 经营账本与现金口径（E02）

令 $\mathrm{Rev}_t^{PAB}=\sum_i C_{i,t}^{energy}$。成本包括真实 PG 进口、备用额外外部费用和调用费用、ESS 吞吐、DER 弃电机会成本、事件 DR 实际补偿及合同赔付。定义：

$$
\begin{aligned}
\Pi_t^{book}={}&\sum_i\left[b_{i,t}^{r}p_{i,t}^{r}+b_{i,t}^{u}(p_{i,t}^{u}+p_{i,t}^{back})\right]\Delta t\\
&-c_t^{PG}p_t^{b,exec}\Delta t
-(h_t^{back}+a_t^{back})B_t^{back}\Delta t\\
&-C_t^{ESS}-C_t^{curt}-C_t^{DER}
-\sum_i I_{i,t}^{DR}-\sum_i C_{i,t}^{comp}.
\end{aligned}\tag{41}
$$

**不要重复扣除 PG 备用购电成本**：它已经包含在 $c_t^{PG}p_t^{b,exec}\Delta t$ 中。$C_t^{comp}$ 为实际合同赔付（记入双方现金流），不应再在“利润”里把同额法定赔付作为服务惩罚重复扣除。$C_t^{ESS}$、$C_t^{curt}$ 必须分别注明吞吐成本和弃电机会成本的实际或模拟来源。

### 10.2 UC 的必要服务责任（问题 1 的第二部分）

为了避免 UC 因边际报价低于 PG 成本而故意断供，需要把履行服务的责任写成合同约束或有明确物理意义的服务考核项。记本轮原生失供和到期任务未完成金额为：

$$
\boxed{
L_t^{u,service}=\lambda_t^{native}\sum_i\ell_{i,t}^{native}\Delta t
+\lambda_t^{due}\sum_i\ell_{i,t}^{due}\Delta t.
}\tag{42}
$$

$\ell_i^{due}\le\ell_i^{rep}$ 为截止期内未完成的应回补功率，相关电量进入既有 FIFO 逾期状态。$\lambda^{native},\lambda^{due}$ 为元/kWh，可解释为**无补偿情况下的可靠性损失评估**或明确列出的合同违约费用。若已经通过 $C^{comp}$ 完整计入相同的法律赔付，则对应 $L^{service}$ 不应重复计为另一笔现金支出；作为训练目标的附加考核可单独展示。

对于固定同价、同一 kWh 普通 PG 供电，不卖与卖的最简单比较：若节省购电净损失 $c^{PG}-b^u$，却承担服务损失 $\lambda^{native}$，则只要 $\lambda^{native}>c^{PG}-b^u$，就不会因该**局部固定物理对照**主动减少必要供电。这个阈值并不保证跨时段、不同电池轨迹和联合价格策略都不会策略性限供；必须在策略评估中检验。

### 10.3 UC 与 Consumer 的目标函数

$$
\boxed{R_t^u=\Pi_t^{book}+G_t^{DR}-L_t^{u,service}-C_t^{peak}.}\tag{43}
$$

基础实验令 $C_t^{peak}=0$，本候选规则采用另设的 DR 二次效用，并未假定存在真实需量收费。若开展真实 PG 需量费用消融，可使用 $M_{t+1}=\max(M_t,p_t^{b,exec})$ 和 $C_t^{peak}=\nu(M_{t+1}-M_t)$，并注明计费周期口径。

消费者成本为：

$$
\begin{aligned}
C_{i,t}^{consumer}={}&C_{i,t}^{energy}+C_{i,t}^{adjust}
+\lambda_i^{native}\ell_{i,t}^{native}\Delta t\\
&+C_{i,t}^{waiting}+C_{i,t}^{overdue}
-I_{i,t}^{DR}-C_{i,t}^{comp},\\
&R_{i,t}^c=-C_{i,t}^{consumer}.
\end{aligned}\tag{44}
$$

一种明确可复现的本地调节代价为：

$$
\boxed{C_{i,t}^{adjust}=\kappa_i\frac{c_{i,t}^2+(s_{i,t}^{+})^2}{\max(p_{i,t}^{*},\varepsilon_p)}\Delta t.}\tag{45}
$$

等待/逾期按真实任务队列余额与截至当前的期限计费。为了体现可延期任务不会永远免费压在最终一天，真正的有限回合结束时加入终端待办责任项，而采样片段的截断不视为真实终止。

---

## 11. 两项经济激励审核及局限

### 11.1 问题 1 的正确比较：同服务、同 PG 成本下，普通换备用

假设同一消费者本轮报价 $b^u=0.40$ 元/kWh，PG 成本 $0.65$，备用外部费用 $0.10$，UC 调用费用 $0.20$，负荷 $4$ kW、时长 $0.25$ h。电量是 $1$ kWh。

普通供电 UC 利润（不考虑其他成本）为：

$$
\boxed{\Pi^{ordinary}=0.40-0.65=-0.25\ \text{元}.}\tag{46}
$$

同样完整服务、但全部归到备用时：

$$
\boxed{\Pi^{backup}=0.40-0.65-0.10-0.20=-0.55\ \text{元}.}\tag{47}
$$

所以同一报价和同一总服务量下：

$$
\boxed{\Pi^{backup}-\Pi^{ordinary}=-0.30\ \text{元}.}\tag{48}
$$

这消除了 v5 中“普通亏 $0.25$，备用仅亏 $0.20$”的**渠道切换套利**。如果 UC 试图不服务这 1 kWh，那么 $\Pi^{unserved}=-\lambda^{native}$（扣除对应惩罚后），取 $\lambda^{native}>0.25$ 可使其不优于普通完整服务；其他复杂成本需额外核对。只靠一个有限违约罚价，不足以证明 UC 在所有场景都不策略性限供。

### 11.2 问题 2 的改善边界：备用同价但并不保证双 PAB 非退化

同一消费者的普通 UC 和备用 UC 都按 $b^u$ 计费，故其在固定报价下从普通变为备用不再额外省钱。但如果**任意合法报价都可确保全部需求被 PG 备用补足**，消费者总 UC 电量恒定，费用 $b^u(d_i-p_i^r)\Delta t$ 将随 $b^u$ 上升，消费者最佳反应容易退化为 UC 合法报价底价。

本研究不人为抬高备用零售单价，而采用：

- 有上限的 $\bar P^{back,contract}$ 和 PG 物理容量；
- 在普通双 PAB 中引入必要/可延期服务顺序，同等级内按报价竞争；
- 备用先满足必要缺口，不保证能覆盖所有可延期请求；
- 通过**备用充足度扫描**检验何时全体 UC 报价退化到下限。

**必须承认的边界**：有限备用和必要性优先只能保留发生资源稀缺时的竞价动机；若试验场景始终供过于求，Pay-as-Bid 最优报底价本身可能是合理结果，不应以人工惩罚强制生成报价竞争。

### 11.3 问题 3 的解决程度

将本人任务到期桶和合同额度加入 Actor 输入并直接使用合法动作分布，能够消除“同一公开观测下因隐藏本地队列不同而被 `clip` 出不同 DR 执行值”的缺陷。其他消费者的当前报价、实际市场分配仍然未知，所以**系统整体依然是部分可观测的 Markov Game**；这不是错误，也不是隐私的形式化证明。

---

## 12. 一个时段的完整算例（三位消费者、双资源竞价与有限备用）

所有功率以 kW 计，$\Delta t=0.25$ h。UC 无 ESS 充放电，DER 当期发电 $p_t^g=4$、可交易 $P_t^{r,avail}=4$；计划 PG/普通 UC 容量 $P_t^{u,avail}=6$，PG 接口最大 $8$，备用合同 $2$，PG 成本 $0.65$ 元/kWh。$h^{back}=0.10$，$a^{back}=0.20$ 元/kWh。

消费者观测与动作：

| 消费者 | 原负荷 $p^*$ | 削减 $c$ | 移峰 $\Delta p^s$ | 净需求 $d$ | $\rho^r$ | DER 报价 | UC 报价 | 必要 $m$／可延期 $v$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 1 | +1 | 4 | 0.5 | 0.70 | 0.90 | 4 / 0 |
| 2 | 5 | 1 | 0 | 4 | 0.5 | 0.65 | 0.85 | 4 / 0 |
| 3 | 4 | 0 | −1 | 5 | 0.4 | 0.60 | 0.80 | 4 / 1 |

消费者 3 当前有不少于 $0.25$ kWh 的尚**未到期**历史待办，所以这 1 kW 的本轮回补请求合法但可延期；三位消费者本地调节上限与 FIFO 期限均假设合法。

**DER 初始申报**：$q^{r,0}=(2,2,2)$，总请求 6；实际 DER 可售 4，必要优先且按 DER 买方报价排序，得到 $p^r=(2,2,0)$。

**UC 条件需求**：$q^{u,cond}=d-p^r=(2,2,5)$，其中必要部分为 $(2,2,4)$，可延期部分为 $(0,0,1)$。普通 UC 可售 6，必要优先且按 UC 报价排序，得到 $p^u=(2,2,2)$。

**PG 备用**：普通 PG 基础量 $g^{base}=6$，物理余量 $8-6=2$，合同容量 2，备用上限 2。只有消费者 3 还有 2 kW 必要缺口和 1 kW 可延期回补；备用先补必要，分配 $p^{back}=(0,0,2)$。

**最终服务**：$p^{served}=(4,4,4)$，合计 12 kW；三个消费者的当期原生负荷 $n=(4,4,4)$ 全部得到供应；消费者 3 本轮计划回补的 1 kW 仍未实现，因此 $r_3^{served}=0$、历史任务不减少，$\ell_3^{rep}=1$ kW，$\ell_i^{native}=0$。

**功率守恒**：PG 实际进口 $6+2=8$ kW，DER $4$ kW，ESS 0；消费者得到 $4+4+4=12$ kW，供需残差为 0。

**PAB 结算**：DER 收入 $[2(0.70)+2(0.65)](0.25)=0.675$ 元；UC 收入 $[2(0.90)+2(0.85)+4(0.80)](0.25)=1.675$ 元；UC 所得总电费为 2.35 元。消费者 3 的普通 UC 2 kW 与备用 UC 2 kW 都按自身 UC 报价 $0.80$ 元/kWh，合计电费为 $4(0.80)(0.25)=0.80$ 元。

**DR 与原文收益适配**：事件开启，核验净正响应为 $(2,1,0)$ kW；UC 公告 $p_t^m=2$ kW，按比例实际采购量为 $(4/3,2/3,0)$ kW。若 $T_t^r=0.20$ 元/kWh，则补偿总支出为 $2\times0.20\times0.25=0.10$ 元。另取实验性参数 $P^{DR,ref}=4$ kW、$\beta_t^{DR}=0.80$ 元/时段，平方 DR 价值为 $0.8(2/4)^2=0.20$ 元，净 DR 分量 $C_t^{DR,net}=0.10$ 元。

UC 的基础经营账本利润（不计 ESS、弃电和本例为零的原生缺供责任）为：

$$
\boxed{\Pi_t^{book}=2.35-8(0.65)(0.25)-2(0.10+0.20)(0.25)-0.10=0.80\ \text{元}.}\tag{49}
$$

UC 含原文 DR 系统价值的训练 Reward 则为 $R_t^u=0.80+0.20=1.00$ 元。本例中的 $0.20$ 元属于模型 DR 价值项，而不是 UC 真实收取的外部现金。

---

## 13. Stackelberg–Nash 数学问题、决策时序与训练

### 13.1 消费者 Nash 响应

给定 UC 策略 $\pi_u$，每个消费者 $i$ 利用自身观测和公开合同选择 $\pi_i$，其成本评价依赖真实的联合 PAB 分配、备用容量、服务和历史任务演进：

$$
\boxed{\pi_i^*(\pi_u)\in\arg\min_{\pi_i\in\Pi_i}\ \mathbb E\left[\sum_{t=0}^{T-1}\gamma^t C_{i,t}^{consumer}\mid\pi_u,\pi_i,\pi_{-i}^*\right].}\tag{50}
$$

不应将“各 Actor 同时更新后收益变好”误写为已找到 Nash。需要在训练后固定 $\pi_u$ 与其他消费者策略，并训练/搜索单个偏离者，估计单边最佳响应改进。

### 13.2 上层 UC 的 Stackelberg 目标

$$
\boxed{\pi_u^*\in\arg\max_{\pi_u\in\Pi_u}\ \mathbb E\left[\sum_t\gamma^t R_t^u\mid \pi_u,\ \pi_c^*(\pi_u)\right].}\tag{51}
$$

UC 候选必须让消费者**重新适应**后评估；仅比较当前固定 Consumer 动作下 $p_t^m$ 的付款变化不能代替响应后的 UC 净增益。若随 $\pi_u$ 存在多个下层 Nash，需要约定乐观/悲观或特定 Nash 选择规则，并把这一假设列作均衡结论的条件。

### 13.3 一套可执行的训练流程

1. 固定 UC 策略，使用独立 Consumer Actors 和集中价值网络，收集包含阶段公告、市场容量、服务结果和队列状态的真实轨迹；消费者 PPO 训练直到策略稳定；
2. 固定消费者策略，训练 UC Actor，包含 DR 平方价值、真实资金与服务责任；
3. 针对 UC 新候选策略，返回步骤 1 使消费者重新响应；用独立随机种子和历史数据检查 UC 的响应后回报；
4. 记录并比较所有 Agent 的最佳响应增益、供电与负荷指标、PAB 报价分布和备用使用频率；
5. 网络策略更新在**训练时间**反复进行，真实物理时段仍只调用一次 UC/Consumer 决策。

完整实现可采用 PPO/MAPPO 类算法、各 Agent 独立 Actor、集中 Critic；共享局部编码器仅作为独立消融项。原论文 Fig.4 的汇总式 encoder/decoder 不是当前五维个体动作必需的结构。

### 13.4 不同于物理单元测试的均衡评估

$$
\boxed{\epsilon_i^{NE}=J_i(\pi_i,\pi_{-i};\pi_u)-\inf_{\tilde\pi_i}J_i(\tilde\pi_i,\pi_{-i};\pi_u).}\tag{52}
$$

$$
\boxed{\epsilon_u^{SE}=\sup_{\tilde\pi_u}J_u(\tilde\pi_u,\pi_c^*(\tilde\pi_u))-J_u(\pi_u,\pi_c^*(\pi_u)).}\tag{53}
$$

有限样本和有限偏离搜索仅给近似诊断，不自动证明真实 SNE，也不自动继承原论文特定算法假设下的收敛定理。

---

## 14. 软件接口、训练适配与数据协议

### 14.1 模块划分（避免把结算掺到 Actor）

```text
RayMARL/
  RLAlgo/                              # 所有学习算法、RLModule、Learner、动作分布
    src/raymarl/                       # v7 适配为后续工作，不在本文假定已实现
  environments/sn_energy_market/
    src/snmarket/                      # 状态、约束、订单、市场、物理、结算、适配
    configs/                           # 现有 v5 配置；v7 应另设版本，不能直接混用
    tests/                             # 现有 v5 环境测试
    docs/v7/
      modeling_integrated_zh.md         # 本完整整合稿
      modeling_audit_zh.md              # 本轮审查
      source_v7_supplied.md             # 指定 v7 原样快照
      verify_v7_mechanisms.py           # 独立数学核验，不是环境实现
      verification_results.json        # 可复现结果及来源哈希
  experiments/sn_energy_market/        # 场景、训练入口、响应与偏离实验
  datasets/                            # 外部数据，不放进算法包
  outputs/                             # 运行记录、图表、checkpoint
```

环境负责数值约束和合法域描述；策略网络的分布、可逆变换、`log_prob`、优化器与学习循环由 `RLAlgo` 负责。不要为本文另在环境内部复制一套 `src/learning/`。市场、备用、服务可以按现有模块拆分或扩展，模块命名不是新的经济规则。


### 14.2 一笔订单、一条执行账本

输入的 `ConsumerOrder` 必含 `demand_kw`, `der_initial_kw`, `uc_initial_kw`, `uc_bid`, `der_bid`, `mandatory_kw`, `optional_kw`。出清结果保存 `der_essential_kw`, `der_optional_kw`, `uc_essential_kw`, `uc_optional_kw`, `uc_conditional_demand_kw`, `backup_essential_kw`, `backup_optional_kw`。

服务记录保存 `native_served_kw`, `native_lost_kw`, `repay_scheduled_kw`, `repay_served_kw`, `repay_unfinished_kw`, `due_unfinished_kw`。市场账本保存两类成交与两类报价、**无独立备用零售价**；UC 账本输出 `cash_profit`, `book_profit`, `noncash_cost`, `dr_benefit`, `dr_paid_cost`, `service_penalty`, `reward`，便于判断“改善的是现金利润还是模型价值”。

### 14.3 一次物理时段的程序轮廓

```python
uc_action = uc_actor.sample(uc_observation)
public_offer = uc_plan_and_publish(uc_action, state)
orders = []
for i in consumers:
    obs_i = build_observation(i, state, public_offer)
    action_i = actor[i].sample_in_feasible_domain(obs_i)
    orders.append(build_conditional_order(i, action_i, state))

trades_der = clear_der_pab(orders, public_offer.der_capacity)
conditional_uc = residual_uc_orders(orders, trades_der)
trades_uc = clear_uc_pab(conditional_uc, public_offer.uc_capacity)
physical_base = execute_base_supply(state, public_offer, trades_uc)
backup = allocate_limited_backup(orders, trades_der, trades_uc, physical_base)
service = decompose_actual_service(orders, trades_der, trades_uc, backup)
verified_dr, paid_dr = verify_and_procure_dr(service, orders, public_offer)
ledger = settle_same_bid_uc_and_der(orders, trades_der, trades_uc, backup,
                                   verified_dr, paid_dr, physical_base)
candidate = prepare_state_delta(state, physical_base, backup, service, ledger)
reward = build_rewards(ledger, paid_dr, service, candidate,
                       include_true_terminal_cost=True)
validate_complete_transition(candidate, ledger, reward)
new_state = commit_once(state, candidate, ledger, reward)
```

这是流程契约，不声称以上伪代码已经是可训练的 RLlib 环境。

---

## 15. 验证方案：区分通过的机制与未验证的均衡

### 15.1 机制单元测试

| 检验 | 必须检查的关系 |
|---|---|
| UC/DER 双订单 | $q_i^{r,0}+q_i^{u,0}=d_i$；$q_i^{u,cond}=d_i-p_i^r$ |
| 本地合法动作 | $a_i\in\mathcal A_i(o_i)$；不可行的期限场景显式报错 |
| 必要优先 | 稀缺资源不先满足同池的可延期请求再遗失必要服务 |
| PAB 支付 | 每个消费者的 UC 普通与备用用同一个 $b_i^u$，DER 用 $b_i^r$ |
| 备用容量 | $B_t^{back}\le\bar P^{back,contract}$，PG 总进口不超物理接口 |
| 系统守恒 | 式 (25) 残差在数值容差内为 0 |
| 未完成任务 | 只能按 $r_i^{served}\Delta t$ 消除，不能按 Action 直接减债 |
| DR 核验 | 断电不产生主动 DR；负回补不能凭空领取正响应补偿 |
| DR 采购 | $Q_t^{paid}=\min(p_t^m,Q_t^{ver})$；$p_t^m=0$ 时不存在除零 |
| 两套 UC 经济指标 | `cash_profit` 不包含尚无付款方的 $G_t^{DR}$ |

### 15.2 最小经济回归情景

A. **渠道转换**：同一个 $b^u$、同物理供给和同服务，改为备用时 UC 利润下降 $(h+a)e$；消费者 UC 电费不变。

B. **策略性断供**：UC 报价低于 PG 成本时，取不同服务违约权重，观察不供电与履约的收益切换；报告哪一类场景仍可能限供。

C. **备用稀缺度**：固定消费者基线与随机种子，遍历 $\bar P^{back,contract}/\sum_i d_i$，比较 UC 报价分布是否退化到底价、失供风险、DR 量与服务损失。

D. **观测缺失**：同样的 10 基础字段、不同到期任务状态，验证带到期桶的 Actor 合法回补动作边界确实不同；去掉桶作为消融。

E. **DR 价值归一化**：令 $p_t^m=0,10^{-6},0.1,1,\bar P$，验证收益连续有限；比较不同 $\beta^{DR}$ 和固定 $P^{DR,ref}$ 下采购分布，区分现金利润与系统价值。

### 15.3 训练/均衡实验

训练时至少同时报告：UC 现金利润、UC 综合 Reward、总 DR 实际削减、付费 DR、每人到期任务服务率、供电缺口、最大 PG 进口、弃 DER、电池 SOC、个人成本、价格分布、UC 报价底价比例、DR 采购上限、消费者最优偏离增益及 UC 响应后收益。训练稳定不自动等于均衡，必须附独立最佳响应实验。

---

## 16. 原文对照、参数说明和保留的研究风险

| 内容 | 原论文 | v7 |
|---|---|---|
| 供电结算 | UC 采用 TUTT；DER 按原文激励价 | UC 与 DER 都按**消费者自己的单段报价 PAB**；TUTT 为合法 UC 报价底价 |
| Consumer 需求 | 原文下层功率动作含 $p_i^u,p_i^r,\Delta p_i^l,\Delta p_i^s$ | 五维 DR + $\rho_i^r$ + 双报价，订单推导 $q_i^u,q_i^r$ |
| $p_t^m$ | 原文预期 DR 功率并参与式 (10) 分母 | v7 为**补偿采购上限**，不再作为收益分母 |
| DR 收益 | 原式 (10) 为总响应平方除以目标功率，再减 DR 支出 | 用 $Q^{paid}/P^{DR,ref}$ 构造同类平方价值，分开现金与系统价值 |
| Action 约束 | 原论文采用安全掩码思想 | 状态相关合法动作采样、本人任务截止约束 |
| PG 备用 | 本研究 v5 新增机制，不是原文原式 | 有限容量、必要优先、普通和备用按同一 UC 报价 |
| 下层训练 | 原论文共享 Encoder + 个体 Decoder | 消费者独立 Actor，集中 Critic，可另作共享局部 Encoder 消融 |
| 均衡结论 | 原论文提出 SN-MAPPO 与特定收敛分析 | 本研究数值训练需重新评估；不继承未验证定理 |

需特别说明的开放条件：

1. **同价备用不会自动保证报价多样性**。备供充足时 UC 报价可能趋底价；这是模型内生经济结果，不应隐藏。
2. **不引入 $F_i$ 就不能保证所有联合 DR Action 都有全额电力**。合法 DR Action、物理交付和合同违约分开记录。
3. **服务责任需要现实或实验依据**。若设得过大，可相当于把 UC 强制服务硬约束；研究中应报告敏感性。
4. **加入平方增益不保证内点非零采购**，可导致全额/零额边界解；需做参数扫描与响应后检验。
5. **收益函数里模拟的 DR 价值不是银行流水**，也不得与另设的外部 DR 合同付款重复计算。
6. **优先级会影响公平与报价策略**，必须同时展示不同必要性分类与无优先级 PAB 的对照结果。
7. **网络潮流、负载电压、线路热限、配电网损耗**暂未加入；声称现实多节点物理可行须另有电网模型。

### 参考文献

[R1] Y. Nie, J. Liu, X. Liu, Y. Zhao, K. Ren, C. Chen. *Asynchronous Multi-Agent Reinforcement Learning-Based Framework for Bi-Level Noncooperative Game-Theoretic Demand Response*. IEEE Transactions on Smart Grid, 15(6), 5622–5637, 2024. DOI: [10.1109/TSG.2024.3417535](https://doi.org/10.1109/TSG.2024.3417535)。重点 §II-B 式 (5)、(7)–(10)，§II-C 式 (21)–(31)，§III-D Algorithm 1 与图 4。

[P1] 《Stackelberg–Nash Pay-as-Bid Modeling v5》，此前批准的 Q1=A、Q2=A、Q3=B、Q4=A、Q5=B。本文调整 Q2 的**消费者备用零售价格**为本人 UC 买方报价；备用 PG 成本和外部调用费用仍真实入账；Q3、Q4、Q5 的任务与 DR 核验原则继续保留。

[P2] 指定 v7 文末提及《DER PAB / UC TUTT Modeling v6》，但本轮未取得该独立文件。此条仅保留来源追踪；不根据其名称补造 v6 规则或审批事实。


---

## 17. 全量状态、日历、任务和失效边界

### 17.1 两个决策时刻和外生信息

定义物理期初、UC 行动前状态：

$$
s_t^U=\big(t,\tau_t,p_t^g,c_t^{PG},E_t^{ESS},\chi_t,
\{p_{i,t}^*,\Delta p_{i,t}^{max},E_{i,t}^u,\mathcal Q_{i,t}\}_{i=1}^N,
\text{配置版本、历史汇总、已提交键、随机数状态}\big).
$$

其中 $\tau_t$ 为带时区时间戳，$\mathcal Q_{i,t}$ 保存每笔来源期、截止期及剩余 kWh。UC 公告后、消费者采样前的状态为：

$$
s_t^C=(s_t^U,a_t^u,\mathrm{PublicOffer}_t).
$$

Actor 使用相应时刻的信息投影。PPO 的 UC Value 使用 $s_t^U$，消费者 Value 使用 $s_t^C$；当前成交、当前补供和当前 Reward 不能作为同一决策的 Value 基线输入。动作价值型 Critic 可按其算法定义接收联合动作，但不能把动作后结果泄漏给 Actor。

**补充约定 S01**：本整合稿的确定性主场景采用本时段可观测的 DER 可用出力与实际 PG 单价，公告—出清—结算使用同一份样本。第 7.3 节所谓“DER 预测”在该基准中取当前可用值。若加入预测误差，应独立定义预测值、实现值、误差发生时刻、缺额承担与补救规则；当前公式不自动覆盖这种扩展。

数据至少包含 `timestamp`, `der_generation_kw`, `actual_grid_cost_per_kwh`, 各消费者 `baseline_kw` 和 `dr_max_kw`。价格和负荷取值来自数据或明确的合成生成器；不得用论文曲线图反推数据后宣称获得原始训练集。

### 17.2 跨月、FIFO 和终端

采用以下可执行边界约定，替代分散的“沿用”表述：

1. 一期是 $[\tau_t,\tau_t+\Delta t)$。报价底价取期初累计量；同一期不事后提高已提交报价。它是 TUTT 引导的报价下界，不是原论文直接按 TUTT 边际电量收费的完全复刻。
2. 先计入本期实际普通 UC 和备用电量，再归档月结余额；若下一时刻进入新月，再把下一月累计置零。DER 电量不混入 UC 月累计。跨计费边界的数据必须对齐时段，或事先切分。
3. 移出电量 $s_i^+\Delta t$ 生成截止期为 $t+H_i$ 的新任务。允许在截止期对应的物理时段完成；该时段结束仍未完成的部分进入逾期。期初 $h=0$ 桶包含截止期不晚于当前时段的任务。
4. 先服务当前原生负荷，再按最早截止/FIFO 扣历史任务；只有 $r^{served}\Delta t$ 可以销账。相同 $H_i$ 下 FIFO 与截止顺序一致。桶只汇总观测，明细仍保存在状态中。
5. 训练采样切片不重置月账、SOC 或任务。真实有限期结束时报告全部未完成任务；移峰终端责任进入最后一步 Reward。用有限罚项允许剩余任务，不等同于原论文硬性的全时域移峰零和约束。

消费者等待、逾期和终端项补充为：

$$
C_{i,t}^{waiting}=\lambda_i^{wait}B_{i,t+1}\Delta t,\qquad
C_{i,t}^{overdue}=\lambda_i^{over}B_{i,t+1}^{over}\Delta t,
$$

$$
C_{i,t}^{terminal}=\mathbf1_{t=T-1}\lambda_i^{term}B_{i,T}.
$$

前两项系数为元/(kWh·h)，终端系数为元/kWh。式（44）在真实最后一步另加 $C^{terminal}$。UC 若承担期末剩余任务责任，以同样方式另加 $\lambda_u^{term}\sum_iB_{i,T}$，与式（42）的当期服务考核分栏，不能称同一法律赔付的第二笔现金流。

### 17.3 本地可行不等于可持续运行

**未闭合 U01**：式（6）—（9）只定义 $r^{min}\le\min(\Delta p^{max},B/\Delta t)$ 时的合法分布。v7 同时允许缺供后任务滚存，因此下一期可能出现 $r^{min}>r^{max}$；此时 $\mathcal A_i(o_i)=\varnothing$，环境转移和策略分布不能继续。它可能由合法动作和真实缺供到达，不能只靠 reset 时检查场景来排除。证据 C03。

简单候选修复是：在采样前显式计算“本期最大可尝试回补”与“不可避免的期限违约”，允许未完成任务继续滚存，并承担真实逾期代价；即把截止从硬动作下界改为受能力约束的服务目标。该候选会改变 v7 硬期限语义，**本稿不偷偷采用**。另一选择是给出可达性/鲁棒可行约束，但会增加模型复杂度。

若存在物理不可削减、不可移出的原生下限 $L_{i,t}$，应同时约束 $c_i+s_i^+\le p_i^*-L_i$。原 v7 仅建议修改削减上限，而式（7）仍允许移出全部剩余原生负荷，不能保护 $L_i$；证据 C14。未启用该物理下限时明确取 $L_i=0$。

### 17.4 终端 SOC 与仿真初始状态

每次 reset 必须记录初始 SOC、队列、月累计和计费月份。若每个独立 episode 免费恢复相同高 SOC，UC 可能在末期出售初始电量以增加回报；即使电池退化成本不为零，也不自动计入电量机会成本，见 C12。

**未闭合 U02**：应在“连续运行跨 episode 传递 SOC”“相同首尾 SOC”“终端储能价值”中选择并固定一种。终端价值不可同时作为真实现金收入和研究效用重复加计。当前 v7 未明确，不能将不同剩余 SOC 的现金收入直接解释为可持续运营收益提升。

v7 式（26）没有自放电项，本整合稿按自放电为零解释；v5 可配置的自放电不是被无声继承。加入非零自放电时，应同步修改动作能量界和状态方程。

## 18. 资金、经营成本和研究效用的完整口径

### 18.1 支付主体与资产边界

消费者电费进入 UC；UC 支付 PG 购电、真实备用外部服务和 DR 补偿。共享 DER 的经营权不等于免费产权：基础合成算例取 $C^{DER}=0$，必须写明仅是算例约定；真实研究需选择 UC 自有资产的运行成本，或第三方 DER 的采购/分成规则。不得既把全部 DER 售电收入归 UC，又遗漏已存在的 DER 所有者付款。

UC 给消费者的每笔 DR 补偿、实际缺供赔付均为双方等额反向现金流。满意度、任务等待、非现金可靠性损失和 $G^{DR}$ 没有交易对手，分别进入效用列。

### 18.2 修正式（41）的名称，而不改变其目标数值

把各成本拆成有真实付款的 $C^{cash,out}$ 与非现金经济成本 $C^{noncash}$。例如，电池退化通常是估计的资产消耗，弃电机会成本通常是未实现价值；若确有对应发票/罚款，才把该部分归入现金。

$$
\Pi_t^{cash}=\mathrm{Rev}_t^{PAB}-c_t^{PG}p_t^{b,exec}\Delta t
-C_t^{cash,out}-\sum_iI_{i,t}^{DR}-\sum_iC_{i,t}^{comp},
$$

$$
\Pi_t^{book}=\Pi_t^{cash}-C_t^{noncash},\qquad
R_t^u=\Pi_t^{book}+G_t^{DR}-L_t^{u,service}-C_t^{peak}-C_t^{u,terminal}.
$$

这里 $C^{cash,out}$ 包含确认为支付的备用外部费/调用费、DER 采购、运维等；已在现金账本扣过的项不能再扣一次。$C^{peak}$ 也是实际费用时应放入现金账本并从外层移除。第 12 节算例将未列出的成本取零、把调用费视作支付，因此其 $\Pi^{cash}=\Pi^{book}=0.80$ 元；这不适用于所有配置。

应核对的转移支付恒等式为：

$$
\sum_i(\mathrm{Bill}_i-I_i^{DR}-C_i^{comp})-\Pi_t^{cash}
=\text{本期 UC 向 PG 和其他外部主体的真实支付}.
$$

### 18.3 三种 DR 指标必须分开

原 v7 的规则保持：$q_i^{ver}=[c_i+s_i^+-r_i^{served}]_+$（事件内），$Q^{paid}=\min(p^m,\sum_iq_i^{ver})$。

同时新增诊断量，而不偷偷替换付款规则：

$$
Q_t^{signed}=\sum_i(c_i+s_i^+-r_i^{served}),\qquad
Q_t^{net,+}=[Q_t^{signed}]_+,\qquad
Q_t^{gross,+}=\sum_i[c_i+s_i^+-r_i^{served}]_+.
$$

$Q^{gross,+}$ 可以大于 $Q^{net,+}$，也不等于 PG 接口减载量；后者还受 DER、充放电和弃电改变影响。需要与相同外生轨迹及明确无 DR 基准策略配对比较 PG 净负荷。证据 C04。

**未闭合 U03**：由于式（33）使用实际回补，未交付回补会扩大核验正响应。固定消费者动作、减少未到期回补可提高 $G^{DR}$，见 C05。必须选择：把已请求却未交付的回补作为不得获奖的缺口扣除；或把此效应明确视作允许采购的延后服务并另外计价。后者需要论证，不能把它称为消费者新产生的主动削减。

**未闭合 U04**：同一物理 DR 从未付费变为付费，会使 $G^{DR}$ 增加，但物理效果可以不变，见 C06。若目标是可验证的系统减载，应让物理效果指标和购买服务的价值分别定义；若保留付费数量的二次效用，只能称 UC 的研究偏好/服务效用，并单独检验它是否改善真实能源指标。

### 18.4 平方价值的参数尺度

设固定响应下可付费上限 $M=\min(\bar P^{DR,proc},Q^{ver})$。仅看式（39）：

$$
f(q)=\beta(q/P^{DR,ref})^2-T^r q\Delta t,\quad 0\le q\le M.
$$

它是凸函数，最大值在端点；在 $M>0$ 时，全额优于零额的条件是：

$$
\beta>\frac{T^r\Delta t(P^{DR,ref})^2}{M}.
$$

相等时两个端点同值。内部导数为零对应最小值，不是最优内点。这个结论只针对固定消费者响应的两项比较，不能代替重新适应后的领导者评估。

若改变时段长度或消费者数量，应同步解释 $P^{DR,ref}$ 和 $\beta$ 的尺度。每小时价值不变的分辨率对照中，不能把“元/时段”的 $\beta$ 原数复制到更多时段。纯正向二次项是偏好假设，不是已证明的物理收益曲线。

## 19. 合法动作分布、两阶段经验和数据契约

### 19.1 合法分布属于算法，合法域属于环境

消费者先采样 $c$，再在依赖 $c$ 的区间采样有符号移峰。联合密度应使用条件分布：

$$
\pi_i(a_i\mid o_i)=\pi_i(c_i\mid o_i)
\pi_i(s_i\mid o_i,c_i)\pi_i(\rho_i,b_i^u,b_i^r\mid o_i,c_i,s_i).
$$

可以作进一步条件独立假设，但不能在算概率时忽略动作相关上下界。保存每一维合法域、模式、变换前变量和对应 `log_prob`；重算新策略密度时使用同一时刻、同一已采样前置动作条件。物理动作空间密度需含变换 Jacobian；在固定同一可逆映射下使用基础变量密度时须证明概率比一致，熵不能直接冒充物理动作熵。

UC 的充电/放电/空闲是混合模式，不能把 6 个物理字段当相互独立连续量后忽略互斥。零宽维度用确定性质量点。非法输入应在提交前报错，而不是修改已采样动作；浮点残差归零必须有公开数值容差，不得借此改变实质决策。

### 19.2 观测 schema 与仍存在的公开信息缺口

Consumer 物理动作 `[5]`，名义观测 `[H+12]`；UC 物理独立动作 `[6]`，总充电为派生量。模式分布的神经网络输出维度不等于这两个物理维度。各消费者静态参数（任务容量、期限、舒适成本、报价上限等）需固定可访问；跨场景随机改变时应进入观测上下文，不能仅存在于隐式环境裁剪中。

**未闭合 U05**：相同 $P^{u,avail}$ 和备用合同上限，可能对应不同 ESS/PG 组成，从而具有不同补供能力。C13 中，普通容量同为 4 kW，ESS 供普通电时还可用 PG 补 4 kW；普通全由 PG 供应时没有该余量。名义观测不能区分两者。实际成交后的备用余量当然不能提前透露，但已公布的供给组成不是未来信息；可公开 `planned_discharge_minus_pg_charge_kw` 及变化的 PG 接口上限，用以描述结算函数，而不泄漏别人动作。是否追加这些字段应在最终 schema 中明确；本人队列问题改善不意味着局部观测已成 Markov 状态。

### 19.3 完整执行记录

每个 `transition_id=(episode_id,period_id,mechanism_version)` 保存：

| 记录 | 必要内容 |
|---|---|
| 输入 | 冻结外生样本、阶段状态摘要、原始/物理动作、密度、模式和合法域 |
| 公告 | 两池容量、各户报价区间、事件与 DR 采购量、备用合同和已公开供给信息 |
| 订单 | 净需求、两池初始量、条件式 UC 量、必要/可延期量 |
| 出清 | 两池两等级成交、同价组、边际分配、未成交原因 |
| 执行 | 基础 PG、实际 PG、ESS、弃电、实际备用与缺口 |
| 服务 | 原生服务/损失、请求/实际回补、到期未完成、任务增减与余额 |
| DR | 有符号响应、个体正响应合计、采购量、付款和二次价值 |
| 账本 | 全部真实双边支付、非现金成本、经营利润、系统价值、个体 Reward |
| 结束 | 真实终止、采样截断、终端任务/SOC处理、残差与错误诊断 |

同价分配需要明确价格精度。建议选用最小报价刻度再形成同价组，或严格相等的浮点规则并报告其稀少性；不能使用顺序相关的“近似相等”分组。离散刻度是可选修订，不假定已包含在 v7 连续动作模型中。

### 19.4 两次接口调用、一次物理折扣

`reset()` 返回 UC 观测。UC `step()` 只发布公告并返回消费者观测；消费者 `step()` 完成整期，产生全部主体奖励、下一期 UC 观测。每个主体相邻物理行动间只计一次折扣，不能因为两次接口调用重复折扣或插入虚假消费者动作。

消费者的下一观测在下一次 UC 公告后补齐；PPO bootstrap 使用对应决策阶段。真正终止返回有效终端信息；采样片段切断只终结 GAE 递推，不抹去队列、计费状态与后续价值。两个结束字典保留 `__all__`。集中 Value 和联合回放不能假定每次接口调用所有 Agent 都同时行动。

### 19.5 训练与均衡

沿用 v5 的“UC 候选—消费者重新适应—独立验证”接受流程。冻结参数的消费者仍应对新的公开公告作动作响应；固定其实际动作做比较仅是局部机制实验。外层冻结跟随者的 PPO 更新不等于领导者的完整 Stackelberg 总导数；重新适应是候选评价步骤，不构成收敛证明。

最佳响应搜索必须保持同样的信息、动作域、终端条件和物理缺供规则。小的已发现 regret 是有限搜索未发现偏离；显著正值足以否定该候选的近似均衡宣称，但不能用训练曲线平稳或 5000 个守恒样本替代均衡验证。

连续 PAB 排序与平局点一般不光滑；要宣称纯策略均衡存在、唯一或算法收敛，需另列足够假设并证明。有限动作、有限时域的混合策略结果也不能直接推广为本连续状态、连续动作模型的纯策略结论。

## 20. 参数、数据与可复现的最小实例

### 20.1 参数语义

| 参数 | 单位 | 来源/要求 |
|---|---|---|
| $\Delta t$ | h | 默认 0.25；所有功率到能量和价格到费用显式乘它 |
| $\bar p^{b,phys},\bar P^{back,contract}$ | kW | 物理接口与商业备用额度是两个不同上限 |
| $E^{ESS},E^{min},E^{max}$ | kWh | 不把 kWh 能量变量写成无量纲 SOC；如使用 SOC 比例需除容量 |
| $\eta_c,\eta_o$ | 无量纲 | $0<\eta_c,\eta_o\le1$；v7 基准自放电取零 |
| $\bar B_i,H_i$ | kWh、时段 | 任务容量、截止延迟；$H\ge\max_iH_i$ |
| $T^u,T^d,b^u,b^r,c^{PG},h,a,T^r$ | 元/kWh | 需有来源或明确合成标记，底价不得超过上限 |
| $P^{DR,ref}$ | kW | 固定外生正值；不能由 UC 动作修改 |
| $\beta^{DR}$ | 元/时段 | 研究效用系数，需标定/敏感性，不能称外部转账 |
| $\kappa_i,\lambda^{native},\lambda^{due},\lambda^{term}$ | 元/kWh | 非现金效用与实际赔付分别列示 |
| $\lambda^{wait},\lambda^{over}$ | 元/(kWh·h) | 对各自期末未完成余额积分计费 |
| $\varepsilon_p$ | kW | 正功率尺度；不以无量纲极小数代替零负荷处理 |

PG 价格若允许负值，要重新检验弃电、充电、补供的经济动机；本轮反例和随机核验使用非负价格。配置拒绝 NaN/Inf、负容量、报价区间倒置、未知机制版本、时间不对齐和非法初始任务。

### 20.2 第 12 节算例的缺省值补齐

这是一个合成单时段核验快照，不是论文数据，也不是完整 episode。取 $H_i=4$、$\bar B_i=4$ kWh，DR 能力至少分别为 2、1、1 kW；前两户初始队列为空，第三户队列为 0.25 kWh、在未来时段到期。给定负荷与动作完全采用第 12 节。

为使报价区间合法，可取 UC 底价 0.40、DER 底价 0.30、两池报价上限 1.50 元/kWh。本例 ESS 初始能量可设 10 kWh、范围 0—20 kWh、效率 0.95，充放电动作均为零。事件开启，$\bar P^{DR,proc}\ge2$ kW。所有未列出的成本、现金赔付在该算例中取零，故它不能证明完整 episode 的服务责任或现金盈利。

完整研究配置还须给出：一整段负荷/DER/价格/DR 能力序列、计费分段、初始状态、事件日历、服务与终端代价、C03 不可行分支处理、SOC 终端口径、训练/验证/测试划分。缺少这些字段时只允许运行明示的局部算例，不把其结果称为模型训练验收。

## 21. 原子执行、实现差异和验证证据

### 21.1 先生成完整候选，再提交

所有市场、物理、服务、队列、费用、终端价值和 Reward 先在候选状态上计算；任何失败都不提交时间、SOC、任务或付款。校验通过后以唯一键原子提交。相同键与相同输入返回已有记录；相同键与不同输入拒绝，不能二次支付。

快照保存阶段、待处理公告、原始动作、任务、月账、ESS、随机数状态、已提交键、数据和配置哈希。恢复时校验机制与观测 schema；v5 与 v7 的 checkpoint 不可仅凭维度相似直接混用。

### 21.2 当前仓库仍是 v5

本轮阅读现有 `observation.py`、`dr.py`、`settlement.py` 可确认：消费者仍为十字段、四动作映射；本地 DR 仍用 v5 可行性映射；备用仍按 PG 成本加外部费结算。因此，本文五动作、同价备用、优先级、到期桶、二次 DR 价值尚未进入生产环境。已有 v5 测试的历史通过数量不是 v7 的通过证据。

### 21.3 本轮实际执行的核验

运行独立脚本：

```powershell
.\.venv\Scripts\python.exe .\environments\sn_energy_market\docs\v7\verify_v7_mechanisms.py
```

脚本按 v7 正文另写最小双池/备用/服务/支付计算，未复用 v5 环境，包含 14 个确定性审查情景和 5000 个固定种子的随机静态样本。它检查功率、请求分解、队列余额和现金转移一致性，也主动复现不良激励。

**“所有断言通过”表示反例或修复确实被复现，不表示所有机制正确。** 该脚本不是新环境、不是训练器，也未完成全空间最佳响应或 SNE 证明。具体数字和哈希见 `verification_results.json`；详见独立审查报告。

## 22. 需要定稿的最少决策与本次审查结论

以下问题直接影响模型定义，不能靠训练参数补救：

| 决策 | 当前候选规则的问题 | 最少需要明确的内容 |
|---|---|---|
| D1 期限失败 | 缺供后可到达空动作域 | 选择显式软期限违约分支，或证明硬期限可持续可行 |
| D2 DR 价值 | 正响应合计/付费标签不等于系统净减载；失败回补可增奖 | 选择奖励依据、失败回补扣除及净负荷评估口径 |
| D3 UC 竞价 | 同价备用仍可让消费者降低报价、保留供电 | 接受并研究该均衡退化，或改变备用收费/资格；不能仅宣称已解决 |
| D4 终端 | 移峰与储能存量有跨期机会成本 | 规定连续运行、首尾约束或终端价值，保持各算法一致 |
| D5 公共信息 | 普通容量无法区分 ESS/PG 来源 | 明确需要公开的计划组成和 Actor schema |

无需增加新的逐户签约流程就能处理 D1、D2、D4、D5。D3 是市场制度取舍，有限备用和服务优先级本身没有普遍消除降价依赖备用的激励。

本模型在正常合法状态下能构造守恒的单时段账本；第 12 节算例计算正确。同价备用消除了固定报价的旧渠道套利，观测新增队列也解决了本人欠账隐藏的主要问题。但空动作域、DR 奖励的物理归因、低报价备用偏离和终端存量边界仍需定稿。完成这些决策后，才能把本整合稿升级为可执行的 v7 环境规格，并进行完整轨迹、偏离和学习评估。
