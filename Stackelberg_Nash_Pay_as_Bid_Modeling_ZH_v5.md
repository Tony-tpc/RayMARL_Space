# 基于 Stackelberg–Nash 博弈与 Pay-as-Bid 出清的需求响应建模与工程架构

> **版本**：v5.0，2026-10-09
> **机制状态**：作者已确认 Q1=A、Q2=A、Q3=B、Q4=A、Q5=B；本文据此统一公式、数据契约与运行顺序。  
> **核验范围**：附带纯函数参考实现及机制/账本测试；完整能源环境、FIFO期限队列和MARL训练按工程章节继续实现。  
> **性质**：论文建模方案与工程实现设计文档  
> **基础文献**：Nie 等，*Asynchronous Multi-Agent Reinforcement Learning-Based Framework for Bi-Level Noncooperative Game-Theoretic Demand Response*，IEEE Transactions on Smart Grid，2024，DOI：[10.1109/TSG.2024.3417535](https://doi.org/10.1109/TSG.2024.3417535)。  
> **工程参考**：[MARLEM](https://github.com/salazarna/marlem)；[MAAC](https://github.com/shariqiqbal2810/MAAC)。

## 修订记录与阅读约定

本文以原论文的 UC–Consumers Stackelberg 博弈、Consumers–Consumers Nash 博弈、抽象 DR 负荷与 UC/DER/ESS 物理框架为起点。本文的 Pay-as-Bid 定价、双资源分配规则、10 维消费者观测、独立 Actor、集中 Critic 及扁平化工程架构均为**拟采用的扩展设计**。

截至本版保持的主要设计如下：

1. 将消费者基础网络固定为**独立 Actor**：每个消费者输入自己的 10 维 Observation，直接输出自己的四维策略动作。
2. 区分神经网络隐藏特征、消费者战略动作和市场成交结果。市场根据全部消费者报价生成 `p_i^u` 与 `p_i^r`。
3. 保留原论文 Fig. 4 的共享 Encoder＋个体 Decoder 作为比较对象，并单独说明其聚合输入、聚合动作及一致性问题。
4. 将系统模型、接口契约、单时段运行顺序、RLlib 适配、训练流程和测试验收写进同一份文档。
5. 明确双层训练的内层 Nash 响应、外层 UC 更新及重新响应检验；给出上层近似收敛的可实施判据。
6. 消费者成交不足按“市场未成交 → PG补供 → 原生服务与历史任务分解”记账，并与主动DR及采购补偿分别核算。
7. 补充针对 SNE 强化学习、跟随者均衡响应和双时间尺度学习的论文及开源代码索引。

### v5 的确认记录与阅读方式

本版按作者最新选择固定以下五项机制。其余主体结构保持为一个 UC、多个独立消费者 Actor、两池 Pay-as-Bid、集中训练与分散执行，以及抽象削减/移峰负荷。

| 项目 | 已选分支 | 本版实际规则 |
|---|---|---|
| Q1 | A | 普通成交偏低时下调 ESS 放电；备用阶段使用 PG |
| Q2 | A | 备用账单按可核验采购与外部服务成本回收；UC另承担调用代价 |
| Q3 | **B** | \(p_t^m\) 表示本时段**可获得补偿的 DR 采购功率上限**；各消费者的物理 DR 由本地能力决定 |
| Q4 | A | 实际供电先满足当前原生需求，再回补移峰任务；余额按真实完成量更新 |
| Q5 | B | 预定 DR 事件内，对核验的净正响应支付激励；回补电量进入实际购电账单 |

**选项的本版实现约定**：Q3 的补偿份额按核验正响应比例分配；Q4 的历史任务按先到先服务（FIFO）回补；Q5 使用外生原始负荷作为核验基线，并在实际服务分解之后计算净正响应。上述三个计算约定在正文和配置中显式给出，后续可作为机制消融项。

本版的执行顺序为：**UC预调度 → 消费者本地DR与报价 → 两池PAB → 普通成交供给调整 → PG补供 → 实际负荷服务分解 → 事件DR核验及采购 → 结算 → 原子提交状态**。

原论文内容按下表定位。本版的选项、比例采购规则、成本回收账本及执行推导属于本项目建模设计。文献与训练方式沿用上一版索引；本轮更新依据为作者的五项确认和相应代数核验。

### 原文依据索引

| 内容 | 原论文位置 | 本文处理 |
|---|---|---|
| Stackelberg–Nash 双层问题 | §II-A，式 (1)–(4)，印刷页 5625 | 保留博弈组织形式 |
| UC 的资源与收益 | §II-B，式 (5)–(20)，印刷页 5625–5626 | 保留资源类别，重整“计划—成交—执行”口径 |
| 消费者购电、DR 和成本 | §II-C，式 (21)–(31)，印刷页 5626 | 保留 DR 抽象；成交功率由市场给出；按 Q3 将 \(p_t^m\) 改为激励采购上限 |
| POMG 的 S/O/A/R | §III-A，印刷页 5627 | 修改消费者 O 与 A，保持环境动态 |
| Shared Encoder / Individual Decoder | §III-D，Fig. 4，印刷页 5629–5630 | 审核并替换消费者 Actor 的执行路径 |
| 原算法与收敛分析 | Algorithm 1、§III-E，印刷页 5629–5631 | 作为基线文献；拟采用分层训练并单独验证均衡 |

---

# 第一部分　系统建模

## 1. 研究对象：UC 如何供电，消费者如何竞争

考虑一个含有 1 个 UC 和 \(N\) 个消费者的社区。UC 统一管理外部电网购电接口、DER 发电和共享 ESS。每个消费者具有自己的原始负荷及可削减、可转移的 DR 能力。

在时段 \(t\)，UC 首先公布本轮 DR 激励采购上限、可供市场购买的 UC 电力容量和 DER 电力容量，以及适用的交易价格信息。各消费者利用自己的用电状态和公共信息做出 DR 决策，并向两类供电资源分别报价。市场按 Pay-as-Bid 规则确定每个消费者从 UC、DER 获得的实际功率。实际成交先进入备用供电与缺供处理，再由实际服务量、DR 结果和结算账本构成消费者成本与 UC 收益，最后更新下一时段的系统状态。

**本文第一版的固定分工如下：**

| 参与者 / 模块 | 输入 | 决定或计算的量 |
|---|---|---|
| UC Actor | UC 供给侧观测 | PG/DER/ESS 计划及 \(p_t^m\) |
| 消费者 \(i\) 的 Actor | 自身 10 维 Observation | \(\Delta p^l_{i,t},\Delta p^s_{i,t},b^u_{i,t},b^r_{i,t}\) |
| 本地 DR 模块 | 本消费者 DR 候选动作、负荷与移峰余额 | 可行削减/移出/回补请求及净需求 \(d_{i,t}\) |
| DR 激励模块 | 实际负荷服务、事件标记、\(p_t^m\) | 核验正响应、采购份额、补偿金额 |
| Pay-as-Bid 市场 | \(d_{i,t},b^u_{i,t},b^r_{i,t}\) 与两类容量 | 成交功率 \(p^u_{i,t},p^r_{i,t}\) |
| 备用供电模块 | 市场未成交需求、PG物理余量 | 备用补供与剩余请求缺口 |
| 结算模块 | 实际交易、采购补偿、原生服务与任务状态 | 每个消费者账单与 UC 账本 |
| 物理模块 | 供给计划、市场成交、备用补供 | PG 实际购电、DER 使用、ESS 状态、下一状态 |
| Critic（训练期） | 全局状态、联合动作、成交反馈 | 各 Agent 的价值估计 |

这是**UC 先决策、消费者同时响应、市场统一分配、系统一次推进物理时间**的结构。

## 2. 时间、单位与状态

离散时段：

\[
 t=0,1,\ldots,T-1,\qquad \Delta t=0.25\ \mathrm{h}.
\]

可按原论文案例采用每天 96 个时段。功率量使用 `kW`；电量使用 `kWh`；价格使用 `元/kWh` 或统一的货币/kWh；在所有费用中明确乘以 \(\Delta t\)。

月累计 UC 购电量保留跨天状态，并包含在 UC 供电账户结算的备用电量：

\[
 E_{i,t+1}^u=E_{i,t}^u+(p_{i,t}^u+p_{i,t}^{\mathrm{back}})\Delta t.
\]

在跨月边界按月度计费规则重置。完整训练时可使用连续多日 episode，单日调试可指定月初累计量作为初始条件。

## 3. UC 资源和功率流

### 3.1 原论文保留的物理量

| 变量 | 含义 |
|---|---|
| \(p_t^b\) | UC 从 PG 购买功率 |
| \(p_t^g\) | DER 发电功率，作为外生资源输入 |
| \(p_t^a\) | DER 弃电功率 |
| \(p_t^c\) | 共享 ESS 总充电功率 |
| \(p_t^d\) | DER → ESS 的充电功率 |
| \(p_t^e\) | UC/PG → ESS 的充电功率 |
| \(p_t^o\) | ESS 放电功率 |
| \(p_t^m\) | UC 本时段可支付补偿的 DR 采购功率上限（Q3=B） |
| \(p_t^s\) | UC 体系向消费者实际销售的总功率 |

原文的总体功率平衡式 (17) 为：

\[
 \boxed{p_t^b+p_t^g+p_t^o=p_t^a+p_t^c+p_t^s.}
\]

本文将市场成交功率定义为：

\[
 \boxed{p_t^{s,\mathrm{exec}}=\sum_{i=1}^{N}(p_{i,t}^{u}+p_{i,t}^{r}+p_{i,t}^{\mathrm{back}}).}
\]

这使供给侧与 Pay-as-Bid 分配具有清晰的连接点。ESS 的充电分量满足：

\[
 p_t^c=p_t^d+p_t^e.
\]

### 3.2 UC 计划、普通成交与 PG 备用补供

UC 在报价前生成计划，出清后的确定性程序落实最终功率。计划接口延续七字段：

\[
 a_t^u=(p_t^{b,\mathrm{plan}},p_t^{a,\mathrm{plan}},p_t^{c,\mathrm{plan}},
 p_t^{d,\mathrm{plan}},p_t^{e,\mathrm{plan}},p_t^{o,\mathrm{plan}},p_t^m).
\]

其中 \(p_t^{c,\mathrm{plan}}=p_t^{d,\mathrm{plan}}+p_t^{e,\mathrm{plan}}\)。`action.py` 将策略原始采样值映射到联合可行计划，检查 PG 进口上限、DER分流、充放电互斥和计划期末SOC。七个接口字段与策略的独立采样自由度分别记录。

**Q1=A 的执行约定**：充电及来源分量保持已通过校验的计划；ESS放电根据普通成交量作向下调整；备用调用由PG提供。初版同一物理时段内使用同一份DER出力和PG成本数据完成公告、出清和执行。

#### 3.2.1 公布普通市场资源额度

\[
 P_t^{u,\mathrm{avail}}=p_t^{b,\mathrm{plan}}+p_t^{o,\mathrm{plan}}-p_t^{e,\mathrm{plan}},
 \qquad
 P_t^{r,\mathrm{avail}}=p_t^g-p_t^{a,\mathrm{plan}}-p_t^{d,\mathrm{plan}}.
\]

两项额度均为非负。普通成交记为：

\[
 U_t=\sum_i p_{i,t}^u,\qquad D_t^r=\sum_i p_{i,t}^r.
\]

出清保证 \(U_t\le P_t^{u,\mathrm{avail}}\)、\(D_t^r\le P_t^{r,\mathrm{avail}}\)。电网物理上限 \(\overline p_t^{b,\mathrm{phys}}\) 同时约束普通和备用的实际购电。

#### 3.2.2 按普通成交量落实 ESS 与基础 PG

充电按计划执行：

\[
 p_t^{d,\mathrm{exec}}=p_t^{d,\mathrm{plan}},\quad
 p_t^{e,\mathrm{exec}}=p_t^{e,\mathrm{plan}},\quad
 p_t^{c,\mathrm{exec}}=p_t^{d,\mathrm{plan}}+p_t^{e,\mathrm{plan}}.
\]

放电采用以下确定映射：

\[
 \boxed{p_t^{o,\mathrm{exec}}
 =\min\{p_t^{o,\mathrm{plan}},\ U_t+p_t^{e,\mathrm{plan}}\}.}
\]

放电模式下由互斥约束得到 \(p_t^{e,\mathrm{plan}}=0\)，因此该式就是“计划放电与普通 UC 成交量取较小值”。充电模式下计划放电为0。

普通成交所需的基础 PG 购电为：

\[
 \boxed{g_t^{\mathrm{base}}
 =U_t+p_t^{e,\mathrm{plan}}-p_t^{o,\mathrm{exec}}
 =\max\{0,U_t+p_t^{e,\mathrm{plan}}-p_t^{o,\mathrm{plan}}\}.}
\]

由 UC 额度约束可得：

\[
 0\le g_t^{\mathrm{base}}\le p_t^{b,\mathrm{plan}}
 \le\overline p_t^{b,\mathrm{phys}}.
\]

记录下调量 \(\Delta p_t^{o,\downarrow}=p_t^{o,\mathrm{plan}}-p_t^{o,\mathrm{exec}}\)。这部分能量保留在共享电池中，并通过下一时段SOC影响后续策略。

普通 DER 余量计入实际弃电：

\[
 \boxed{p_t^{a,\mathrm{exec}}
 =p_t^g-p_t^{d,\mathrm{plan}}-D_t^r
 =p_t^{a,\mathrm{plan}}+(P_t^{r,\mathrm{avail}}-D_t^r).}
\]

#### 3.2.3 使用 PG 剩余能力补供

记备用合同功率额度 \(\overline P_t^{\mathrm{back,contract}}\)，可调用容量为：

\[
 P_t^{\mathrm{back,cap}}=
 \min\{\overline P_t^{\mathrm{back,contract}},
 \overline p_t^{b,\mathrm{phys}}-g_t^{\mathrm{base}}\}.
\]

`reserve.py` 按市场未成交需求比例分配实际备用功率。令：

\[
 B_t^{\mathrm{back}}=\sum_i p_{i,t}^{\mathrm{back}},\qquad
 p_t^{b,\mathrm{exec}}=g_t^{\mathrm{base}}+B_t^{\mathrm{back}}.
\]

因此最终电网进口满足 \(p_t^{b,\mathrm{exec}}\le\overline p_t^{b,\mathrm{phys}}\)。备用阶段使用第3.2.2节已经落实的ESS功率。

#### 3.2.4 最终功率平衡及推导

\[
 \boxed{p_t^{b,\mathrm{exec}}+p_t^g+p_t^{o,\mathrm{exec}}
 =p_t^{a,\mathrm{exec}}+p_t^{c,\mathrm{exec}}
 +U_t+D_t^r+B_t^{\mathrm{back}}.}
\]

代入基础购电、实际弃电和充电分量后，两侧都化为：

\[
 U_t+B_t^{\mathrm{back}}+p_t^{e,\mathrm{plan}}+p_t^g.
\]

`physics.py` 返回计划、实际功率、放电下调量及守恒残差。计划所满足的充放电互斥继续成立。对已经满足SOC边界的放电计划，下调放电把更多能量留在电池内；充电计划保持原来的能量校验。

#### 3.2.5 低成交算例

取 \(p_t^g=0\)、计划电池放电4 kW、普通UC成交2 kW、充电0：

\[
 p_t^{o,\mathrm{exec}}=2,\qquad
 \Delta p_t^{o,\downarrow}=2,\qquad
 g_t^{\mathrm{base}}=0.
\]

最终2 kW电池出力全部供给消费者。该规则由环境直接计算，UC策略仍在本时段开始时生成一次计划。

### 3.3 共享 ESS 的状态更新

本文拟采用常见的能量形式：

\[
E_{t+1}^{ESS}=(1-\rho)E_t^{ESS}
+\eta_c p_t^{c,\mathrm{exec}}\Delta t
-\frac{p_t^{o,\mathrm{exec}}\Delta t}{\eta_o}.
\]

其中 \(E^{ESS}\) 的单位为 kWh，\(\eta_c,\eta_o\) 是充、放电效率。代码约束为 \(E^{\min}\le E_t^{ESS}\le E^{\max}\)、充放电功率上限和同时充放电互斥。

这是本文拟采用的实现表达；原论文的 ESS 动态与符号约定参见其式 (19)、(20)。

## 4. 消费者 DR、移峰请求与真实回补

本节把“消费者提出的动作”“进入市场的净需求”“真正完成的服务”依次定义。原始负荷 \(p_{i,t}^*\) 来源于冻结的数据序列；正移峰把当前原生用电移到后续时段，负移峰请求回补已经积累的任务。

### 4.1 本地动作映射与净需求

策略输出仍为削减请求 \(\Delta p_{i,t}^{l}\) 和带符号的移峰请求 \(\Delta p_{i,t}^{s}\)。经过本地可行性映射，定义：

\[
 c_{i,t}=\Delta p_{i,t}^{l,\mathrm{act}},\quad
 s_{i,t}^{+}=\max(\Delta p_{i,t}^{s,\mathrm{req}},0),\quad
 r_{i,t}^{\mathrm{req}}=\max(-\Delta p_{i,t}^{s,\mathrm{req}},0).
\]

本节 \(r^{\mathrm{req}}\) 表示回补功率，训练奖励使用无该上标的 \(r_{i,t}\)。代码分别使用 `repay_requested_kw` 和 `reward`。

本地可行域取：

\[
 \begin{aligned}
 &0\le c_{i,t}\le\min(p_{i,t}^*,\Delta p_{i,t}^{\max}),\\
 &0\le s_{i,t}^{+}\le
   \min\{p_{i,t}^*-c_{i,t},\Delta p_{i,t}^{\max}-c_{i,t},
   (\overline B_i-B_{i,t})/\Delta t\},\\
 &0\le r_{i,t}^{\mathrm{req}}\le
   \min\{\Delta p_{i,t}^{\max},B_{i,t}/\Delta t\},
 \qquad s_{i,t}^{+}r_{i,t}^{\mathrm{req}}=0.
 \end{aligned}
\]

\(B_{i,t}\) 是待回补电量，\(\overline B_i\) 是本地队列能量上限；两者单位kWh。每个新移峰任务携带原生时段和约定完成期限。`dr.py` 只使用本消费者数据完成上述映射。

当前仍需服务的原生需求、总市场需求分别是：

\[
 \boxed{n_{i,t}=p_{i,t}^*-c_{i,t}-s_{i,t}^{+},\qquad
 d_{i,t}=n_{i,t}+r_{i,t}^{\mathrm{req}}.}
\]

这也等于 \(p_{i,t}^*-\Delta p_{i,t}^{l,\mathrm{act}}-\Delta p_{i,t}^{s,\mathrm{req}}\)。新一轮原始负荷继续读取数据序列，先前移峰量以 \(B_{i,t}\) 和任务队列单独保存。

### 4.2 Q3=B：物理 DR 与激励采购分别处理

本版将 \(p_t^m\) 定义为 **DR 激励采购功率上限**，单位kW。每个消费者的物理动作由第4.1节的本地范围决定。激励模块在实际服务完成后计算采购量 \(q_{i,t}^{\mathrm{paid}}\)，并满足：

\[
 \boxed{\sum_iq_{i,t}^{\mathrm{paid}}\le p_t^m.}
\]

例如两位消费者各自执行3 kW削减，合计物理削减6 kW，UC采购上限4 kW。实际需求按6 kW削减计算，按比例补偿时每人获得2 kW对应的补偿。其余物理调整仍影响实际购电需求与本地成本。

### 4.3 Q4=A：先满足当前原生需求，再回补历史任务

市场及备用结束后，消费者实际得到 \(p_{i,t}^{\mathrm{served}}\) kW。按照确定服务顺序计算：

\[
 \begin{aligned}
 p_{i,t}^{\mathrm{native,served}}
 &=\min\{p_{i,t}^{\mathrm{served}},n_{i,t}\},\\
 r_{i,t}^{\mathrm{act}}
 &=\min\{r_{i,t}^{\mathrm{req}},
     [p_{i,t}^{\mathrm{served}}-n_{i,t}]_+\},\\
 \ell_{i,t}^{\mathrm{native}}
 &=n_{i,t}-p_{i,t}^{\mathrm{native,served}},\\
 q_{i,t}^{\mathrm{repay,unfilled}}
 &=r_{i,t}^{\mathrm{req}}-r_{i,t}^{\mathrm{act}}.
 \end{aligned}
\]

实际服务满足：

\[
 p_{i,t}^{\mathrm{served}}=p_{i,t}^{\mathrm{native,served}}+r_{i,t}^{\mathrm{act}}.
\]

\(\ell^{\mathrm{native}}\) 记录当前原生服务的实际损失；\(q^{\mathrm{repay,unfilled}}\) 记录本次回补请求中继续挂账的部分。总的当期请求缺口为：

\[
 \boxed{\ell_{i,t}^{\mathrm{req}}
 =d_{i,t}-p_{i,t}^{\mathrm{served}}
 =\ell_{i,t}^{\mathrm{native}}+q_{i,t}^{\mathrm{repay,unfilled}}.}
\]

### 4.4 余额、实际 DR 与期限

余额按实际完成量更新：

\[
 \boxed{B_{i,t+1}=B_{i,t}+s_{i,t}^{+}\Delta t-r_{i,t}^{\mathrm{act}}\Delta t.}
\]

相应的实际移峰及主动 DR 为：

\[
 \Delta p_{i,t}^{s,\mathrm{act}}=s_{i,t}^{+}-r_{i,t}^{\mathrm{act}},\qquad
 \delta_{i,t}^{\mathrm{act}}=c_{i,t}+s_{i,t}^{+}-r_{i,t}^{\mathrm{act}}.
\]

原生负荷的执行恒等式为：

\[
 \boxed{p_{i,t}^*=c_{i,t}+\Delta p_{i,t}^{s,\mathrm{act}}
 +p_{i,t}^{\mathrm{served}}+\ell_{i,t}^{\mathrm{native}}.}
\]

在时域内求和，移峰账本满足：

\[
 \sum_t s_{i,t}^{+}\Delta t
 =\sum_t r_{i,t}^{\mathrm{act}}\Delta t+B_{i,T}-B_{i,0}.
\]

代码使用FIFO队列保存每笔移出能量及期限。实际回补从最早待办任务开始扣减，未完成量保留原期限。达到期限后仍待办的电量标记为 `overdue_kwh`，产生约定的延迟成本，后续服务继续减少该任务余额。

日边界和PPO采样截断保留队列。连续仿真把余额带入后续时段；独立场景真实结束时，将余额保留在最终状态、记录 `terminal_pending_kwh` 并计入终端服务责任成本。该余额与实际已回补电量分别存档。

### 4.5 回补不足算例

初始待办量2 kWh，\(\Delta t=0.25\) h，当前原生需求4 kW，回补请求8 kW。因此本轮总需求12 kW。若实际服务8 kW：

\[
 p^{\mathrm{native,served}}=4,\quad r^{\mathrm{act}}=4,\quad
 q^{\mathrm{repay,unfilled}}=4,\quad\ell^{\mathrm{native}}=0.
\]

本轮真正完成回补 \(4\times0.25=1\) kWh，余额为1 kWh，后续时段继续服务这1 kWh。

---

# 第二部分　POMG、行动与市场机制

## 5. Consumer Agent 的 10 维 Observation

将每个 Consumer Actor 的输入**固定为**：

\[
\boxed{
 o_{i,t}=
 \left[
 t,\ p_{i,t}^*,\ E_{i,t}^{u},\ \Delta p_{i,t}^{\max},\ T_t^u,
 T_t^d,\ T_t^r,\ p_t^m,\ P_t^{u,\mathrm{avail}},\ P_t^{r,\mathrm{avail}}
 \right].
}
\]

| 顺序 | 字段名 | 数学变量 | 数据来源与作用 |
|---:|---|---|---|
| 1 | `time_index` | \(t\) | 时间序列，识别峰平谷 |
| 2 | `original_load_kw` | \(p^*_{i,t}\) | 自身负荷数据，确定当前需求 |
| 3 | `monthly_uc_energy_kwh` | \(E^u_{i,t}\) | 已完成结算的 UC 购电量，决定阶梯位置 |
| 4 | `max_dr_kw` | \(\Delta p^{\max}_{i,t}\) | 自身 DR 能力和动作边界 |
| 5 | `uc_tariff` | \(T_t^u\) | 适用于当前消费者的 TUTT 价格 |
| 6 | `der_tariff` | \(T_t^d\) | 当前 DER 参考价 |
| 7 | `dr_incentive` | \(T_t^r\) | 当前 DR 激励价 |
| 8 | `dr_procurement_cap_kw` | \(p_t^m\) | 本时段可获得补偿的 DR 采购功率上限 |
| 9 | `uc_available_kw` | \(P_t^{u,\mathrm{avail}}\) | UC 侧公开可售额度 |
| 10 | `der_available_kw` | \(P_t^{r,\mathrm{avail}}\) | DER 侧公开可售额度 |

这里沿用用户约定的 \(T_t^u\) 记号；在具体计算中由消费者 \(i\) 的月累计购电量求得：

\[
T_t^u\big|_i = T^u(t,E_{i,t}^{u}).
\]

`observation.py` 对每个消费者单独调用 `build_consumer_observation`，返回严格按表中顺序排列的 `float32[10]`。UC 行动后形成的公共容量和 DR 信号，进入本时段消费者 Observation。备用成本回收公式、事件日历和补偿分配规则作为事先公布的合同配置。当前备用实际价格、动态PG余量、移峰队列等保存在全局状态与执行记录；消费者Actor仍取上述10个字段，保持部分可观测策略。事件外将本轮DR激励价及有效采购上限编码为0；相关信息扩展列在第25节。

## 6. Consumer Actor 的四维 Action

\[
\boxed{
 a_{i,t}^c=
 [\Delta p_{i,t}^l,\Delta p_{i,t}^s,b_{i,t}^u,b_{i,t}^r].
}
\]

| 字段 | 作用 |
|---|---|
| `dr_reduce_kw` | Actor提出的削减负荷，执行值由DR规则确定 |
| `dr_shift_kw` | Actor提出的移峰功率：正值移出当前任务，负值请求回补历史任务 |
| `uc_bid_price` | 为获得 UC 电力提交的单位价格 |
| `der_bid_price` | 为获得 DER 电力提交的单位价格 |

消费者根据经 DR 调整后的需求 \(d_{i,t}\) 参与市场。本文初版采用**单段报价**，每个消费者针对 UC 和 DER 各提交一个报价价格，共享一个净需求量。双方资源的实际成交功率交给市场决定。

当前报价的允许区间设为：

\[
T_t^u\big|_i\le b_{i,t}^u\le \overline b^u,
\qquad
T_t^d\le b_{i,t}^r\le\overline b^r.
\]

在这个约定下，TUTT 和 DER 参考价用作最低可接受报价，成交电力按消费者最终申报的中标报价支付。消费者报价直接影响分配优先级及实际支付成本。

## 7. Pay-as-Bid 市场出清

本方案将市场组织成两个供给资源池：DER 和 UC。使用固定的 **DER 优先、UC 补足** 清算次序，市场在每个池中根据买方报价从高到低分配。报价相同时采用可复现的 `consumer_id` 顺序，边际订单允许部分成交。

### 7.1 DER 市场

输入：\(\{d_{i,t},b_{i,t}^r\}_i\) 与 \(P_t^{r,\mathrm{avail}}\)。

按 \(b_{i,t}^r\) 降序逐一分配，得到 \(p_{i,t}^r\)：

\[
0\le p_{i,t}^r\le d_{i,t},\qquad
\sum_i p_{i,t}^r\le P_t^{r,\mathrm{avail}}.
\]

### 7.2 UC 市场

消费者剩余需求：

\[
\boxed{d_{i,t}^{\mathrm{rem}}=d_{i,t}-p_{i,t}^r.}
\]

按 \(b_{i,t}^u\) 降序分配 UC 资源，得到 \(p_{i,t}^u\)：

\[
0\le p_{i,t}^u\le d_{i,t}^{\mathrm{rem}},\qquad
\sum_i p_{i,t}^u\le P_t^{u,\mathrm{avail}}.
\]

市场成交后，先计算每位消费者的**市场未成交功率**：

\[
\boxed{q_{i,t}^{\mathrm{unfilled}}=d_{i,t}-p_{i,t}^u-p_{i,t}^r\ge0.}
\]

随后由 `reserve.py` 进行备用补供，获得 \(p_{i,t}^{\mathrm{back}}\)，本次请求的剩余缺口记为 \(\ell_{i,t}^{\mathrm{req}}\)。完整需求账本为：

\[
\boxed{d_{i,t}=p_{i,t}^u+p_{i,t}^r+p_{i,t}^{\mathrm{back}}+\ell_{i,t}^{\mathrm{req}}.}
\]

其中 \(p_{i,t}^u,p_{i,t}^r\) 是 Pay-as-Bid **市场成交量**，\(p_{i,t}^{\mathrm{back}}\) 是按照公开备用供电规则形成的**市场外补供量**，\(\ell_{i,t}^{\mathrm{req}}\) 是**本次请求缺口**。服务模块随后将其分成原生损失和待回补请求两项；各量在订单、结算、物理执行和训练记录中分别存储。

### 7.3 Pay-as-Bid 支付

市场清算量与支付金额分别计算：

\[
 C_{i,t}^{u,\mathrm{PAB}}=b_{i,t}^u\,p_{i,t}^u\Delta t,
 \qquad
 C_{i,t}^{r,\mathrm{PAB}}=b_{i,t}^r\,p_{i,t}^r\Delta t,
\]

\[
\boxed{C_{i,t}^{\mathrm{PAB}}= C_{i,t}^{u,\mathrm{PAB}}+C_{i,t}^{r,\mathrm{PAB}}.}
\]

`market.py` 实现出清数量与每一笔 trade 的归属；`settlement.py` 根据中标订单金额计算账单。因此还可以在后续对比实验中更换 Uniform Price，而保持容量分配接口不变。

### 7.4 数值示例

本例验算订单分配与付款；对应UC物理计划另按第3.2节验证。

设三个消费者经 DR 后净需求为 \((5,4,3)\) kW，DER 容量 7 kW，UC 容量 8 kW。DER 报价依次为 \((0.45,0.40,0.32)\) 元/kWh。

DER 按报价高低分配：消费者 A 得 5 kW、B 得 2 kW、C 得 0 kW。剩余需求 \((0,2,3)\) kW 由 UC 成交；设 B、C 的 UC 报价分别为 \(0.66\) 和 \(0.58\) 元/kWh。

| 消费者 | 净需求 kW | DER 成交 kW | UC 成交 kW | 本时段实际总成交 kW |
|---|---:|---:|---:|---:|
| A | 5 | 5 | 0 | 5 |
| B | 4 | 2 | 2 | 4 |
| C | 3 | 0 | 3 | 3 |
| **合计** | **12** | **7** | **5** | **12** |

当 \(\Delta t=0.25\) h，按 PAB 计算：A 的支付为 \(0.45\times5\times0.25=0.5625\) 元；B 的支付为 \((0.40\times2+0.66\times2)\times0.25=0.53\) 元；C 的支付为 \(0.58\times3\times0.25=0.435\) 元。该时段总支付为 \(1.5275\) 元。


### 7.5 市场成交量不足时的履约机制

#### 7.5.1 “报价量”在当前模型中的来源

消费者 Actor 输出的四维动作是：

\[
 a_{i,t}^c=(\Delta p_{i,t}^{l},\Delta p_{i,t}^{s},b_{i,t}^{u},b_{i,t}^{r}).
\]

消费者本时段参与出清的需求由负荷与 DR 计算得到：

\[
\boxed{d_{i,t}=p_{i,t}^*-\Delta p_{i,t}^{l}-\Delta p_{i,t}^{s}.}
\]

因此 \(d_{i,t}\) 是本轮请求满足的**净购电功率**，两个 \(b\) 分别是 UC、DER 资源的单位报价。市场决定的 \(p_{i,t}^{u},p_{i,t}^{r}\) 为**实际中标功率**。

#### 7.5.2 出清不足后的三段账本

**第一段：市场成交。** 两池PAB返回 \(p^u,p^r\)，按中标报价逐笔计费，市场未成交量为：

\[
 q_{i,t}^{\mathrm{unfilled}}=d_{i,t}-p_{i,t}^u-p_{i,t}^r.
\]

**第二段：PG补供。** 根据第3.2节落实ESS和基础PG，再计算备用可用额度 \(P_t^{\mathrm{back,cap}}\)。令 \(Q_t=\sum_iq_{i,t}^{\mathrm{unfilled}}\)：

\[
 p_{i,t}^{\mathrm{back}}=
 \begin{cases}
 q_{i,t}^{\mathrm{unfilled}}\min(1,P_t^{\mathrm{back,cap}}/Q_t),&Q_t>0,\\
 0,&Q_t=0.
 \end{cases}
\]

**第三段：服务分解。** 总服务和请求缺口为：

\[
 p_{i,t}^{\mathrm{served}}=p_{i,t}^u+p_{i,t}^r+p_{i,t}^{\mathrm{back}},\qquad
 \ell_{i,t}^{\mathrm{req}}=q_{i,t}^{\mathrm{unfilled}}-p_{i,t}^{\mathrm{back}}.
\]

第4.3节据此计算 \(\ell^{\mathrm{native}}\)、\(r^{\mathrm{act}}\) 和 \(q^{\mathrm{repay,unfilled}}\)，再更新真实任务余额。旧日志 `lost_load_kw` 升级为语义明确的 `unserved_request_kw`、`native_lost_kw`、`repay_unfilled_kw`。

#### 7.5.3 Q2=A：备用按真实成本回收

设 \(c_t^{PG}=T_t^b+\varepsilon_t\) 为本时段 PG 实际单位采购成本，\(h_t^{\mathrm{back}}\) 为可核验的外部备用服务单价。基础采购场景采用 \(c_t^{PG}\ge0\)、\(h_t^{\mathrm{back}}\ge0\) 的当期线性成本，得到：

\[
 \boxed{T_t^{\mathrm{back}}=c_t^{PG}+h_t^{\mathrm{back}}.}
\]

消费者补供账单为：

\[
 C_{i,t}^{\mathrm{back}}=(c_t^{PG}+h_t^{\mathrm{back}})p_{i,t}^{\mathrm{back}}\Delta t.
\]

这笔款用于支付相同备用电量的 PG 成本与外部服务费用。UC另承担自己的调用代价：

\[
 C_t^{\mathrm{activation}}=a_t^{\mathrm{back}}B_t^{\mathrm{back}}\Delta t,
 \qquad a_t^{\mathrm{back}}>0.
\]

\(h_t^{\mathrm{back}}\) 与 \(a_t^{\mathrm{back}}\) 在账本中分栏：前者属于支付给外部服务方的成本，后者属于 UC 自担的操作/可靠性代价。基准实验固定采购和服务数据，再由该公式生成备用价。

#### 7.5.4 两消费者补供算例

两位消费者经主动DR后的需求均为8 kW，本例回补请求均为0。DER为3 kW，UC普通额度为7 kW，ESS功率为0，PG物理上限为9 kW。A的两类报价均较高，普通成交可为：

| 消费者 | 原生净需求 | DER成交 | UC成交 | 未成交请求 |
|---|---:|---:|---:|---:|
| A | 8 | 3 | 5 | 0 |
| B | 8 | 0 | 2 | 6 |
| 合计 | 16 | 3 | 7 | 6 |

单位均为kW。基础PG为7 kW，剩余PG能力为2 kW，备用2 kW全部供给B。最终：

\[
 p_B^{\mathrm{served}}=2+2=4,\qquad
 \ell_B^{\mathrm{native}}=4,\qquad
 p_t^{b,\mathrm{exec}}=9.
\]

取 \(c_t^{PG}=0.50\)、\(h_t^{\mathrm{back}}=0.10\)、\(a_t^{\mathrm{back}}=0.20\) 元/kWh以及 \(\Delta t=0.25\) h：

\[
 C_B^{\mathrm{back}}=0.60\times2\times0.25=0.30\text{ 元},
\]

其中增量PG成本0.25元、外部服务0.05元。UC自担调用代价为0.10元。B的普通2 kW仍按本人的UC中标报价支付。

#### 7.5.5 富余、紧缺与任务服务

资源充足时，两池PAB覆盖净需求，备用和请求缺口均为0。资源紧缺时，按PG额度补供；最终供电由固定的原生优先顺序分配到当前负荷和历史任务。

每一步都分别记录**市场未成交、备用供电、原生失供、回补等待**。每个消费者根据真实购电、主动调整、补偿以及服务代价得到奖励，进入下一时段继续决策。

## 8. 事件DR采购、结算与奖励

### 8.1 Q5=B：核验事件内的净正响应

事件标记 \(\chi_t\in\{0,1\}\) 由实验配置中的日历、基线和合同预先确定。\(p_{i,t}^*\) 为在消费者报价前已经冻结的原始负荷。实际服务分解后，核验主动净正响应：

\[
 \boxed{q_{i,t}^{\mathrm{ver}}
 =\chi_t\,[c_{i,t}+s_{i,t}^{+}-r_{i,t}^{\mathrm{act}}]_+
 =\chi_t\,[p_{i,t}^*-(p_{i,t}^{\mathrm{served}}+\ell_{i,t}^{\mathrm{native}})]_+.}
\]

第二个表达式把原生失供补回到服务基准中，得到与主动削减/移峰相同的核验口径。纯原生缺供的核验量为0；纯回补的核验量为0；事件内同时削减和回补时使用净正响应。

每一笔移出任务绑定原始时段；历史回补服务从对应队列扣除，并用来抵减本时段主动净响应。激励账本使用 `(episode_id, period_id, consumer_id, event_id)` 唯一键，状态提交时登记。环境重放和训练中的轨迹读取只读取既有记录。

### 8.2 Q3=B：在采购上限内分配补偿

令 \(Q_t^{\mathrm{ver}}=\sum_iq_{i,t}^{\mathrm{ver}}\)。本版采用**核验量比例分配**：

\[
 \boxed{q_{i,t}^{\mathrm{paid}}=
 \begin{cases}
 q_{i,t}^{\mathrm{ver}}\min\left(1,\dfrac{p_t^m}{Q_t^{\mathrm{ver}}}\right),
     &Q_t^{\mathrm{ver}}>0,\\
 0,&Q_t^{\mathrm{ver}}=0.
 \end{cases}}
\]

因此：

\[
 0\le q_{i,t}^{\mathrm{paid}}\le q_{i,t}^{\mathrm{ver}},\qquad
 \sum_iq_{i,t}^{\mathrm{paid}}=\min(p_t^m,Q_t^{\mathrm{ver}}).
\]

补偿金额为：

\[
 I_{i,t}^{DR}=T_t^r q_{i,t}^{\mathrm{paid}}\Delta t.
\]

事件外 \(\chi_t=0\)，有效 \(T_t^r=0\)、\(p_t^m=0\)，所有DR补偿为0。物理DR继续按本地约束执行。UC的 \(p_t^m\) 自由度由当前事件的允许采购上限约束，消费者Actor在观察到它以后自行选择物理调整。

UC采购动作的有限范围正式设为：

\[
 0\le p_t^m\le\chi_t\overline P^{DR,\mathrm{proc}},
\]

其中 \(\overline P^{DR,\mathrm{proc}}\) 为预定的采购功率额度，记录在配置中。每个时段的实际付款仍按核验量和比例分配计算。

**例**：两位消费者各执行3 kW削减、事件有效、\(p_t^m=4\) kW、\(T_t^r=0.50\) 元/kWh、\(\Delta t=0.25\) h。总物理削减6 kW；每位消费者的核验量3 kW、获补偿量2 kW、补偿0.25元。若 \(p_t^m=0\)，相同物理动作仍减少6 kW需求，补偿金额为0。

### 8.3 消费者账本

实际电力支出为：

\[
 C_{i,t}^{\mathrm{energy}}
 =(b_{i,t}^up_{i,t}^u+b_{i,t}^rp_{i,t}^r)\Delta t
 +(c_t^{PG}+h_t^{\mathrm{back}})p_{i,t}^{\mathrm{back}}\Delta t.
\]

沿用抽象DR调整代价：

\[
 C_{i,t}^{\mathrm{adjust}}
 =\kappa_i\frac{(\delta_{i,t}^{\mathrm{act}})^2}
 {\max(p_{i,t}^*,\epsilon_p)}\Delta t,
 \qquad\epsilon_p>0\text{，单位kW}.
\]

当前原生服务损失、任务等待及逾期成本分别为：

\[
 \begin{aligned}
 C_{i,t}^{\mathrm{native}}&=\lambda_i^{\mathrm{native}}
                 \ell_{i,t}^{\mathrm{native}}\Delta t,\\
 C_{i,t}^{\mathrm{waiting}}&=\lambda_i^{\mathrm{waiting}}B_{i,t+1}\Delta t,\\
 C_{i,t}^{\mathrm{overdue}}&=\lambda_i^{\mathrm{overdue}}
                 B_{i,t+1}^{\mathrm{overdue}}\Delta t.
 \end{aligned}
\]

\(\lambda^{\mathrm{native}}\) 的单位为元/kWh；后两个系数为元/(kWh·h)。\(B^{\mathrm{overdue}}\) 取截至时段末已越过截止时间的待办能量。连续任务欠账持续保留，费用按等待和逾期时间累积。独立场景真实结束时另计一次 \(\lambda_i^{\mathrm{terminal}}B_{i,T}\) 的终端服务责任成本，采样截断继续使用后续状态估值。

消费者总成本和奖励为：

\[
 \boxed{C_{i,t}=C_{i,t}^{\mathrm{energy}}+C_{i,t}^{\mathrm{adjust}}
 +C_{i,t}^{\mathrm{native}}+C_{i,t}^{\mathrm{waiting}}+C_{i,t}^{\mathrm{overdue}}
 -I_{i,t}^{DR}-C_{i,t}^{\mathrm{comp}},\qquad r_{i,t}=-C_{i,t}.}
\]

终端服务责任在真正终止时单独加入最后一步。\(C^{\mathrm{comp}}\) 为实际赔付，默认值为0；启用后与UC同额支出对应。回补电量按本时段实际来源和成交/备用价格计入购电账单。

### 8.4 Q2=A：UC账本及成本回收恒等式

普通售电收入和备用收款为：

\[
 \mathrm{Rev}_t^{\mathrm{PAB}}
 =\sum_i(b_{i,t}^up_{i,t}^u+b_{i,t}^rp_{i,t}^r)\Delta t,\qquad
 \mathrm{Rev}_t^{\mathrm{back}}=(c_t^{PG}+h_t^{\mathrm{back}})B_t^{\mathrm{back}}\Delta t.
\]

UC最终奖励采用：

\[
 \boxed{\begin{aligned}
 r_t^u={}&\mathrm{Rev}_t^{\mathrm{PAB}}+\mathrm{Rev}_t^{\mathrm{back}}
 -c_t^{PG}p_t^{b,\mathrm{exec}}\Delta t
 -h_t^{\mathrm{back}}B_t^{\mathrm{back}}\Delta t\\
 &-a_t^{\mathrm{back}}B_t^{\mathrm{back}}\Delta t
 -C_t^{\mathrm{ESS}}-C_t^{\mathrm{curt}}-\sum_i I_{i,t}^{DR}
 -\sum_i C_{i,t}^{\mathrm{comp}}-C_t^{u,\mathrm{service}}.
 \end{aligned}}
\]

其中 \(C_t^{\mathrm{ESS}}\) 根据实际充放电吞吐计算，\(C_t^{\mathrm{curt}}\) 根据实际弃电计算。\(C_t^{u,\mathrm{service}}\) 由原生失供、逾期任务和真实终止时服务责任等约定项组成，指标按同一服务账本分栏。

利用 \(p_t^{b,\mathrm{exec}}=g_t^{\mathrm{base}}+B_t^{\mathrm{back}}\)，备用收款抵销对应PG增量成本及外部服务成本，得到核验用的等价形式：

\[
 \boxed{\begin{aligned}
 r_t^u={}&\mathrm{Rev}_t^{\mathrm{PAB}}
 -c_t^{PG}g_t^{\mathrm{base}}\Delta t
 -a_t^{\mathrm{back}}B_t^{\mathrm{back}}\Delta t\\
 &-C_t^{\mathrm{ESS}}-C_t^{\mathrm{curt}}
 -\sum_iI_{i,t}^{DR}-\sum_iC_{i,t}^{\mathrm{comp}}-C_t^{u,\mathrm{service}}.
 \end{aligned}}
\]

代码在同一测试中核对完整账本和简化账本数值相等。备用本身的直接净贡献为 \(-a_t^{\mathrm{back}}B_t^{\mathrm{back}}\Delta t\)。PG成本在完整账本按最终实际购电统一计算一次。

这个恒等式固定了“备用按成本回收、UC承担调用代价”的收支口径。由报价、市场份额、亏损普通订单等产生的整体策略影响，在第25节列作实验检查。

## 9. 环境状态转移与一次性提交

UC决策前状态采用：

\[
 s_t^U=\Big[t,p_t^g,E_t^{ESS},c_t^{PG},
 \{p_{i,t}^*,E_{i,t}^u,\Delta p_{i,t}^{\max},B_{i,t},\mathcal Q_{i,t}\}_{i=1}^N,
 \chi_t,\text{市场历史、备用合同、已结算记录}\Big].
\]

\(\mathcal Q_{i,t}\) 为带来源时段和期限的待回补队列，\(B_{i,t}\) 为队列能量和。UC发布计划后形成 \(s_t^C=(s_t^U,a_t^u,\mathrm{PublicOffer}_t)\)，消费者 Actor 仍取自身10维观测。

一次执行结束，计算以下状态增量：

\[
 E_{i,t+1}^{u}=E_{i,t}^{u}+(p_{i,t}^{u}+p_{i,t}^{\mathrm{back}})\Delta t,
\]

\[
 B_{i,t+1}=B_{i,t}+s_{i,t}^{+}\Delta t-r_{i,t}^{\mathrm{act}}\Delta t,
\]

以及按实际充放电得到的ESS新能量。队列只扣除真实完成的回补量，新移出量添加为新的原生任务；并生成等待、逾期、终端状态。

市场未成交 `market_unfilled_kw`、备用 `backup_kw`、请求缺口 `unserved_request_kw`、原生损失 `native_lost_kw`、待回补请求 `repay_unfilled_kw`、核验DR `verified_positive_dr_kw`、补偿DR `paid_dr_kw` 独立存储。

`engine` 的提交顺序是：**生成全部候选结果 → 校验物理与服务账本 → 核验事件DR并计算补偿 → 结算与Reward → 提交SOC/月累计/任务队列/事件支付键 → 时间加1**。

每轮使用唯一 `transition_id`；`engine.commit()` 为同一ID执行一次状态变更。计算函数保持纯读入/返回结果，回放、诊断或训练重新读取同一轨迹时复用既有结算记录。

## 10. Stackelberg–Nash 博弈定义

给定 UC 的策略 \(\pi_u\)，消费者在联合报价、DR 及市场出清映射上形成非合作博弈。消费者 \(i\) 的成本为：

\[
 J_i(\pi_i,\pi_{-i};\pi_u)=
 \mathbb E_{\pi_u,\pi_i,\pi_{-i},\mathcal M_{\rm PAB}}
 \left[\sum_t\gamma^t C_{i,t}\right].
\]

下层 Nash 响应 \(\pi_c^*(\pi_u)=(\pi_1^*,\ldots,\pi_N^*)\) 满足：

\[
\boxed{
J_i(\pi_i^*,\pi_{-i}^*;\pi_u)
\le J_i(\pi_i,\pi_{-i}^*;\pi_u),\qquad \forall i,\ \forall\pi_i.
}
\]

UC 作为领导者，优化：

\[
\boxed{
\max_{\pi_u}\ J_u\big(\pi_u,\pi_c^*(\pi_u)\big).
}
\]

这是本文拟学习的动态 Stackelberg–Nash 策略关系。Pay-as-Bid 出清、PG成本补供、原生优先服务和DR比例采购共同组成收益与状态转移映射。\(p_t^m\) 通过补偿份额影响消费者收益与策略；物理DR可行性由本地规则提供。

---

# 第三部分　神经网络与训练

## 11. 审核原论文 Fig. 4 的 Encoder / Decoder 结构

### 11.1 原论文如何描述

原论文 §III-D 和 Fig. 4 描述的 Consumer Policy Encoder 接收聚合 Observation：

\[
 o_{\Sigma,t}^{c}=\sum_i o_{i,t}^{c}.
\]

其输出被原文称为聚合消费者动作 \(\sum_i a_{i,t}^c\)；随后 Individual Decoder 读取消费者 \(i\) 的 Observation 及上述聚合动作，生成个人动作。原文将共享 Encoder 用于 Stackelberg 学习阶段，将个体 Decoder 用于下层 Nash 学习阶段。

如果把 Encoder 的输出明确记作聚合预测 \(\widehat a_{\Sigma,t}^c\)，该流程可写为：

\[
 \widehat a_{\Sigma,t}^c=F_{\theta_c}\!\left(\sum_i o_{i,t}^c\right),
\qquad
 a_{i,t}^c=G_{\theta_i}(o_{i,t}^c,\widehat a_{\Sigma,t}^c).
\]

对于这个聚合量，值得在理论与代码实现中审计：

\[
\widehat a_{\Sigma,t}^c\ \overset{?}{=}\ \sum_i a_{i,t}^c.
\]

原论文的网络说明和 Algorithm 1 提供了训练流程，但对上述聚合输出与所有个体 Decoder 实际输出之间的恒等关系、聚合一致性约束，以及成交分配映射的定义，还需要独立建立实验检验。这里的 \(\sum_i a_i^c\) 属于**策略动作聚合**，而本文的市场出清 \(p_{i,t}^u,p_{i,t}^r\) 属于**成交结果**。

### 11.2 本文第一版：每个消费者独立 Actor

\[
\boxed{
 h_{i,t}=f_{\phi_i}(o_{i,t}),\qquad
 a_{i,t}^c\sim\pi_{\psi_i}(\cdot\mid h_{i,t}).
}
\]

其中：

- \(o_{i,t}\in\mathbb R^{10}\)：前述 10 维消费者 Observation；
- \(f_{\phi_i}\)：消费者 \(i\) 自己的特征编码网络；
- \(h_{i,t}\in\mathbb R^H\)：**隐藏特征向量**，例如 H=64；
- \(\pi_{\psi_i}\)：消费者 \(i\) 的动作分布网络；
- \(a_{i,t}^c\in\mathbb R^4\)：自己的削减 DR、转移 DR、UC 报价和 DER 报价。

\(h_{i,t}\) 是本文为了描述网络中间层而引入的符号，由神经网络内部学习；它的每一维属于潜在特征，并在向量空间中供 Decoder 使用。

第一版中每个消费者分别维护参数 \((\phi_i,\psi_i)\)。全部消费者给出动作后，环境一次性调用市场：

\[
 \boxed{
 (p_{1:N,t}^u,p_{1:N,t}^r)
 =\mathcal M_{\rm PAB}(a_{1:N,t}^c,P_t^{u,\mathrm{avail}},P_t^{r,\mathrm{avail}}).
 }
\]

### 11.3 参数共享的后续对照方案

后续可以让所有消费者共享一个**局部特征提取器的参数** \(\theta_{enc}\)：

\[
 h_{i,t}=f_{\theta_{enc}}(o_{i,t}),
 \qquad a_{i,t}^c\sim\pi_{\psi_i}(\cdot\mid h_{i,t}).
\]

这里的输入仍是 \(o_{i,t}\)；不同消费者通过同一个编码函数，分别获得各自的 \(h_{i,t}\)。训练时共享 Encoder 的梯度由各消费者的数据汇总，个体 Decoder 保留各自参数。方案 A（独立 Actor）和方案 B（共享局部 Encoder）使用相同的 Pay-as-Bid 市场与奖励，便于执行公平的消融实验。

## 12. CTDE 的 Critic 与决策时刻

PPO 基础版为每个参与者使用集中状态价值网络。UC 的价值估计在 UC 作出动作前计算；消费者的价值估计在 UC 已公布计划、消费者尚未采样动作时计算：

\[
 \boxed{V_{\omega_u}(s_t^U),\qquad V_{\omega_i}(s_t^C).}
\]

本时段消费者动作与成交结果在其动作生成后产生，用于回报、训练标签和诊断日志。下一时段消费者价值使用 \(s_{t+1}^C\)，由下一时段 UC 公告形成后补齐到对应轨迹。

选用 MAAC 动作价值网络时，集中输入可写为
\(Q_i(s_t^C,a_{1:N,t}^c)\)。市场处理后的特征可作为动作后信息用于相应 Q 模型，PPO Value 基线则遵循上述决策前状态约定。两种 Critic 通过配置分别实现。

第一版保持独立消费者 Actor、自身 10 维输入、集中式决策前 Value Critic 和独立 UC Actor。[R-PPO] 提供 PPO 的状态价值、优势及 surrogate 训练形式；双阶段状态是本模型依据行动时序作出的实现约定。

## 13. 经验采样与策略更新

### 13.1 一个物理时段的采样与执行

1. 读取 \(s_t^U\)，UC Actor 生成候选计划，保存原始动作及 `log_prob`。
2. `uc.prepare_offer()` 形成可行计划、两池额度、事件采购上限和规则版本，得到 \(s_t^C\)。
3. 各消费者按自身10维观测独立产生四维动作，保存原始采样概率。
4. `dr.prepare_local_request()` 根据本地能力和余额得到削减、移出、回补请求及净需求。
5. `market.clear()` 完成两池PAB，返回成交与未成交需求。
6. `physics.prepare_regular_dispatch()` 下调富余ESS放电，得到基础PG和实际弃电。
7. `reserve.allocate_backup()` 在PG余量内按未成交需求比例补供。
8. `physics.finalize_execution()` 得到最终PG与SOC；`dr.realize_service()` 分解原生服务和真实回补。
9. `incentive.procure()` 核验事件内正响应，按 \(p_t^m\) 分配获补偿量。
10. `settlement.settle()` 形成PAB、成本备用、DR和补偿账本；计算个体Reward。
11. `engine.commit()` 一次提交所有状态、事件支付键与时间，保存联合经验。

联合经验保留 `consumer_actions_raw`、`dr_request`、`market_result`、`backup_result`、`physical_execution`、`load_service`、`dr_incentive`、`ledger`、全部原始 `log_prob`、`terminated` 和 `truncated`。

PPO概率比对应当时策略采样的原始动作；本地动作映射、市场分配和实际服务属于产生回报的环境过程。事件采购量是结算结果，四维动作仍为削减、移峰和两类报价。

### 13.2 双层策略学习：学习对象、时间尺度与更新方向

本节把**一个物理时段的环境执行**和**两层 Actor 的参数更新**分开。环境始终是同一个 `EnergyMarketEngine`。策略优化采用“消费者快、UC 慢”的两层安排：

\[
\boxed{\text{固定 UC → 消费者形成近似 Nash 响应 → UC 更新 → 消费者重新适应}.}
\]

设 UC Actor 的参数为 \(\theta_u\)，所有消费者 Actor 参数为 \(\theta_c=(\theta_1,\ldots,\theta_N)\)。定义消费者对给定领导者策略的响应集合：

\[
\mathrm{NE}(\theta_u)
=\Big\{\theta_c:\;J_i(\theta_i,\theta_{-i};\theta_u)
\le J_i(\theta_i',\theta_{-i};\theta_u),\ \forall i,\theta_i'\Big\}.
\]

当数值过程选定某个近似均衡分支时记为 \(\widehat\theta_c^*(\theta_u)\)。UC 想最大化的响应收益是：

\[
\boxed{W(\theta_u)=J_u\big(\theta_u,\theta_c^*(\theta_u)\big).}
\]

其中消费者的 \(J_i\) 使用实际 PAB、成本备用、事件补偿及负荷服务记录计算的长期成本；UC 的 \(J_u\) 使用真实供电和收入账本计算的长期收益。

#### 内层：固定 UC，更新消费者

对于外层轮次 \(k\)，固定 \(\theta_u^k\)。每轮内层从环境采集当前策略的新轨迹，依次训练集中 Critic 与各消费者 Actor。

\[
\delta_{i,t}=r_{i,t}+\gamma(1-d_t^{\mathrm{term}})
V_{\omega_i}(s_{t+1}^{C})-V_{\omega_i}(s_t^{C}),
\qquad
\hat A_{i,t}=\delta_{i,t}+\gamma\lambda m_t\hat A_{i,t+1}.
\]

\(d_t^{\mathrm{term}}\) 标识任务真实终止；\(m_t\) 标识下一条优势是否属于同一连续轨迹。真实终止时bootstrap为0；采样截止时使用合法下一状态的价值，结束本段优势递推。UC使用 \(s^U\) 形成对应目标。每个完整物理时段计一次折扣。价值目标与优势在一次PPO更新批次内固定。[R-PPO]

消费者采用 PPO clipped surrogate：

\[
L_i^{\mathrm{clip}}=\mathbb E\!\left[
\min\!\left(\rho_{i,t}(\theta_i)\hat A_{i,t},
\operatorname{clip}(\rho_{i,t}(\theta_i),1-\epsilon,1+\epsilon)\hat A_{i,t}
\right)\right].
\]

`learning/mappo.py` 对每个消费者分别更新 Actor 参数，使用每人的个体 reward；Critic 训练期读取全局状态。每次 PPO 更新使用对应策略重新采样的 on-policy rollout。内层偏离检查详见第 13.3 节。

#### 外层：固定当前消费者，更新 UC

冻结本轮内层得到的消费者 Actor 参数 \(\widehat\theta_c^*(\theta_u^k)\)，使用当前 UC 与 Consumer Actor 在同一环境上采集新轨迹。先更新 UC Critic，然后以 UC 自身实际长期收益更新 UC Actor。该操作产生候选：

\[
\theta_u^{k+1}=\theta_u^k+\alpha_u\widehat g_u^{\mathrm{PPO}}.
\]

这里 \(\widehat g_u^{\mathrm{PPO}}\) 是消费者参数暂时固定时的 UC 策略梯度估计。消费者 Actor 虽然被冻结，仍会随着 UC 公共供给条件改变而产生不同动作；随后通过重新训练消费者来进一步逼近新 UC 策略下的 Nash 响应。

#### 外层候选接受与重新响应

每次形成 UC 候选策略后，使用完整的“**固定候选 UC → 再训练消费者 → 独立验证轨迹评估长期收益**”过程，估计 \(W(\theta_u^{k+1})\)。同时用旧 UC 策略及其已训练响应计算参考值 \(W(\theta_u^k)\)。在预先规定的候选接受规则下更新当前 UC 策略，并保存对应的消费者响应checkpoint。回退同时恢复Actor、Critic、优化器、归一化器和随机数状态，下一批PPO轨迹由对应当前策略重新采集。

上层每次更新涉及两种不同的评价：

- **固定消费者的快速 UC 更新**：提高迭代效率，产生新候选。
- **消费者重新适应后的响应评价**：检验这个候选是否改善了消费者适应后的响应收益估计。

### 13.3 怎样判断消费者已经接近 Nash？

固定 \(\theta_u\) 和其余消费者的 \(\pi_{-i}\)，单独训练消费者 \(i\) 的最佳响应候选 \(\widetilde\pi_i\)。使用期望**收益最大化**的记号 \(U_i=-J_i\)，定义搜索得到的单边改进量：

\[
\boxed{\widehat\epsilon_i^{NE}=\left[
U_i(\widetilde\pi_i,\pi_{-i};\pi_u)
-U_i(\pi_i,\pi_{-i};\pi_u)\right]_+.}
\]

每轮验证时执行：固定 UC、固定其他消费者、重训单个消费者、使用相同验证场景计算收益差。对所有消费者取最大值：

\[
\widehat\epsilon_c^{NE}=\max_i\widehat\epsilon_i^{NE}.
\]

PPO KL 与 episode return 用于跟踪更新稳定性；\(\widehat\epsilon_c^{NE}\) 用于发现能够改进的单边策略。有限预算的最佳响应搜索给出的是**已发现偏离的改进量**：若该值显著为正，当前策略存在可利用的单边偏离；若很小，则表示在规定搜索预算内没有找到显著改进。小规模离散报价/DR 博弈可以使用精确枚举获得更强的均衡检验。

### 13.4 怎样判断 UC 已经接近 Stackelberg 最优响应？

UC 的评价必须在消费者对候选 UC 策略重新适应后执行。定义：

\[
\widehat W(\pi_u)=
\widehat J_u\big(\pi_u,\widehat\pi_c^*(\pi_u)\big).
\]

对一组候选 UC 策略 \(\mathcal P_{\mathrm{test}}\)，每个候选都独立训练消费者响应并评估 \(\widehat W\)。得到**UC 候选响应改进量**：

\[
\boxed{\widehat\epsilon_u^{SE}
=\max_{\widetilde\pi_u\in\mathcal P_{\mathrm{test}}}
\left[\widehat W(\widetilde\pi_u)
-\widehat W(\pi_u^{\mathrm{current}})\right]_+.}
\]

在实现上，每个候选使用相同验证负荷、DER和价格序列，配合多种子均值与置信区间。候选选择使用验证集；最终论文测试集留至算法冻结后。每个候选采用同一响应训练预算及明确的初始化/分支选择规则。[R-SP] 所有候选使用同一事件日历、成本备用规则、DR采购分配、服务优先级与期末待办成本。机制版本号、事件支付状态和队列初始化一同写入实验清单。

将下列量同时用于停止决策：

| 检测量 | 含义 | 角色 |
|---|---|---|
| \(\widehat\epsilon_c^{NE}\) | 消费者搜索得到的最大单边改进 | 下层已发现偏离的检验 |
| \(\widehat\epsilon_u^{SE}\) | 测试候选 UC 经响应优化后的最大收益改进 | 上层候选响应改进的检验 |
| `uc_policy_kl` | 固定评估观测下 UC 新旧策略的 KL | 更新幅度与早停辅助 |
| `uc_return_ci` | 经响应后的 UC 收益置信区间 | 对比噪声与稳定性 |
| `consumer_return_ci` | 消费者成本分布与置信区间 | 保证个体评价稳定 |
| `market_unfilled_kwh` | 市场未成交电量 | 市场紧缺程度 |
| `backup_energy_kwh` | 备用补购电量 | 应急依赖程度 |
| `native_lost_kwh` | 当前原生实际损失电量 | 服务可靠性 |
| `pending_shift_kwh` / `overdue_kwh` | 移峰待办与逾期余额 | 跨时段履约 |
| `verified_dr_kwh` / `paid_dr_kwh` | 核验与获补偿电量 | DR采购效果 |

可作为第一版起始配置的验收条件为：\(\widehat\epsilon_c^{NE}<\tau_c\)、\(\widehat\epsilon_u^{SE}<\tau_u\)，并且 UC 收益、策略 KL 在连续多个外层轮次内稳定。阈值 \(\tau_c,\tau_u\) 使用与 reward 同量纲的成本/收益尺度，结合统计置信区间选取。

### 13.5 训练频率与理论界限

基础方案采用 `consumer_updates_per_outer > uc_updates_per_outer`，例如消费者先做多轮 PPO、UC 再做一轮 PPO。若构建随机逼近理论，可设消费者步长 \(\beta_k\)、UC 步长 \(\alpha_k\) 且满足：

\[
\sum_k \alpha_k=\sum_k\beta_k=\infty,\qquad
\sum_k\alpha_k^2<\infty,\quad\sum_k\beta_k^2<\infty,\qquad
\frac{\alpha_k}{\beta_k}\to0.
\]

这表达消费者更新更快、UC 更新更慢。上述条件属于分析理想化随机逼近过程时使用的假设；实际 PPO 使用固定学习率、神经网络逼近和排序式 PAB 时，以响应偏离评估和多种子实验报告训练结果。

如果未来引入局部连续可微的消费者均衡响应 \(b^*(u)\)，UC 可以学习：

\[
\frac{dJ_u(u,b^*(u))}{du}
=\frac{\partial J_u}{\partial u}
 +\left(\frac{db^*(u)}{du}\right)^\top \nabla_b J_u.
\]

该式表示 leader 的 total derivative，包括跟随者响应变化对 UC 收益的影响。排序式 PAB 在报价排序发生切换时会出现非光滑结构；可微近似出清或基于候选策略的黑箱响应搜索适合后续研究。

## 14. 双层训练的结构伪代码

下方将训练与环境采样、Nash/Stackelberg 验证明确分开。`sample_episodes` 必须完整经历：UC → 消费者 DR/报价 → PAB 出清 → 供给调整与PG补供 → 实际服务及移峰更新 → 事件DR采购 → 奖励。

```python
# 学习逻辑示意：与 RLlib/PyTorch 具体 API 解耦
uc_policy = init_uc_actor()
consumer_policies = init_independent_consumer_actors(num_consumers)

for outer_round in range(max_outer_rounds):
    # (A) 内层：保持 UC 策略参数固定，让消费者适应
    freeze(uc_policy)
    unfreeze(consumer_policies)
    for _ in range(consumer_updates_per_outer):
        episodes = sample_episodes(engine, uc_policy, consumer_policies)
        update_consumer_critics(episodes)
        update_consumer_actors_on_policy(episodes)

    inner_metrics = evaluate_follower_response(
        uc_policy, consumer_policies, validation_episodes
    )

    # 保存当前 UC 及其近似 Nash 响应
    reference_checkpoint = snapshot_training_state()  # 网络/优化器/归一化/随机数
    old_uc, old_consumers = reference_checkpoint.policy_views()

    # (B) 外层：保持消费者参数固定，对 UC 进行小步更新
    freeze(consumer_policies)
    unfreeze(uc_policy)
    for _ in range(uc_updates_per_outer):
        episodes = sample_episodes(engine, uc_policy, consumer_policies)
        update_uc_critic(episodes)
        update_uc_actor_on_policy(episodes)

    # (C) 用新 UC 条件重新训练消费者并评价真实响应收益
    freeze(uc_policy)
    unfreeze(consumer_policies)
    adapt_followers_to_uc(uc_policy, consumer_policies)

    candidate_return = evaluate_uc_with_adapted_followers(
        uc_policy, consumer_policies, validation_episodes
    )
    old_return = evaluate_uc_with_adapted_followers(
        old_uc, old_consumers, validation_episodes
    )
    uc_policy, consumer_policies = accept_or_restore_training_state(
        candidate_return, old_return,
        candidate=(uc_policy, consumer_policies),
        reference=reference_checkpoint,
    )

    # (D) 周期性昂贵检查：消费者 BR 搜索与 UC 候选响应搜索
    if outer_round % equilibrium_check_interval == 0:
        ne_gap = search_unilateral_consumer_improvements(
            engine, uc_policy, consumer_policies
        )
        se_gap = search_uc_deviations_with_follower_adaptation(
            engine, uc_policy, consumer_policies
        )
        log_metrics(ne_gap, se_gap, candidate_return, old_return)
        if meets_predeclared_stop_rule(ne_gap, se_gap):
            break
```

第一版可以把昂贵的 `search_uc_deviations_with_follower_adaptation` 降低检查频率；每次真正调用该函数时均完整执行“UC 候选 → 消费者重新适应 → 收益评估”。

### 14.1 SNE 强化学习相关文献：学习对象和可借鉴部分

下表的文献来自作者/会议/期刊页面与开源仓库，按与本文“**1 个 UC + 多个 Nash 消费者**”的相关程度排列。它们研究的求解假设各有不同。

| 文献 | 实际研究的 Leader–Follower 关系 | 训练或求解思路 | 对本项目的用处与范围 |
|---|---|---|---|
| **Nie et al., IEEE TSG 2024**（[原论文](https://doi.org/10.1109/TSG.2024.3417535)） | 1 UC + 多个消费者 Nash | 分 Stackelberg 与 Nash 学习阶段，UC total gradient，消费者 MAPPO | **能源场景最直接基线**；关注异步更新与原模型 DR 目标 |
| **Zhong et al., JMLR 2023**（[全文](https://jmlr.org/papers/v24/22-0203.html)） | 1 Leader + 多个 Nash Followers | 在线 optimistic、离线 pessimistic least-squares value iteration | **SNE 定义和理论最贴近**；主要理论针对 **myopic followers（只最大化当期回报）**，与跨时段 DR 动态响应需要分开论证 |
| **Wang et al., AAAI 2022**（[论文](https://ojs.aaai.org/index.php/AAAI/article/view/20457)；[源码](https://github.com/guaguakai/equilibrium_gradient)） | 1 Leader + 多个会形成 Nash 的 Followers | 迭代 Nash oracle + 可微均衡层，让 Leader 对跟随者达到的均衡响应求导 | **最接近多消费者均衡响应求导**；适合做小规模精确/近似 Nash oracle，源码含 `model.py` |
| **Brero et al., Stackelberg POMDP**（[论文](https://arxiv.org/abs/2210.03852)） | 经济系统设计者 + 会学习的参与者 | 把 Followers 的响应学习过程嵌入 Leader 的 POMDP | **适合解释“UC 候选 → 消费者重新训练 → 再评价”**；论文部分结果是 no-regret learners 的 coarse-correlated equilibrium，和 Nash 指标分别看 |
| **Fiez et al., ICML 2020**（[论文](https://proceedings.mlr.press/v119/fiez20a.html)；[源码](https://github.com/fiezt/ICML-2020-Implicit-Stackelberg-Learning)） | 一般连续 Stackelberg 游戏 | 隐式函数、total derivative、局部稳定与随机梯度分析 | 用于以后构造可微上层响应梯度和局部收敛证明 |
| **Zheng et al., AAAI 2022**（[论文](https://ojs.aaai.org/index.php/AAAI/article/view/20908)；[源码](https://github.com/LeoZhengZLY/stackelberg-actor-critic-algos)） | Actor–Critic **网络之间**的 Stackelberg 关系 | STAC / STDDPG / STSAC，局部 Stackelberg 收敛 | 学习 total-gradient 技术。论文中的 Leader/Follower **指 Actor 与 Critic**，可作算法参考，不直接对应 UC/Consumer 市场博弈 |
| **Yang et al., ST-MADDPG 2023**（[预印本](https://arxiv.org/abs/2305.03735)） | 双 Agent 的 Leader–Follower 竞争 | ST-MADDPG 联合连续控制与 Stackelberg 学习 | 适合参考多智能体 Actor-Critic 结构；扩展到 N 个消费者 Nash 需另建响应机制 |
| **Salazar-Peña et al., MARLEM 2026**（[论文](https://doi.org/10.1016/j.apenergy.2026.127546)；[源码](https://github.com/salazarna/marlem)） | 本地能源市场多智能体交易，另有 DSO | 订单撮合后增加 DSO 残余出清 | **直接借鉴未成交需求的备用供电规则与工程模块**；属于市场仿真与 MARL 框架参考 |

#### 推荐的研究顺序

最先读 **原 SN-MAPPO 的 §II、§III-D–E**，建立领域建模；再读 **Zhong 等 JMLR 2023** 明确 SNE 的数学定义及 myopic 前提；接着看 **Wang 等 AAAI 2022** 的多跟随者 Nash oracle 与 Equilibrium Gradient；最后读 **Brero 等** 理解把跟随者训练环节嵌入 Leader 评价过程的框架。实现上优先复用当前环境已有 PPO/MAPPO，均衡 oracle 和隐式导数属于第二阶段开发。


# 第四部分　扁平化工程代码架构

## 15. 总体设计原则与 MARLEM 的对应关系

参考 [MARLEM](https://github.com/salazarna/marlem) 中的 `environment/lem.py`、`environment/observation.py`、`environment/action.py`、`market/order.py`、`market/matching.py`、`market/mechanism.py` 和 `environment/reward.py` 的模块分工。

本项目以一个独立运行的 `EnergyMarketEngine` 调度完整时段，其上增加 RLlib / PettingZoo 的薄适配层。核心物理和市场代码可直接使用 NumPy 执行，并通过配置替换市场规则或 Agent 算法。

### 15.1 项目目录

```text
sn-energy-market/
├── README.md
├── pyproject.toml
├── configs/
│   ├── system.yaml             # Agent 数量、功率边界、ESS、时域
│   ├── market.yaml             # PAB、资源优先级、报价上限与缺供规则
│   ├── rewards.yaml            # DR、购电、弃电、缺供等成本系数
│   └── training.yaml           # MAPPO/PPO、学习率、rollout 等
├── data/
│   ├── load/                   # 消费者原始负荷 p*_{i,t}
│   ├── der/                    # DER 出力 p^g_t
│   └── tariff/                 # PG 购电价、TUTT、DER 与 DR 价格
├── src/
│   ├── snmarket/               # 独立的仿真环境内核
│   │   ├── __init__.py
│   │   ├── contracts.py        # state/action/order/trade/result 数据契约
│   │   ├── config.py           # 环境配置读取与校验
│   │   ├── profiles.py         # 时序负荷、DER 与电价加载
│   │   ├── uc.py               # UC 供给计划、可售容量与合法性
│   │   ├── consumer.py         # 消费者状态与月累计购电量
│   │   ├── dr.py               # 本地DR、真实回补、FIFO任务队列
│   │   ├── incentive.py        # 事件正响应核验、比例DR采购
│   │   ├── tariff.py           # TUTT、DER 基准价、DR 激励价
│   │   ├── observation.py      # Consumer 10 维 / UC 观测生成
│   │   ├── action.py           # Action 解码、截断、可行性检查
│   │   ├── market.py           # PAB 交易出清，输出未成交功率
│   │   ├── reserve.py          # 仅PG备用补供、请求缺口
│   │   ├── settlement.py       # PAB、备用成本回收、DR双边账本
│   │   ├── physics.py          # ESS放电下调、PG实际功率、SOC增量
│   │   ├── reward.py           # UC 和个体消费者奖励
│   │   ├── metrics.py          # DR、DER、市场、费用与均衡指标
│   │   ├── engine.py           # 单物理时段的唯一编排入口
│   │   ├── rllib_env.py        # RLlib MultiAgentEnv 薄适配层
│   │   └── pettingzoo_env.py   # 可选 AEC/Parallel 适配层
│   └── learning/               # 网络与训练流程，与物理环境分开
│       ├── __init__.py
│       ├── actors.py           # 独立 Consumer Actor 与 UC Actor
│       ├── critics.py          # 中心化 Value / 可选 Attention Q
│       ├── mappo.py            # PPO/GAE 算法组件
│       ├── bilevel_train.py   # 内层消费者快更新、外层 UC 慢更新
│       └── equilibrium.py     # consumer BR 与 UC candidate 响应评估
├── scripts/
│   ├── smoke_test.py           # 固定动作跑单步/全天，无 RL 依赖
│   ├── train.py                # 启动训练
│   ├── evaluate.py             # 固定策略、多种子评估
│   ├── compare_markets.py      # PAB 与 Uniform 对比
│   ├── best_response.py       # Nash/Stackelberg 偏离搜索
│   └── evaluate_shortage.py   # 缺供与备用服务消融
├── tests/
│   ├── test_contracts.py
│   ├── test_uc.py
│   ├── test_dr.py
│   ├── test_incentive.py
│   ├── test_shift_queue.py
│   ├── test_market.py
│   ├── test_reserve.py
│   ├── test_bilevel_train.py
│   ├── test_equilibrium.py
│   ├── test_settlement.py
│   ├── test_physics.py
│   ├── test_observation.py
│   ├── test_engine.py
│   └── test_actor_information.py
└── docs/
    ├── modeling.md
    ├── architecture.md
    └── experiments.md
```

第一版实现时，`src/snmarket/` 下每个业务文件只表达一个主要数学模块，所有跨模块串联发生在 `engine.py`。训练器可直接替换成自研 MARL 框架或 RLlib，市场层仍使用相同订单、成交和状态转移接口。

## 16. 数据契约与张量规范

基础接口保持7字段UC计划、10维消费者观测、4维消费者动作。第8个观测的语义升级为DR采购上限；新版字段统一命名 `dr_procurement_cap_kw`，原 `dr_target_kw` / `uc_dr_target_kw` 在升级时显式迁移。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class UCAction:
    grid_buy_plan_kw: float
    der_curtail_plan_kw: float
    ess_charge_plan_kw: float
    der_to_ess_plan_kw: float
    uc_to_ess_plan_kw: float
    ess_discharge_plan_kw: float
    dr_procurement_cap_kw: float

@dataclass(frozen=True)
class ConsumerAction:
    dr_reduce_kw: float
    dr_shift_kw: float
    uc_bid_price: float
    der_bid_price: float

@dataclass(frozen=True)
class PublicOffer:
    uc_available_kw: float
    der_available_kw: float
    dr_procurement_cap_kw: float
    der_tariff: float
    dr_incentive: float
    time_index: int
    rule_version: str

@dataclass(frozen=True)
class DRRequest:
    consumer_id: str
    reduce_kw: float
    shift_out_kw: float
    repay_requested_kw: float
    native_need_kw: float
    net_demand_kw: float

@dataclass(frozen=True)
class MarketOrder:
    consumer_id: str
    net_demand_kw: float
    uc_bid_price: float
    der_bid_price: float

@dataclass(frozen=True)
class Allocation:
    consumer_id: str
    uc_kw: float
    der_kw: float
    market_unfilled_kw: float

@dataclass(frozen=True)
class Fulfillment:
    consumer_id: str
    market_uc_kw: float
    market_der_kw: float
    market_unfilled_kw: float
    backup_kw: float
    unserved_request_kw: float

@dataclass(frozen=True)
class LoadService:
    consumer_id: str
    native_served_kw: float
    native_lost_kw: float
    repay_actual_kw: float
    repay_unfilled_kw: float
    backlog_after_kwh: float
    active_dr_kw: float

@dataclass(frozen=True)
class DRIncentive:
    consumer_id: str
    event_id: str
    verified_positive_dr_kw: float
    paid_dr_kw: float
    payment: float

@dataclass(frozen=True)
class ShiftTask:
    source_period: int
    deadline_period: int
    remaining_kwh: float

@dataclass(frozen=True)
class MarketResult:
    allocations: dict[str, Allocation]
    uc_total_kw: float
    der_total_kw: float
```

`RegularDispatch` 保存基础PG、实际弃电和ESS放电下调量；`BackupContext` 保存物理余量、合同额度和成本定价规则；`Ledger` 分开记录普通账单、备用成本回收、外部服务支出、UC调用代价、DR补偿和服务责任；`StateDelta` 保存整轮待提交更新。

| 内容 | 基础形状 | 含义 |
|---|---|---|
| Consumer Observation | `[10]` / `[B,N,10]` | 当前10字段，各参与者分别构造 |
| Consumer Action | `[4]` / `[B,N,4]` | 原始削减、移峰、两类报价 |
| 市场成交 | `[N,2]` | UC、DER功率 |
| 未成交/备用/请求缺口 | 各`[N]` | 市场与补供账本 |
| 原生服务/原生损失/真实回补/回补等待 | 各`[N]` | 实际服务分解 |
| 移峰余额 | `[N]` | 队列能量和，kWh |
| 核验正响应/获补偿DR | 各`[N]` | 第8节采购分配 |
| Consumer Reward | `[N]` | 个体实际长期成本的负值 |
| UC Reward | `[]` | UC收益标量 |

完整状态还保存每个消费者的 `ShiftTask` 队列和已提交 `transition_id` 集合。向量化训练时，队列等可变长结构由环境管理，Critic输入由对应状态编码器按确定方式生成。

## 17. 各模块函数接口

| 文件 | 输入 | 核心函数 | 输出 |
|---|---|---|---|
| `profiles.py` | 场景、日期、consumer id | `load_profiles()` | 原始负荷、DER、PG价、事件日历 |
| `uc.py` | UC状态与原始动作 | `prepare_offer()` | 可行计划、普通容量、采购上限 |
| `dr.py` | 本地动作、本地负荷/余额 | `prepare_local_request()` | `DRRequest` |
| `dr.py` | `DRRequest`、实际服务、任务队列 | `realize_service()` | `LoadService`、队列更新量 |
| `incentive.py` | 实际DR、事件、采购上限 | `procure()` | 每人核验量、获补偿量和金额 |
| `observation.py` | 本地状态、公告 | `build_consumer_observation()` | `[10]` |
| `market.py` | 净需求和两类报价、公告 | `clear()` | 两池成交与未成交 |
| `physics.py` | 计划、普通成交 | `prepare_regular_dispatch()` | ESS下调、基础PG、实际弃电 |
| `reserve.py` | 未成交、PG余量、合同限额 | `allocate_backup()` | 补供、请求缺口 |
| `physics.py` | 普通调度与补供 | `finalize_execution()` | 最终PG、能量和残差 |
| `settlement.py` | 交易/服务/采购记录、价格 | `settle()` | 双边资金账本 |
| `reward.py` | 账本、本地调整、队列 | `calculate_rewards()` | 各Agent奖励 |
| `engine.py` | UC/Consumer Actions | 两阶段`apply_*`与`commit()` | 下一观测与完整转移 |
| `learning/equilibrium.py` | policy状态、验证场景 | BR与UC响应搜索 | 改进量与置信区间 |

### 17.1 Observation 单独实现

```python
def build_consumer_observation(state, public_offer, consumer_id):
    c = state.consumers[consumer_id]
    return np.asarray([
        state.time_index,
        c.original_load_kw,
        c.monthly_uc_energy_kwh,
        c.max_dr_kw,
        tariff.uc_price(state.time_index, c.monthly_uc_energy_kwh),
        public_offer.der_tariff,
        public_offer.dr_incentive,
        public_offer.dr_procurement_cap_kw,
        public_offer.uc_available_kw,
        public_offer.der_available_kw,
    ], dtype=np.float32)
```

这里 `np`、`tariff` 和状态类由项目相应模块导入。表中第8项保持原位置，并明确采购含义。

### 17.2 Pay-as-Bid 市场的核心实现思路

```python
class PayAsBidMarket:
    def clear(self, orders, offer) -> MarketResult:
        remaining = {o.consumer_id: o.net_demand_kw for o in orders}
        der_fill = {o.consumer_id: 0.0 for o in orders}
        uc_fill = {o.consumer_id: 0.0 for o in orders}

        # 第 1 轮：竞争 DER 容量
        der_left = offer.der_available_kw
        for o in sorted(orders, key=lambda x: (-x.der_bid_price, x.consumer_id)):
            q = min(remaining[o.consumer_id], der_left)
            der_fill[o.consumer_id] += q
            remaining[o.consumer_id] -= q
            der_left -= q

        # 第 2 轮：剩余需求竞争 UC 容量
        uc_left = offer.uc_available_kw
        for o in sorted(orders, key=lambda x: (-x.uc_bid_price, x.consumer_id)):
            q = min(remaining[o.consumer_id], uc_left)
            uc_fill[o.consumer_id] += q
            remaining[o.consumer_id] -= q
            uc_left -= q

        return build_market_result(orders, uc_fill, der_fill, remaining)
```

该片段展示排序分配的核心；项目实现还需在 `action.py` 与 `market.py` 完成边界验证、零负荷处理及成交记录。支付发生在 `settlement.py`：

```python
payment_i = (
    order.uc_bid_price * allocation.uc_kw
    + order.der_bid_price * allocation.der_kw
) * dt_hours
```

### 17.3 PG补供与服务分解接口

`reserve.py` 输入市场出清后尚未成交的功率和真实备用供给余量，输出每个消费者的备用补供功率及当期请求缺口。示意：

```python
def allocate_backup(market_result, reserve_available_kw):
    import math
    if not math.isfinite(reserve_available_kw) or reserve_available_kw < 0:
        raise ValueError("备用额度应为有限非负值")
    allocations = market_result.allocations
    remaining = {
        i: a.market_unfilled_kw
        for i, a in allocations.items()
    }
    if any(not math.isfinite(q) or q < 0 for q in remaining.values()):
        raise ValueError("市场未成交量应为有限非负值")
    total_unfilled = sum(remaining.values())
    ratio = min(1.0, reserve_available_kw / total_unfilled) if total_unfilled > 0 else 0.0
    result = {}
    for i, a in allocations.items():
        backup_kw = remaining[i] * ratio
        result[i] = Fulfillment(
            consumer_id=i,
            market_uc_kw=a.uc_kw,
            market_der_kw=a.der_kw,
            market_unfilled_kw=remaining[i],
            backup_kw=backup_kw,
            unserved_request_kw=remaining[i] - backup_kw,
        )
    return result
```

物理约束与备用资源校验由 `physics.py` 和 `uc.py` 完成；上例的 `reserve_available_kw` 由可调用备用资源计算。`settlement.py` 在 PAB 付款之外单独计入备用费用，再由 `dr.realize_service()` 将 `unserved_request_kw` 分解为原生损失与待回补请求，分别进入服务代价及任务状态。

### 17.4 事件采购接口

```python
def procure_dr(services, event_active, procurement_cap_kw,
               incentive_per_kwh, dt_hours):
    eligible = {
        i: max(0.0, service.active_dr_kw) if event_active else 0.0
        for i, service in services.items()
    }
    total = sum(eligible.values())
    ratio = min(1.0, procurement_cap_kw / total) if total > 0.0 else 0.0
    return {
        i: {
            "verified_positive_dr_kw": q,
            "paid_dr_kw": q * ratio,
            "payment": q * ratio * incentive_per_kwh * dt_hours,
        }
        for i, q in eligible.items()
    }
```

该函数接收已执行的服务记录。物理请求生成函数独立完成本地DR可行性；`p_t^m` 只进入上述采购环节及消费者观测。

### 17.5 物理时间与市场阶段分离

为适配先 UC、后 Consumers 的决策顺序，`EnergyMarketEngine` 提供两阶段接口：

```python
class EnergyMarketEngine:
    def reset(self, *, seed: int | None = None):
        """重置 episode，返回 UC 的时段初始观测。"""
        ...

    def apply_uc_action(self, uc_action: UCAction):
        """固定本时段 UC 供给计划，向消费者公布 10 维观测。"""
        ...

    def apply_consumer_actions(self, actions: dict[str, ConsumerAction]):
        """完成DR请求、PAB、补供、真实服务、事件采购、结算与状态提交。"""
        ...

    def run_period(self, uc_action, consumer_actions):
        self.apply_uc_action(uc_action)
        return self.apply_consumer_actions(consumer_actions)
```

`apply_uc_action()` 更新市场阶段，保留当前 \(t\)；`apply_consumer_actions()` 结束本轮出清及物理执行后更新 \(t\leftarrow t+1\)。这使每个完整物理时段恰好对应一次消费者成交和一次物理状态更新。

## 18. 完整 `engine` 流程与 RLlib 适配

### 18.1 Environment Engine 的内部运行顺序

```text
reset() -> UC observation, state s_t^U
  |
apply_uc_action(raw_uc_action)
  |-- action.decode_uc()                  -> 物理可行UC计划
  |-- uc.prepare_offer()                  -> 两池容量、事件采购上限
  |-- build_consumer_observation()         -> 每人10维，状态s_t^C
  |
apply_consumer_actions(raw_consumer_actions)
  |-- dr.prepare_local_request()          -> 本地DR/回补请求/市场净需求
  |-- market.clear()                      -> DER和UC成交、unfilled
  |-- physics.prepare_regular_dispatch()  -> 富余时下调ESS，反算基础PG
  |-- reserve.allocate_backup()           -> PG补供、请求缺口
  |-- physics.finalize_execution()         -> 最终功率/SOC校验
  |-- dr.realize_service()                 -> 原生优先、实际回补、队列更新量
  |-- incentive.procure()                  -> 事件核验、比例采购、DR金额
  |-- create_execution_record()            -> 完整功率/服务/采购记录
  |-- settlement.settle()                  -> PAB与成本备用、DR、赔付
  |-- reward.calculate_rewards()           -> UC和各消费者奖励
  |-- engine.commit(state_delta)           -> 一次提交队列/SOC/月累计/支付键
  |-- metrics.record()
  v
next UC observation, state s_(t+1)^U
```

每个模块先返回结果，由 `engine.commit()` 统一更新状态。原始采样动作和 `log_prob` 由采样器记录，并通过 `transition_id` 与执行记录关联。计算异常返回带状态、输入和残差的诊断；当前已确认的五个机制均使用确定执行分支。

### 18.2 RLlib 动态行动智能体

`rllib_env.py` 维护两个阶段：`UC_TURN` 与 `CONSUMER_TURN`。

| 调用 | 环境输入 Action 字典 | 返回的下一阶段 Observation 字典 |
|---|---|---|
| `reset()` | — | `{"uc": uc_obs}` |
| `step()`：UC 阶段 | `{"uc": uc_action}` | `{"consumer_0": obs_0, ..., "consumer_N": obs_N}` |
| `step()`：Consumer 阶段 | `{"consumer_0": a_0, ..., "consumer_N": a_N}` | `{"uc": next_uc_obs}` |

所有本时段的 UC 和消费者 reward 在消费者阶段完成出清和物理执行后归属至对应 Agent；episode 完成时设置 `terminateds["__all__"]`。训练侧保留同一物理时段完整joint transition；两个接口调用共同组成一个物理步。消费者下一观测在下一次UC公告后补齐。真实终止、采样截断和bootstrap分别记录。

### 18.3 网络与环境的依赖边界

```text
configs/data
   -> snmarket.contracts, profiles, tariff
   -> snmarket.uc, consumer, dr, incentive, market, reserve, settlement, physics
   -> snmarket.engine
   -> snmarket.rllib_env / pettingzoo_env
   -> learning.actors / critics / mappo / bilevel_train
```

`market.py` 接收订单和公共容量，生成市场成交结果；`reserve.py` 处理PG补供；`dr.py` 处理真实服务和队列；`incentive.py` 处理事件采购；`reward.py` 读取统一结算记录；Actor、Critic 和优化器位于 `learning/`。由此可以分别替换报价机制、DR 模型或 MARL 算法。

## 19. 配置与开发示例

### 19.1 `configs/system.yaml`

```yaml
resolution_minutes: 15
num_consumers: 8
horizon_steps: 96
uc:
  grid_import_max_kw: 100.0
  ess_energy_capacity_kwh: 200.0
  ess_charge_max_kw: 25.0
  ess_discharge_max_kw: 25.0
consumers:
  dr_limit_ratio: 0.20
```

### 19.2 市场与奖励配置示例

`configs/market.yaml`：

```yaml
mechanism_version: v5_Q1A_Q2A_Q3B_Q4A_Q5B
mechanism: pay_as_bid
resource_priority: [der, uc]
tie_breaking: consumer_id
partial_fill: true
uc_bid_cap: 1.50
der_bid_cap: 1.50
backup_source: pg_only
surplus_ess_policy: reduce_discharge_to_regular_sales
backup_ess_policy: keep_regular_dispatch
backup_allocation: proportional_unfilled
backup_capacity_limit_kw: 20.0
backup_price_mode: actual_grid_cost_plus_external_service
backup_external_service_rate_per_kwh: 0.10
backup_retained_markup_per_kwh: 0.0
# actual_grid_cost由profiles/当前状态读取。
dr_physical_rule: local_feasible_adjustment
dr_cap_semantics: compensated_procurement_power
dr_procurement_max_kw: 40.0       # 示例：UC在事件内可选择的采购上限
dr_procurement_allocation: proportional_verified_positive
dr_payment_policy: event_positive_net_response
dr_baseline: frozen_original_load
dr_event_intervals: [[68, 80]]      # 示例：15分钟粒度下[17:00,20:00)
shift_direction: defer_then_repay
shift_service_priority: native_then_fifo_repay
shift_backlog_limit_kwh: 25.0
shift_deadline_steps: 96            # 示例：新移峰任务允许24小时完成
shift_unfilled_repayment: carry_forward
state_commit: once_per_transition_id
```

`configs/rewards.yaml`：

```yaml
backup_activation_rate_per_kwh: 0.20   # UC自担，与外部服务项分开
consumer_native_loss_rate_per_kwh: 3.0
consumer_wait_rate_per_kwh_hour: 0.01
consumer_overdue_rate_per_kwh_hour: 0.50
consumer_terminal_pending_rate_per_kwh: 3.0
uc_native_loss_rate_per_kwh: 2.0
uc_overdue_rate_per_kwh_hour: 0.20
uc_terminal_pending_rate_per_kwh: 2.0
actual_compensation_per_kwh: 0.0       # 开启时双方按同额转移入账
# 抽象DR代价、ESS吞吐和弃电系数按场景另行设置。
```

这些数值用于接口展示，实验时依据统一量纲、成本标定和敏感性分析设置。事件外有效激励价与采购上限取0；事件内 \(p_t^m\) 由UC在允许区间内选择。

### 19.3 `configs/training.yaml`

```yaml
consumer_actor: independent
consumer_obs_dim: 10
consumer_action_dim: 4
consumer_critic: centralized_value
uc_actor: independent
algorithm: ppo
consumer_updates_per_outer: 5
uc_updates_per_outer: 1
follower_readaptation_rounds: 5
ne_check_interval_outer: 5
se_check_interval_outer: 10
consumer_ne_gain_tolerance: 0.01      # 示例阈值，按归一化回报尺度调整
uc_se_gain_tolerance: 0.01            # 示例阈值，按归一化回报尺度调整
stable_outer_rounds: 5
validation_seeds: [17, 31, 47, 73, 101]
# final_test_seeds 在正式实验前独立设置
```

这里的数字属于工程初始样例。运行时记录机制版本、价格来源、任务期限、奖励尺度、种子和训练状态；同一消融实验仅调整指定因素。

## 20. 单元测试、验收与实验记录

| 测试 | 验收关系 |
|---|---|
| `test_observation.py` | 10维及固定字段顺序；TUTT按本消费者普通UC＋备用累计量计算 |
| `test_actor_information.py` | 固定自身观测时，其他消费者私有状态变化保持本Actor直接输出分布 |
| `test_dr.py` | 本地能力、回补余额上限；改变 \(p_t^m\) 后，相同动作的物理请求保持一致 |
| `test_incentive.py` | 事件门控、净正响应、\(\sum q^{paid}\le p^m\)、相同结算ID只提交一次 |
| `test_shift_queue.py` | 原生优先、FIFO真实回补、未完成继续挂账、逾期保留来源与期限 |
| `test_market.py` | 两池排序、平局、部分成交、请求缺口 |
| `test_reserve.py` | PG物理余量、合同限额、比例补供与零需求 |
| `test_settlement.py` | 备用成本回收恒等式、PG采购只计一次、DR和赔付双边相等 |
| `test_physics.py` | 放电下调、充电来源、实际弃电、功率平衡及SOC |
| `test_engine.py` | 先结果后结算，一次状态提交；采样截断保留任务和月累计 |
| `test_bilevel_train.py` | 每层冻结/解冻、on-policy轨迹、候选接受/完整回退 |
| `test_equilibrium.py` | 偏离完整重跑出清、补供、服务及事件采购；UC候选重新适应消费者 |

随附 `mechanism_reference.py` 是Q1—Q5的纯函数参考核，`verify_mechanisms.py` 生成JSON测试记录。本轮实际执行了确定算例、5000组供给随机样本、3000组本地DR/服务样本及96时段聚合移峰账本。另用三消费者96时段参考驱动器按执行顺序串联全部纯函数，产生288条服务记录。完整FIFO期限队列、原子提交、接口训练与配电网潮流属于下一步工程验收范围，见第25节。

实施顺序仍为：先固定三消费者动作验证PAB及执行账本；再跑连续多时段、期限队列和边界条件；最后接入PPO/MAPPO并执行训练与偏离实验。

---

# 第五部分　论文实验、理论验证与边界

## 21. 对比实验与消融设计

以原论文固定价格 DR 框架作为参考，设计以下实验：

| Case | UC 优化 | 消费者动作 | 市场机制 | 对照目的 |
|---|---|---|---|---|
| A | 固定供给计划 | 原论文式 DR 与 UC/DER 用电调整 | 固定 TUTT/DER 价 | 参考基线 |
| B | 固定供给计划 | DR + 两类报价 | Pay-as-Bid | 单独观察消费者议价和市场竞争 |
| C | UC Actor | DR + 两类报价 | Pay-as-Bid | 完整 Stackelberg–Nash 市场 |
| D | UC Actor | DR + 两类报价 | Uniform Price | 定价机制消融 |
| E | UC Actor | DR + 两类报价 | Pay-as-Bid | 分别删去两类公开容量观测 |
| F | UC Actor | DR + 两类报价 | Pay-as-Bid | 独立 Actor 与共享局部 Encoder 对比 |

此外，设置资源富余、中度稀缺和高度稀缺三个工况，分别调整 \(P_t^{r,\mathrm{avail}}\) 与 \(P_t^{u,\mathrm{avail}}\)，检验不同稀缺程度下报价竞争与 DR 行为的变化。

### 21.1 能源与 DR 指标

- PG 购电 Peak-to-Average Ratio：\(PAR=\max_t p_t^{b,\mathrm{exec}}/(T^{-1}\sum_t p_t^{b,\mathrm{exec}})\)；
- PG 峰值、日峰值和爬坡幅度；
- DER 消纳率：\(\sum_t(p_t^g-p_t^{a,\mathrm{exec}})/\sum_t p_t^g\)；
- 各消费者实际削减量、移出量、真实回补量、待办/逾期能量、月累计 UC 购电量；
- 事件核验正响应、获补偿DR量和采购预算使用率，分别记录物理量与补偿量。

### 21.2 市场与成本指标

- UC、DER 市场成交量、容量利用率；
- UC/DER 平均加权中标报价；
- 各消费者 PAB 购电成本、DR 激励和总成本；
- UC 收入、PG 购电成本、ESS/弃电成本、总收益；
- 市场未成交、备用、当期请求缺口、原生实际损失、未完成回补请求及后续待办余额（分别报告）；
- 备用回收款、外部服务成本、UC自担调用成本、原生损失成本、任务等待/逾期成本与实付补偿；
- 不同消费者的成交份额和资源获得差异。

### 21.3 算法与博弈指标

- 各消费者与 UC 的 episode return；
- 训练 wall-clock、环境步数、单次出清耗时；
- 多随机种子均值、方差和置信区间；
- 消费者单边最佳响应收益差：

\[
\epsilon_i^{NE}=
J_i(\pi_i^*,\pi_{-i}^*;\pi_u)
-\inf_{\pi_i'}J_i(\pi_i',\pi_{-i}^*;\pi_u),
\]

- UC 响应偏离指标：对每个候选 UC 重新训练消费者，计算 \(\widehat\epsilon_u^{SE}\)；
- follower/leader 检验计算预算、候选数量、独立测试种子及收益置信区间。

使用明确的单边偏离搜索预算和固定评估随机数种子报告近似均衡质量。

## 22. 可解性与收敛分析的组织方式

### 22.1 市场、补供与账本层

固定合法输入、事件日历和tie-break规则后，两池排序、ESS放电下调、PG补供、原生优先服务与比例DR采购都对应确定映射。本版可以直接验证四组代数性质：

\[
 \begin{aligned}
 &p_t^{b,\mathrm{exec}}+p_t^g+p_t^{o,\mathrm{exec}}
 =p_t^{a,\mathrm{exec}}+p_t^{c,\mathrm{exec}}+\sum_ip_{i,t}^{\mathrm{served}},\\
 &d_{i,t}=p_{i,t}^{\mathrm{served}}+
          \ell_{i,t}^{\mathrm{native}}+q_{i,t}^{\mathrm{repay,unfilled}},\\
 &B_{i,t+1}=B_{i,t}+s_{i,t}^{+}\Delta t-r_{i,t}^{\mathrm{act}}\Delta t,\\
 &0\le\sum_iq_{i,t}^{\mathrm{paid}}\le p_t^m.
 \end{aligned}
\]

同时，备用直接收款恰好覆盖对应采购与外部服务，UC的额外调用成本为自身支出。单时段映射的守恒与有限性可以在上述可行计划、本地动作范围和线性成本设定下逐项推导。

当前物理能力支持多少原生服务、任务能否按期完成，由实际资源和后续动作决定。论文分别报告代数守恒、原生失供以及跨期任务履约，再对加入预测可行域的扩展讨论递归履约条件。

### 22.2 消费者层

给定 UC 策略和 Pay-as-Bid 规则，消费者构成一般和动态随机博弈。理论部分可先研究单时段离散报价格点构成的有限博弈：其混合策略 Nash equilibrium 在常规有限博弈条件下存在；动态版本需结合状态空间、策略类和转移核增加相应假设。由实验的 best-response 搜索量化已训练策略的近似 Nash 性质，并在偏离测试中完整重新执行 PAB、成本备用、任务服务和事件DR采购。

### 22.3 UC 层

UC 目标是对下层均衡响应求最优：\(\max_{\pi_u}J_u(\pi_u,\pi_c^*(\pi_u))\)。对下层响应唯一、连续并且上层策略集紧等条件成立的简化版本，可讨论 Stackelberg 解存在；采用双时间尺度及局部稳定条件时，再分析理论学习动态的局部收敛。深度 PPO/MAPPO 的实际训练表现通过回报曲线、策略变化与偏离检验共同报告。 UC 候选策略的评估包含消费者重新适应步骤，在相同外生数据上比较完整响应收益。

---

# 第六部分　确认记录、核验与后续需要考虑的内容

## 23. 本轮模型与实现变更

| 位置 | v5统一后的关系 |
|---|---|
| 供给执行 | \(p^{o,exec}=\min(p^{o,plan},U+p^{e,plan})\)，下调量留在ESS |
| PG补供 | 先算基础PG，再用剩余物理能力；最终PG覆盖普通与备用 |
| 备用费用 | 消费者支付可核验PG增量成本＋外部服务；UC自担调用代价 |
| \(p_t^m\) | 只约束获补偿DR采购量；物理DR按本地可行域执行 |
| 移峰服务 | 当前原生需求优先；真实回补扣队列；未完成继续保留 |
| 事件激励 | 基线固定、事件门控、净正响应核验、比例采购 |
| 原生失供与移峰等待 | 两项分开计量、分开进入费用和后续状态 |
| 月累计 | 普通UC与PG备用电量一同累计 |
| 时序 | 执行结果→激励核验→结算→原子状态提交 |
| 工程 | 新增`incentive.py`和对应采购记录，更新服务及队列契约 |
| 学习 | 两层PPO候选/重新适应流程保持，所有收益读取新版账本 |

这些变化将作者确认的五项分支写入整篇模型，而保留原有10维观测、四维消费者动作、两个资源池和双层策略训练。

## 24. Q1—Q5 已确认选项与核验例

| 确认项 | 本版核验 |
|---|---|
| Q1=A | 计划ESS放电4 kW、普通成交2 kW：实际放电2、PG0，守恒 |
| Q2=A | 备用2 kW、15分钟，PG价0.50、外部服务0.10：回收0.30元；UC调用代价0.10元 |
| Q3=B | 实际削减3＋3 kW，采购上限4 kW：需求按6 kW减少，补偿份额2＋2 kW |
| Q4=A | 待办2 kWh，计划回补2、实际回补1：余额保留1 kWh |
| Q5=B | 事件外和纯回补的DR补偿为0；纯缺供的核验DR为0 |

本轮参考核的运行结果见 `verification_results.json`。5000组随机供给样本全部通过所列功率和资金关系，其中609组进入ESS放电下调；3000组本地DR/服务随机样本通过需求与移峰守恒；96时段聚合移峰序列实现先延期、部分回补遇缺供、后续继续完成的余额记录。三消费者96时段参考驱动器进一步串联PAB、Q1、补供、Q4、Q3/Q5与Q2，逐步核对供电、服务与资金。测试为合成输入的机制核验，逐项范围写在随附审阅报告。

## 25. 后续需要考虑的内容

1. **成本备用与报价竞争**：成本回收保证备用直接账本按成本流转。若普通成交价低于PG成本，UC仍可能通过减少亏损普通订单改变收益；后续采用完整偏离实验、服务义务或经过论证的费率约束研究此问题。
2. **消费者转向备用的选择**：成本备用价可能低于部分普通报价或TUTT底价。价格排序、备用合同容量和用户服务损失应共同做敏感性分析，观察主动竞争和依赖补供的策略分布。
3. **10维观测的部分信息**：当前保留用户指定的输入。备用实际价、移峰余额、期限和动态备用余量存在于全局状态，后续比较历史编码与追加字段对响应质量的影响。
4. **DR采购份额的选择**：本版按核验量比例分配。后续可以比较先签约配额、核验后比例分配、带服务质量权重的分配；\(p_t^m\) 继续表示采购功率上限。
5. **事件和基线标定**：本版使用固定外生基线和预定事件。正式数据需要标明基线误差、正响应可计量性，以及消费者改变历史负荷时的核验机制。
6. **激励与后续履约的联系**：事件内确认的削减/移出获得补偿，回补按购电计费。研究可进一步引入完成条件、期限服务责任、补偿预算及追踪证据，评估整个事件周期的利益分配。
7. **任务队列和期限**：下一阶段工程实现FIFO任务、超期标记、日/月跨界继承和终端待办责任。多时段测试需覆盖连续供紧缺造成的长期积压，以及最终状态可达性。
8. **原生优先规则**：本版确定先服务当前原生需求。任务优先级、关键负荷等级和不同截止时间可作为明确的服务规则扩展，并检验公平性和可靠性。
9. **预测可行性**：当前单时段规则保证账本闭合并显式记录缺口。跨时段零违约保证需要结合未来资源、回补上限和备用合同建立可达或鲁棒可行域。
10. **ESS富余处理**：本版允许出清后下调计划放电，充电分量保持计划。后续可比较富余充电、向外售电或联合再调度，并重新定义UC承诺与补救权限。
11. **真实网络与扰动**：参考核验证聚合节点。后续加入馈线电压、潮流、网损、爬坡以及公告到执行之间的DER变化，建立对应的物理校验与补救模型。
12. **报价与出清边界**：保留两池DER优先及单段报价。后续比较统一联合出清、数量报价、资源选择和不同平局规则，并报告经济效率及市场势力。
13. **DR代价函数**：当前采用净主动调整的抽象平方代价。后续可将削减、移出和真实回补分别定价，校准零原始负荷时的分母尺度，检验等待成本与调整成本的组合。
14. **UC动作参数化**：七字段计划中的充电恒等式和模式互斥由动作解码完成。策略原始自由度、动作映射和PPO概率密度需要联合单元测试；容量级动作可作为对照。
15. **均衡与训练范围**：两层PPO产生候选并做消费者重新适应。精确SNE、局部响应梯度和有限预算偏离证据分别报告，市场与激励的完整结果共同决定最佳响应。
16. **完整实现验收**：本轮纯函数测试覆盖静态出清、供给、费用、局部DR和聚合余额。FIFO期限队列、单次提交、并行采样、断点续训和训练算法按第20节继续验收。
17. **数据与案例**：保持抽象负荷、UC/DER/ESS场景；逐步建立可公开复现的负荷/光伏/电价序列、事件配置和成本参数。案例区分普通成交、备用补供、真实原生损失和任务待办。

## 26. 依据与版本追踪

**[D1] 主修文档**：`Stackelberg_Nash_Pay_as_Bid_Modeling_ZH_v4_review.md`。v4原件保留，SHA-256记录于本轮修改报告。

**[D2] 作者确认**：Q1=A、Q2=A、Q3=B、Q4=A、Q5=B。第3、4、7、8、9节及工程契约依据这一组合修订。

**[D3] 本版实现约定**：比例DR采购、原生优先且FIFO回补、外生基线下实际净正响应；均已在正文与配置显式定义。

**[R-PPO]** Schulman et al. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347。沿用上一版训练依据：[原文](https://arxiv.org/abs/1707.06347)。

**[R-IFT]** Fiez, Chasnov & Ratliff. *Implicit Learning Dynamics in Stackelberg Games*. ICML 2020。沿用上一版局部响应梯度分析依据：[原文](https://proceedings.mlr.press/v119/fiez20a.html)。

**[R-SP]** Brero et al. *Stackelberg POMDP*. arXiv:2210.03852。沿用上一版跟随者响应过程依据：[原文](https://arxiv.org/abs/2210.03852)。

本轮新增公式属于已选机制的推导，新增算例来自随附参考核。文献段落保留上一版整理的适用范围；本次测试输出描述机制核验与软件样例，不填入尚待实验获得的性能和收敛结果。
