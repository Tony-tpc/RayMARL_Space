# 能源市场强化学习数据选型

调研日期：2026-10-19。适用对象：当前六维 UC、四维 Consumer、共享 DER／ESS、聚合柔性负荷与 FIFO 移峰模型。

## 1. 推荐方案

**主实测基准优先选择 Ausgrid Solar Home Electricity Data；以 ResStock 的实际气象年基准场景开展 15 分钟和大规模扩展实验。** Ausgrid 同时测量住户总用电与光伏，总量模型可以直接利用，无需新增家电动力学。原始分辨率为 30 分钟，按该步长训练最清楚。[数据作者的机构记录](https://researchportalplus.anu.edu.au/en/publications/residential-load-and-rooftop-pv-generation-an-australian-distribu/)

Ausgrid 的旧官方发布页已下线，当前可查机构副本及第三方镜像。选用前应取得来源可追溯的文件和原始使用说明，核对列定义与哈希；不要把第三方代码许可证当作原始数据许可证。如果要求立即从明确开放渠道取得数据，优先启动 ResStock 仿真基准，并把 Ausgrid 列为实测补充。Pecan Street 大学版适合保留 15 分钟的实测方案，但需先确认账户权限、同地区共同覆盖期和完整户数。

当前工作是在线环境交互训练。数据提供负荷、发电、价格等外生场景，报价、削减、移峰、成交与奖励由策略和环境产生，不需要“最优行动标签”。本次调研未改训练或环境代码，也未下载完整数据集。

## 2. 候选比较

| 数据集 | 数据性质与规模 | 对本项目的用途 | 主要取舍 |
|---|---|---|---|
| Ausgrid Solar Home | 300 户，2010-07 至 2013-06，30 分钟，负荷与光伏分别实测 | 首选实测社区基准 | 原发布渠道变化；没有设备期限和真实柔性偏好 |
| Pecan Street Dataport | 同户负荷、光伏和设备回路；秒／分钟级 | 15 分钟实测及柔性分项校准 | 公开样例仅 10 户、3 天；大学扩展样本最多 75 户，需申请并核共同覆盖期 |
| ResStock / EULP | 实测校准的物理仿真，约 55 万住宅模型，一年 15 分钟分项用能 | 32／128 户及跨气候扩展；立即可用的仿真主基准 | 应选逐栋基准序列，不能称为真实表计记录；PV 与价格要另行匹配 |
| London Smart Meter | 5,567 户，2011-11 至 2014-02，30 分钟 | 大规模真实负荷对照 | 无成套分项／PV；优先非动态电价组作为固定历史参考负荷 |
| REFIT / UK-DALE | REFIT 20 户、8 秒；UK-DALE 5 户、常用低频约 6 秒，含设备功率 | 校准可调整电量的量级，设备级补充检查 | 户数少、缺少成套社区光伏与采购价格 |
| UMass Smart* | 多个不同子集：114 公寓长期负荷、部分家庭回路、独立光伏子集 | 外生负荷和光伏鲁棒性对照 | 不同子集不能默认同户同地同时；443 户子集仅 24 小时 |
| OPSD Household / CoSSMic | 德国 11 个计量站点，包括住宅和非住宅；提供 15 分钟格式 | 小规模独立实测对照 | 只有其中 6 个是住宅；需分清电网进／出、PV、累计表计及补缺标记 |
| CityLearn 数据 | 按具体发布版本组织建筑、气象、价格、PV 等 | 算法环境外部基准 | 建筑热需求、储能任务与当前交易／FIFO模型不一一对应，不直接复制动作和奖励 |

来源：[Ausgrid 机构说明](https://unsw-ceem.github.io/DARTH-docs/ausgrid300/)、[Pecan Street 免费数据范围](https://www.pecanstreet.org/2025/07/public-data/)、[EULP 官方说明](https://www.nlr.gov/buildings/end-use-load-profiles)、[London Datastore](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d)、[REFIT](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/)、[UK-DALE 作者论文](https://arxiv.org/abs/1404.0284)、[Smart*](https://traces.cs.umass.edu/docs/traces/smartstar/)、[OPSD 数据字典](https://data.open-power-system-data.org/household_data/2020-04-15/)、[CityLearn 数据格式](https://www.citylearn.net/overview/dataset.html)。

Pecan Street 的大学免费扩展样本不等于完整 Dataport 数据库，也不能保证 75 户都有全年光伏和所有子回路。REFIT 已清洗数据的部分长空缺被填零，需查看原始缺失处理；OPSD 也显式标注了插值数据。使用前按版本冻结质量规则。[Pecan Street 访问条件](https://www.pecanstreet.org/access/)、[REFIT 原论文](https://strathprints.strath.ac.uk/58873/8/Murray_etal_SD_2016_An_electrical_load_measurements_dataset_of_United_Kingdom.pdf)、[OPSD 原始说明](https://data.open-power-system-data.org/household_data/2020-04-15/README.md)

## 3. 原论文使用了什么

Ye 等 2021 年论文使用 Ausgrid 的 2011-07-01 至 2012-06-30 数据，300 户、30 分钟，前 11 个月训练、最后 1 个月测试。柔性设备运行参数取自文献，部分需求时间与初始状态另外采样。实际负荷与仿真设备参数在该研究中也是分开构造的。[原论文](https://ieeexplore.ieee.org/document/9509579)

Nie 等 2024 年论文使用国网湖北提供的 2021 年三栋办公楼、一个商场、一所大学、三个居民社区数据，每日 96 点；另用 2022 年两个月验证。这些 Consumer 是不同规模的建筑或社区，该案例段未提供可直接下载的公开数据地址。本地核查依据为 `.tmp/model-review/paper_units_extract.txt` 第 1763 行起和原 PDF 第 10—11 页；Ye 数据说明见 `.tmp/model-review/ye2021_units_extract.txt` 第 1667 行起。

Ye 原文引用的补充数据 DOI 为 `10.21227/80dj-hs08`。另外可检索到 [IEEE DataPort 的同题 Supplementary Dataset 条目](https://ieee-dataport.org/documents/supplementary-datasetascalableprivacy-preservingmulti)，其 DOI 是 `10.21227/tv4k-9t38`，列有约 29.23 MB 的 XLSX。两条记录应按各自元信息核对，尚未下载核验其工作表，不能直接认定其中含完整三年原始数据。

## 4. Ausgrid 与当前输入的映射

原始 GC 为普通用电，不含 CL；CL 为单独计量的受控用电；GG 为光伏总发电。三者原始数值均为半小时电量 kWh。[Ausgrid 原始说明 PDF 的镜像](https://raw.githubusercontent.com/rockey-abhishek/Ausgrid-Dataset/main/Ausgrid%20solar%20home%20electricity%20data%20notes%20%28Aug%202014%29.pdf)

| 当前字段 | 建议构造 |
|---|---|
| `loads_kw[t,i]` | `(GC[t,i] + CL[t,i]) / 0.5`；无受控计量与缺失记录要分别识别 |
| `der_kw[t]` | `sum_i GG[t,i] / 0.5`，可按预先定义的装机规模调整 |
| `rigid_kw[t,i]` | 明确的场景参数或设备分项标定结果 |
| `flex_kw[t,i]` | 本地调节能力参数；历史 CL 不是该能力的直接测量 |
| `grid_cost_per_kwh[t]` | UC 上游采购费率，独立于消费者阶梯零售价 |
| SOC、任务队列、月累计 | 环境的持续状态；按实验初态或预热轨迹建立 |

GG 聚合成共享池是本文的资源归属设定，并非原始数据中的集中产权。将“电网净购电量”直接作为负荷再加入 PV 会重复计算光伏；有本地电池的其他数据集还需分离电池充放电。

历史用电是给定行为和原价格下的观测轨迹，在本模型中作为固定参考负荷；它不是已识别的无 DR 反事实。最大削减、移峰期限、队列容量、不适／迟延系数继续由模型参数给定，以敏感性实验检验。当前 20% 柔性只是合成配置，不能写成数据集实测结论。

### 获取与时间处理

Ausgrid 旧官网页面及原 ZIP 链接在本轮查询中失效。[UNSW DARTH](https://unsw-ceem.github.io/DARTH-docs/)有机构托管，但访问要求 UNSW 身份。[第三方分析仓库](https://github.com/pierre-haessig/ausgrid-solar-data)提供数据来源线索，其 [LICENSE](https://github.com/pierre-haessig/ausgrid-solar-data/blob/master/LICENSE)明确排除原始 Ausgrid 数据，不能将代码的 CC BY 4.0 延伸到原始数据。原始说明表述为研究和政策分析公开释放；分发原始副本前需核对对应使用条件。

历史原始说明使用 EST／夏季 EDT，表头表示区间结束时刻，最后 `0:00` 指下一日零点。接入时统一到 UTC 或固定标准时，并记录夏令时转换日如何处理；不能仅根据现代网站的通用表计时间说明推定这份历史文件全年不换时。

## 5. 步长及价格

Ausgrid 主实验建议使用 30 分钟，不把插值后的 15 分钟当成新增实测信息。保持物理含义时，需要同时协调以下配置：

| 项目 | 原 15 分钟 | 30 分钟 |
|---|---:|---:|
| `resolution_minutes` | 15 | 30 |
| `dt_hours` | 0.25 | 0.5 |
| 24 小时移峰期限 H | 96 | 48 |
| 三天有限时域 T | 288 | 144 |
| 17:00—20:00 每日事件索引 | 68—80 | 34—40 |
| 同物理时长折扣 | 0.99 | 0.9801 |

功率额定值、能量容量、电量单价保持原物理单位，结算公式已有一次时长乘法。按小时计的等待成本率不变；UC 按次到期考核若要保持每小时强度，应随时长同比调整。观测维数随 H 改变，因此使用对应规格重新训练网络。

最简单的主实验先固定一套明确、非负的 UC 采购 TOU 费率，消费者阶梯价、DER 报价底价和各项成本分别记录。这构成“实测负荷／光伏驱动的机制实验”。同年 [AEMO NSW 价格](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data)适合作为价格扩展，2010—2013 对应原半小时结算；[五分钟结算于 2021-10-01 实施](https://www.aemo.com.au/initiatives/major-programs/past-major-programs/five-minute-settlement)。AEMO 时间口径、货币和采购层级都要匹配。

当前 `Profiles` 对价格也要求非负，真实负现货价不能直接导入，亦不能悄悄裁零。若研究负价，应先明确负价采购模型；这不是本轮接入现有模型的必要步骤。AUD/MWh 转 AUD/kWh 除以 1000；若统一为人民币，应同时转换其余所有金额系数，不能只改 PG 价格。

## 6. 训练划分及规模

主实验先固定 16 或 32 户，经统一质量规则选定同一区域、共同可用日期，再按时间划分：2010-07—2011-06 训练，2011-07—2012-06 验证，2012-07—2013-06 最终测试。这个方案是项目建议，不沿用 Ye 的单月测试设计。若只拿到一年，应预先固定日期块并报告季节覆盖范围。

先划分日期再切窗口；训练与测试窗口不重叠，补缺和参数拟合不能利用测试结果。候选比较使用相同日期和随机采样种子，统计区间按独立日期块计算。不同 seed 如果选中了同一批真实窗口，并不构成独立数据切分。

月阶梯价与跨日移峰采用完整月份或长连续轨迹。短 `rollout` 可以用于更新网络，不能因此每天重置队列、SOC 和月累计。有限期须满足时域足够长，避免 `T<=H` 令新移出功能关闭。

当前每户使用独立 Actor，未见日期测试是直接支持的泛化范围。新用户测试需要另外规定迁移、重新训练或共享参数方案，不能把替换 `consumer_i` 数据当作已完成跨户泛化。

设备和市场尺度按训练集标定：例如 PV 年能量占比、ESS 可支撑平均负荷的小时数、PG 限额相对社区负荷分位数、DR 上限相对可调负荷。当前 8 个合成 Consumer 的 100 kW PG／200 kWh ESS 不能自动视为适合 8 个真实家庭。所有测试场景固定训练阶段选定的尺度。

## 7. ResStock 扩展及最小接入边界

ResStock 选固定发布版的 **AMY2018、baseline／upgrade 0、同一地理区域、逐住宅序列**。不要把县级汇总复制成多个家庭，不要把统计抽样权重乘到单个 Consumer 的负荷上，不要把已应用需求灵活性改造的升级场景当未干预基线。AMY 保留真实气象年同步性；不同地点的 TMY 拼接天气不保证同日同步。[官方版本目录](https://resstock.nlr.gov/datasets)、[公开数据结构](https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/README.md)、[公开存储和许可入口](https://registry.opendata.aws/nrel-pds-building-stock/)

该数据的负荷是仿真结果；共享 PV 应选择同地点、同气象年的输出或发电模型，并标注来源。分项用能可约束柔性量级，但用能量不等于允许削减／延迟的权限。

当前 `MarketTrainingEnv(config, profiles)` 已支持注入真实 `Profiles`。缺口在 `MarketEnvFactory` 固定调用合成函数，以及 CLI 尚无真实数据清单入口。后续最小实现放在 `experiments`：数据转换、固定用户／日期清单和工厂；沿用现有算法与环境接口。工厂契约绑定原始与处理后哈希、单位、时间口径、用户顺序、窗口切分和参数。无需为数据集增加算法分支。

当前观测使用本期已知负荷、DER 与价格，主实验沿用这一信息时序。预测误差实验需要另外区分预测值、实际值和揭示时刻；历史曲线本身并不构成预测模型。

中国案例不能仅靠使用人民币把海外家庭称作中国家庭。广西公开住宅数据是小时级变压器／社区汇总，更适合社区作为 Consumer 的研究；新公布的中国住户净功率数据也需先核验是否有单独 PV 与原始采样。两者均不能未经处理直接替代当前逐户总负荷加共享 PV 的输入。[广西数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11920200/)、[中国住户净功率数据论文](https://doi.org/10.1038/s41597-026-08508-1)
