# 代码库维护说明

算法放在 `RLAlgo/src/raymarl/algorithms/`，可复用组件放在 `common/`；环境负责动作、状态、物理和结算；实验负责数据、训练、审计、评估和输出。各算法从自己的包导入，公共回放从 `raymarl.common.replay` 导入。

## 当前能源市场入口

| 入口 | 用途 |
|---|---|
| [建模文档](environments/sn_energy_market/docs/v9/modeling_complete_zh.md) | 唯一维护的完整模型规格，含当前训练算法 |
| [训练说明](experiments/sn_energy_market/TRAINING.md) | 通用参数、连续采样、候选恢复、日志及评价 |
| [固定预算入口](experiments/sn_energy_market/batch_critic_pilot.py) | 当前 Critic 结构与候选检查，默认 3 轮、24 批 |
| [Critic 稳定性实验](experiments/sn_energy_market/CRITIC_STABILITY_PILOT.md) | LayerNorm、UC 时间支路及供电／现金检查的固定预算对照 |
| [小规模结果](experiments/sn_energy_market/BATCH_CRITIC_PILOT.md) | 分日期 RMSE、供电率、UC 收益与真实逐批曲线 |
| [中长期验证](experiments/sn_energy_market/YEAR_STUDY.md) | 72 轮固定预算，沿用同一算法、训练年与保留测试集 |
| [HTML 阅读版](environments/sn_energy_market/docs/v9/modeling_complete_zh.html) | 从当前 Markdown 生成的离线目录与 SVG 公式 |
| [交接说明](experiments/sn_energy_market/HANDOFF.md) | 当前实现、证据位置与未解决问题 |

当前训练入口直接依赖 `evaluation.py` 和 `training_audit.py`，不再从历史实验脚本导入函数。数据场景每个 seed 对应一个完整月，月初启动、批间保留状态、真实月末重置。

## 删除和保留规则

2026-10-20 清理了旧 MC 预拟合、月中规则暖启动及相应测试；删除 `raymarl.ppo/networks/learner/replay` 四个转发模块，调用方使用实际实现路径。旧接口名称是否存在的测试已删除，交易失败不提交状态的检查保留为 `test_transaction_integrity.py`。Value 尺度、初始化及在线冻结检查集中在 `test_stackelberg_learner_phases.py`。

旧 v5/v7/v8 模型、过时 HTML、文档修改过程校验、早期实验脚本和结果摘要从当前目录移除。历史源码与报告可通过 Git 查看；直接对照定位 `ea3d322`，小规模固定预算结果定位 `ed56de0` 及报告内来源哈希。历史 checkpoint 继续对应原代码与数据契约，当前入口不增加迁移兼容层。

本地删除 22 个过时输出目录，共 2,185 个文件、291.49 MiB，包括早期 Ausgrid 训练、旧预拟合、连续试验、诊断，以及被完整复验替代的早期环境验证和训练 smoke。保留：

- `outputs/ausgrid-year-study/20261020/`：72 轮训练、固定节点验证、完整 checkpoint 与逐批日志。
- `outputs/ausgrid-batch-critic-pilot/20261020/`：小规模原始结果、checkpoint、Actor、逐批日志和全部曲线。
- `outputs/ausgrid-online-critic-pilot/20261020/`：直接比较基线。
- `outputs/snmarket-validation/full-current-33f27b7-20261018/`：完整环境验证基线。
- `experiments/sn_energy_market/results/year_study_20261020/`：72 轮机器摘要、576 批曲线数据与三张关键 PNG。
- `experiments/sn_energy_market/results/batch_critic_20261020/`：纳入 Git 的结果摘要、逐批数值和关键曲线。
- 数据集、虚拟环境及其他算法的有效实现和验证。

动作可行域、物理守恒、账本一致性、有限值检查、数据划分及快照契约仍属于必要检查。候选恢复和冻结评价的 `finally` 负责隔离环境与 RNG；删除这些保护会改变算法行为。

## 清理提交 b0a81e7 的验证

清理后通过 328 项非集成测试、5 项本地 Ray CPU 集成测试。集成场景使用微型合成数据，覆盖候选选择、冻结 Adam、主轨迹接续、checkpoint 重放，以及 24 批日志和提取后的审计模块。

当前固定预算配置与直接对照的学习预算、物理配置和数据切分一致；环境生产源码未修改。保留原始报告重新计算的对比与已提交摘要仅有浮点求和末位差异，最大绝对差小于 `3e-13`。历史结果、日志和图件未重写。

该清理提交未重新训练真实数据、未扩大训练预算、未加载或评估保留测试集，也未重跑百万级环境压力测试。静态检查与当前文档本地链接检查通过。

## 后续 72 轮验证与 HTML 交付

已按用户确认预算完成一次 72 轮训练及六个固定节点验证。前 24 批精确复现小规模基线；完整训练年、12 次真实月末、全部 Critic 更新及 checkpoint 重放均已核对。本次没有修改算法或环境生产代码，也没有新增测试文件；实验入口支持显式轮数，整月评价复用已有 MC 轨迹累计能源指标，绘图保留全部实测批次。当前 v9 HTML 作为从 Markdown 生成的阅读交付物保留，公式离线渲染为 SVG；历史过时 HTML 不恢复。详见[中长期结果](experiments/sn_energy_market/YEAR_STUDY.md)。

后续 Critic 稳定性修改在现有测试文件中补充归一化、Actor 初始化一致、候选 0 检查、整轮回滚与 JSON 数值覆盖，没有新增兼容层。24 批实验的三个候选选择轮全部拒绝，最终模型保持新结构初态；首轮日志故障及预算内恢复过程保留。新证据位于 `outputs/ausgrid-critic-stability-pilot/20261020/` 和 `experiments/sn_energy_market/results/critic_stability_20261020/`，见[报告](experiments/sn_energy_market/CRITIC_STABILITY_PILOT.md)。环境生产源码与测试集保持不变。
