# 当前状态与交接

更新：2026-10-20。

当前实现使用六维 UC、四维 Consumer 行动及既有环境奖励。训练从月初开始，普通 64 步批末和外层轮次保留物理状态；真实月末结算后切换月份。每批更新所有 Critic，再更新该阶段选中的 Actor。Critic 两层隐藏网络加入 LayerNorm，UC Value 另有从既有时间观测派生的四维线性支路，Actor 结构不变。

Actor 使用 `lambda_gae=0.9025`，Critic 使用独立 `critic_lambda=1.0` 的批内多步回报。两套目标采样后固定。额外 MC 预拟合、月中规则暖启动和旧导入别名已删除。

候选响应保持公共消费者、Adam、环境和 RNG 起点。候选 0 也与轮初已提交联合策略比较失供比例、绝对失供量及 UC 现金，通过后沿用配对 UC 回报规则。无可提交候选时恢复轮初全部状态；预算和批号不回退、不补跑。能源检查在实验回调中实现，仅是固定验证窗口上的经验约束。逐批日志增加目标分解、预测分布、饱和比例与梯度诊断。

## 入口与证据

- [HTML 阅读版](../../environments/sn_energy_market/docs/v9/modeling_complete_zh.html)：离线目录与已渲染的 SVG 公式。
- [当前建模](../../environments/sn_energy_market/docs/v9/modeling_complete_zh.md)：第 14 节对应实现。
- [训练协议](TRAINING.md)：接口、快照、预算、日志及数据隔离。
- `batch_critic_pilot.py`：当前默认 3 轮、24 批；每批 Critic 4 epoch、Actor 2 epoch、minibatch 32。
- [Critic 稳定性实验](CRITIC_STABILITY_PILOT.md)及 `results/critic_stability_20261020/`：新结构、候选筛选与三轮实测诊断。
- `evaluation.py` / `training_audit.py`：冻结评价与被动审计，现行入口不依赖旧实验脚本。
- [中长期报告](YEAR_STUDY.md)及 `results/year_study_20261020/`：六个固定验证节点、分日期结果和 1–576 批实测曲线。
- [小规模基线](BATCH_CRITIC_PILOT.md)及 `results/batch_critic_20261020/`：真实逐批 loss 和新旧对比。
- [直接对照](ONLINE_CRITIC_PILOT.md)：提交 `ea3d322`；小规模基线运行来源见 `ed56de0` 及产物哈希。

修改前的 72 轮结果由 `8509e88` 定位，已完整结束。相对第 3 轮，UC 全月 RMSE 从 17.5153 降至 14.9314，中旬／月末降低，月初从 14.1151 升至 20.2190。消费者平均 RMSE 为 5.2078，较第 48 轮反弹。完整月供电率 99.6467%，失供量 90.7091 kWh，UC 现金 38.0646 AUD/日，较第 3 轮略降 0.58%。这些结果不代表本次修改后的算法。

## 保留范围

`outputs/ausgrid-critic-stability-pilot/20261020/` 保留本次 24 批、初始／最终验证、checkpoint、Actor 和失败／恢复证据；`outputs/ausgrid-year-study/20261020/` 保留旧 72 轮完整报告；`outputs/ausgrid-batch-critic-pilot/20261020/` 保留旧三轮基线；`outputs/ausgrid-online-critic-pilot/20261020/` 保留更早直接对照；环境完整验证保留 `outputs/snmarket-validation/full-current-33f27b7-20261018/`。历史输出的来源哈希不改写。

本次三轮全部拒绝，共消耗 24 批、1,536 个 PPO 时段，每个 Critic 执行 192 次优化，但保留更新和主轨迹进度均为零。最终验证与新结构初值完全一致：UC 全月 RMSE 35.1674、消费者平均 8.7431，未实现拟合改进。两个候选在验证窗口均增加失供并减少现金；完整回滚使后两轮重复同一尝试。当前方案首先需要解决“拒绝后无有效推进”，不能据此继续扩大预算。

首轮回滚完成后发生 NumPy 布尔值的 JSON 序列化错误，已在实验回调中转换原生数值并补充检查。根据八条完整回滚记录重建 seed 42 初态，仅执行剩余 16 批；两次后续尝试的全部逐批诊断与首轮一致，原失败证据保留。详细候选检查数值只保存了第 2、3 轮；不伪造第一轮缺失的逐步 Adam 参数变化记录。125 项相关检查、修复后的 18 项检查及最终 checkpoint 精确重放通过，测试集未加载或评估。
