# 当前状态与交接

更新：2026-10-20。

当前实现使用六维 UC、四维 Consumer 行动及既有环境奖励。训练从月初开始，普通 64 步批末和外层轮次保留物理状态；真实月末结算后切换月份。每批更新所有 Critic，再更新该阶段选中的 Actor；候选响应恢复公共消费者与物理/RNG 状态，最后保留选中完整响应分支。候选仍按配对 UC 回报选优，未实现供电率不退化约束。

Actor 使用 `lambda_gae=0.9025`，Critic 使用独立 `critic_lambda=1.0` 的批内多步回报。两套目标采样后固定。额外 MC 预拟合、月中规则暖启动和旧导入别名已删除。

## 入口与证据

- [当前建模](../../environments/sn_energy_market/docs/v9/modeling_complete_zh.md)：第 14 节对应实现。
- [训练协议](TRAINING.md)：接口、快照、预算、日志及数据隔离。
- `batch_critic_pilot.py`：固定 3 轮、24 批、1,536 PPO 时段、768 主轨迹时段；Critic 4 epoch、Actor 2 epoch、minibatch 32。
- `evaluation.py` / `training_audit.py`：冻结评价与被动审计，现行入口不依赖旧实验脚本。
- [最新报告](BATCH_CRITIC_PILOT.md)及 `results/batch_critic_20261020/`：真实逐批 loss 和新旧对比。
- [直接对照](ONLINE_CRITIC_PILOT.md)：提交 `ea3d322`；最新运行来源见 `ed56de0` 及产物哈希。

最新 UC 全月 RMSE 为 17.5153，月初改善，中旬和月末略升；消费者平均 RMSE 为 5.4321，较上一轮上升。随机策略验证窗口供电率 99.7279%、UC 现金 35.3490 AUD/48h；均值动作供电率略降。本预算只训练到月内第 17 日，不能判断月末学习或收敛。

## 保留范围

`outputs/ausgrid-batch-critic-pilot/20261020/` 保留最新报告、checkpoint、Actor 和全套曲线；`outputs/ausgrid-online-critic-pilot/20261020/` 保留直接对照；环境完整验证保留 `outputs/snmarket-validation/full-current-33f27b7-20261018/`。历史输出的来源哈希保持当时内容，整理代码后不改写这些哈希。

当前测试保留数学目标、概率与梯度、冻结 Adam、分支隔离、月末终止、checkpoint 重放、数据切分、物理与账本守恒检查。清理记录见 [REPOSITORY.md](../../REPOSITORY.md)。本轮只整理与验证代码，未增加真实数据训练，保留测试集未加载或评估。
