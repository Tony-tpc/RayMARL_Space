# 当前状态与交接

更新：2026-10-23。

当前为三模式 UC 混合策略、独立两路充电和六维物理行动。每阶段连续采样 384 步，γ=0.99、Actor λ=0.97、Critic λ=1；全部 Critic 拟合后重算固定 GAE，再更新本阶段 Actor。Value 使用 LayerNorm，UC / 消费者输出单位为 20 / 10。

2026-10-23 增加 UC Actor 中心化 SOC 输入及三个零初始化模式直连权重，补齐逐 minibatch 模式概率比、PPO 截断、分支梯度与实际 Adam 变化诊断。Critic、奖励、采样起点和训练预算保持；未采用高 SOC 初态或重采样。原始电量观察仍为 kWh；新输入契约拒绝直接恢复旧训练 checkpoint，旧导出 Actor 按原配置仍可读取。实现及检查入口见 `test_stackelberg_mode_diagnostics.py`、阶段合批集成测试和 [TRAINING.md](TRAINING.md)。用户已授权提交后从 seed 42 新开 10 轮，输出 `20261023-soc-mode-10rounds`，对照此前 `20261022-phase-batch-10rounds`；实际进度以新目录 status.json 为准。

月初开始采样，批间保留 SOC、FIFO、月累计和历史反馈，真实月末结算一次。候选从相同消费者网络、Adam、环境及 RNG 分支，只保留选定响应。新候选未通过配对 UC 收益规则时选择候选 0。完整协议见 [TRAINING.md](TRAINING.md)。

最新完整实验为 `20261022-phase-batch-10rounds`，已完成 10 轮、40 批。计划 72 轮的 `20261022-phase-batch-72rounds` 已按用户要求停止，只完成 1 轮，内存训练状态未保存。本次整理没有恢复或新开真实训练。历史最长完整实验为旧 72 轮、576 批。结果统一由 [RESULTS.md](RESULTS.md) 导航。

本次删除 5 份早期 / 过期实验说明、旧三轮专用汇总脚本、重复结果及无依赖的早期原始输出。旧基线报告引用链和冻结 Critic 来源 checkpoint 留作依赖；保留主实验的原始结果维持原样。删除清单和保留文件哈希见 [retained_runs.json](results/retained_runs.json)。

环境审核见 [implementation_audit_zh.md](../../environments/sn_energy_market/docs/v9/implementation_audit_zh.md)。式（1）—（29）的动作、出清、物理、DR、账本与终端计算一致；公共日历尚未编码到观测，文档中的“不折扣有限目标”已改正为实际 γ=0.99。按简洁要求移除异常诊断包装和重复检查，阶段、物理约束、守恒及快照完整性检查保留。

后续建模事项：若要满足 v9 第 3.2 节完整公共日历输入，需要定义字段、维度及网络 / checkpoint 契约；若要优化不折扣有限期目标，应显式记录新的 γ 和训练协议。现有经济性质 U02、A01—A03 仍按建模文档解释。离线公共时间 Critic 的 RMSE 改善属于已保存历史对照，在线收益仍需对应实验。
