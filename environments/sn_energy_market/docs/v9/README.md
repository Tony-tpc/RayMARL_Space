# 六维 UC 调度模型

[完整建模文档](modeling_complete_zh.md)是当前规格：环境式（1）—（29）保持，训练第 14 节已同步月初连续采样、全部 Critic 更新、独立多步目标、候选快照和逐批 loss。

[打开已渲染 HTML](modeling_complete_zh.html)：含目录和公式 SVG，可离线阅读。Markdown 是内容源文件，HTML 随它重新生成。

- [公式—实现—测试映射](formula_implementation_map_zh.md)：物理、结算、输入边界与维护中的测试。
- [环境验证基线](full_validation_current_commit_zh.md)及[机器汇总](full_validation_current_commit.json)：提交 `33f27b7` 的完整规模验证，保留原始哈希。
- [训练协议](../../../../experiments/sn_energy_market/TRAINING.md)：当前接口、预算与运行方式。
- [中长期实验](../../../../experiments/sn_energy_market/YEAR_STUDY.md)：同一算法的 72 轮训练与固定节点验证。
- [小规模固定预算报告](../../../../experiments/sn_energy_market/BATCH_CRITIC_PILOT.md)：Critic 分层 RMSE、供电率、UC 收益与 24 批真实曲线。

旧版本模型、过时 HTML、文档修改过程记录及历史 verifier 已移出当前目录，历史内容由 Git 查阅。当前 HTML 保留为阅读交付物。
