# Ausgrid 储能 72 轮长跑

从 seed 42 的相同初始化重新训练 72 次外层 `training_step()`，主要参照 `20261022-phase-batch-10rounds`。本次检验延长训练是否形成随 SOC 和市场状态变化的充放电策略。

沿用 gamma=0.99、Actor GAE lambda=0.97、Critic lambda=1、384 步阶段学习批、Actor 2/Critic 4 epoch、minibatch 32，以及现有三模式、六维物理动作、网络和奖励。此前提出的 SOC 直连输入和近零功率屏蔽修正尚未实施。

总预算：288 个学习批、110,592 个 PPO 分支时段、55,296 个主轨迹时段；每个 Critic 实际优化 13,824 次，UC Actor 提案优化 1,728 次。各轮候选回滚后保留的更新另计。主轨迹约 1,152 天时段，训练年会轮换；验证集参与候选选择，测试集未加载。

```powershell
.venv/Scripts/python.exe -u experiments/sn_energy_market/hybrid_storage_study.py --outer-rounds 72 --output outputs/ausgrid-hybrid-storage/20261022-phase-batch-72rounds --previous outputs/ausgrid-hybrid-storage/20261022-phase-batch-10rounds
```

每轮保存 checkpoint；完整诊断节点为 0、3、12、24、48、72 轮，每次覆盖四个验证月的 5,904 时段。原始半小时记录、真实逐批 loss、候选接受记录、固定状态模式探针和 UC 权重均保存在新目录。结束后自动生成 `comparison72/report.html`、`summary.json` 和 13 张对比图。

验收同时检查随机与确定性执行的充放电、SOC 近空/近满比例、模式对 SOC 的响应、两路充电、UC/Consumer 收益、原生失供和 Critic 的 RMSE/解释方差。SOC 高或随机模式近似均匀都不单独作为学会调度的证据。

低额度跟进只读进度，正常运行保持安静；完成后一次性核验结果并通知本聊天。进程异常时核对 checkpoint 和实际已执行预算，先报告，不自动重启或增加轮次。
