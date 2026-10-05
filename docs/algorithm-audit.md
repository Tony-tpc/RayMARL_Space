# 算法一致性审计

来源：Ye 等，IEEE TSG 2021，DOI `10.1109/TSG.2021.3103917`；Iqbal 与 Sha，ICML 2019，以及其作者仓库 <https://github.com/shariqiqbal2810/MAAC>。

## 主线机制

| 论文机制 | 实现 | 验收 |
| --- | --- | --- |
| 混合 Gaussian/Bernoulli 策略 | distributions.HybridDistribution | 分支概率、边界 Jacobian、饱和 log-prob |
| 独立局部 Actor | networks.ActorModule | 参数所有权和局部推理 |
| 私有一层 g_i 与两层 f_i | networks.AttentionCritic | 层次、异构输入和输出形状 |
| 共享 Wq/Wk/Wv，屏蔽自身 | networks.AttentionCritic | 权重对角为零、归一化、N=1 |
| 单 Q、目标 Actor、目标 Critic | learner.JointTorchLearner | TD target 和目标插值 |
| 式 44 score-function | learner.score_function_loss | 手算期望梯度 |
| 固定其他动作的反事实 baseline | learner.JointTorchLearner | 离散枚举、独立 MC 样本、恢复一致性 |
| 联合 Replay | replay.JointReplay | 时间与智能体对齐、快照隔离 |
| 局部执行 | inference.ActorBundle | 不创建 Critic 或环境 |

Actor 和 Critic 使用各自的历史 LSTM，避免优化器所有权混淆。Critic 的历史输入先与当前局部特征拼接，再通过一层状态-动作投影 g_i。各 Actor 使用独立的两层 MLP；论文未给出该宽度，默认 64 属于补足参数。

## 原文矛盾及边界

1. Algorithm 1 的 tau 系数与 tau=0.001 软更新率语义不一致。实现采用 `target=(1-tau)*target+tau*online`。
2. 注意力求和排除自身，而式 42 分母包含自身。实现明确排除自身再归一化。
3. 论文编码隐私并无正式泄露上界。此库实现训练信息流，不实现加密协议、差分隐私或真实分布式隐私保障。
4. 共享注意力参数压缩输入并使总运算随智能体数线性扩展；实测性能仍需单独报告。
5. CTDE 不证明消除全部学习非平稳性，也不证明纳什均衡。
6. 充放电符号、Cost 与 reward 符号、EV 约束惩罚方向存在矛盾。环境不复现，故只记录。
7. 能源论文未公开可核对的完整训练程序及所有超参数。不能声称与作者程序逐次更新完全相同。

## 已明确补足的参数

默认 actor/critic 学习率 3e-4/1e-3，RMSProp，gamma=0.99，tau=0.001，temperature=0.1，batch=128，均匀 Replay=100000，warmup=512，每四步更新，连续反事实样本=8。单头不缩放点积；Actor 梯度裁剪 0.5、联合 Critic 50。主线不额外缩放共享梯度。

MASAC 使用固定温度、双 Q、连续重参数化与离散精确期望；ISAC 使用局部软 Critic 与相同混合核。它们不是把现有库的注意力 SAC 改名后当成 MAAC。MADDPG 仅接受连续动作，其他 replay 动作保持固定。

## 公开环境的外推范围

CityLearn 3.0.2 使用原生 MARL 奖励，而非论文 MMR 收益。原生小时分辨率下 48 步是 48 小时，不是论文的 24 小时。官方 168 个时间行实际产生 167 次转换；实验记录实际联合步数和回合完整性带来的预算超出。

原生 MARL 奖励可能不惩罚电器未服务，收益与服务指标同时报告。不得用少服务取得的低成本宣称算法更优，不用短跑收益排序证明机制完整或收敛。
