# 算法一致性审计：Ye 2021 路线

## MAAC 来源与实现

来源：Ye 等，IEEE TSG 2021，DOI [10.1109/TSG.2021.3103917](https://doi.org/10.1109/TSG.2021.3103917)。原始思想参照 [Iqbal 与 Sha 2019](https://proceedings.mlr.press/v97/iqbal19a.html)，本库 MAAC 的基准保持 Ye 能源论文。

| 机制 | 实现位置与语义 |
| --- | --- |
| Gaussian/Bernoulli 混合策略 | MAACRLModule + HybridDistribution；有界 Gaussian 使用稳定 Tanh 密度修正 |
| 历史编码 | Actor/Critic 各自 Encoder 与 LSTM，历史右填充与 mask |
| 私有 g_i / f_i | AttentionCritic 的一层状态-动作投影与两层 Q 头 |
| 共享 Wq/Wk/Wv | 单头注意力，排除自身，不另加点积缩放 |
| 单 Q 与目标策略/目标 Q | MAACCriticModule、MAACRLModule |
| 式 44 策略梯度 | MAACTorchLearner；score-function 与停止梯度的反事实 baseline |
| 联合经验 | 原生 MultiAgentEpisode 转换为完整联合 Replay 行 |
| 局部执行 | 推理侧无 Critic；ActorBundle 仅构造局部策略 |

## 论文明确参数与项目补足参数

论文明确：RMSProp、gamma=0.99、软更新率 tau=0.001、batch=128、Replay 容量 100000、LSTM 历史特征。

项目补足：hidden_dim=64、actor_lr=3e-4、critic_lr=1e-3、temperature=0.1、warmup=512、每四个环境步一次更新、baseline 连续采样数=8、Actor/Critic 梯度裁剪 0.5/50。Actor 与 Critic 的历史编码器分别管理。上述补足不得写成论文原始设置。

原文 Algorithm 1 的 tau 系数与文字软更新率语义不一致，采用 `target=(1-tau)*target+tau*online`。注意力公式求和排除自身、分母却含自身，采用排除自身后归一化。补足参数及修订在库配置中明确，不宣称逐次更新等同作者代码。

## 六种算法的边界

- MAAC：能源论文注意力单 Q 与反事实 score-function，不改成现代双 Q SAC。
- MASAC：联合双 Q、连续重参数化、自身离散动作精确期望、固定温度；其他智能体动作作为无梯度上下文。
- MADDPG：连续确定性 Actor；更新本 Actor 时固定其他 replay 动作，无熵项。
- ISAC：每个策略自己的局部双 Q；当前策略产生下一动作；无目标 Actor、无 MAAC baseline；混合动作属于扩展。
- MAPPO/IPPO：复用 RLlib PPO/GAE；集中与局部 Value 分别定义，Actor 都局部执行。

## 可验证结论的范围

算法库不复现论文的能源设备、MMR 市场与奖励环境；不会自动发起 CityLearn 大型实验。现有实验脚本作为历史研究材料保留，其私有 API 使用需要单独迁移，不能把旧曲线当成新版本验证。

嵌入交换不提供正式隐私保证，原生 Ray 运行也不会自动建立论文的交易平台安全机制。CTDE 和训练稳定性不证明纳什均衡。单个 Critic 输入压缩不等同整体线性计算：全连接注意力有 N×N 配对成本。
