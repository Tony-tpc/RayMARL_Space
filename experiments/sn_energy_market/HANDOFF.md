# 当前状态与交接

更新：2026-10-20。

## 当前实现

按用户要求，只保留全部 Critic 的 `Linear → LayerNorm → Tanh` 隐藏层和诊断日志。UC 时间支路、供电／现金可行性回调、轮初评价缓存、整轮全拒绝恢复路径及对应测试已完整删除。Actor 网络、六维 UC／四维消费者动作、观测、物理和奖励保持。

每批先更新全部 Critic，再更新选定 Actor；Actor λ=0.9025，Critic λ=1.0，批内实际多步回报在普通批末保留旧 Value 尾值、真实终止尾值为零。目标采样后固定；没有额外 MC 预拟合。

本次配置每轮 8 批：消费者适应 2 批 → UC 提案 2 批 → 旧 UC 的消费者响应 2 批 → 新 UC 的消费者响应 2 批。响应分支恢复相同消费者网络、Adam、环境和 RNG，保留对应 UC 网络及 Adam。新候选须通过相对候选 0 的配对 UC 折扣收益规则，否则选择候选 0；不存在整轮全部拒绝。候选 0 的消费者已更新，提交后主轨迹继续推进，Critic 也保留前置适应与所选分支的学习。

采样从训练月月初开始，批间保留 SOC、任务、月累计用电、日历和历史反馈，真实月末才结算并切换月份。验证、bootstrap 前瞻和未选分支不污染主轨迹。日志保留预测、目标、实际多步奖励、bootstrap、Tanh 饱和比例、裁剪前梯度范数和能源指标；批号不随分支恢复重置。

## 本次短测与边界

只运行一次 seed 42、3 轮 24 批短测，输出为 `outputs/ausgrid-layernorm-pilot/20261020/`，日志为 `.tmp/layernorm-pilot-20261020.log`。三轮选择 `[0, 0, 0]`，已完成 1,536 个分支 PPO 时段、768 个主轨迹时段，到达 2010-07-17 零时。每个 Critic 实际执行 192 次优化，所选分支累计保留 96 次；UC Actor 提案未保留，消费者 Actor 各保留 48 次。

这次没有重复同一轮，但没有覆盖训练月末，不能据此宣布解决旧 72 轮实验第 329 批的尖峰。比较基准是不加 LayerNorm 的已训练三轮策略；上次全拒绝后的初值另列。新旧策略的实际轨迹与最终保留优化次数不同，不能把结果解释为同一固定数据上的纯回归比较。

最终四个完整月验证共 5,904 状态，checkpoint 8 步精确重放通过。UC 月初／中旬／月末 RMSE 为 19.8559／13.2477／14.6318，全月 16.1286；相对旧三轮，中旬、月末及全月改善，月初变差。消费者平均全月 RMSE 为 5.3428，旧三轮为 5.4321。完整月供电率 99.1905%，失供 211.7739 kWh，UC 现金 38.1914 AUD/日；供电改善，现金略降。详细 48 小时与完整月口径见当前报告。71 项相关检查通过，1 项 CUDA 检查未运行；测试集未加载或评估。

## 入口与证据

- [当前报告](LAYERNORM_PILOT.md)：回退范围、三轮结果、真实 1–24 批曲线及 GRU 文献分析。
- [当前建模](../../environments/sn_energy_market/docs/v9/modeling_complete_zh.md)及 [HTML 阅读版](../../environments/sn_energy_market/docs/v9/modeling_complete_zh.html)：第 14 节对应实现，公式离线渲染为 SVG。
- [训练协议](TRAINING.md)、[逐批目标与日志](BATCH_CRITIC_TARGETS.md)。
- `batch_critic_pilot.py`：默认三轮，不会覆盖已有输出；每批 64 步、Critic 4 epoch、Actor 2 epoch、minibatch 32。
- `evaluation.py` / `training_audit.py`：冻结评价和被动审计。
- [原三轮基线](BATCH_CRITIC_PILOT.md)及 `outputs/ausgrid-batch-critic-pilot/20261020/`。
- [旧 72 轮实验](YEAR_STUDY.md)及 `outputs/ausgrid-year-study/20261020/`，源码定位 `8509e88`。
- [已撤销方案的三轮实验](CRITIC_STABILITY_PILOT.md)及 `outputs/ausgrid-critic-stability-pilot/20261020/`，源码定位 `1d223cd`；保留全拒绝及日志故障的历史证据，不改写旧结果。

## 后续工作范围

本次未加入 GRU、首批 Critic 特殊保留、额外训练或新候选约束。GRU 仅作为研究建议，下一项优先检验仅 UC Critic 的历史表示、Actor 暂不变；若实施，应明确循环网络的历史输入、连续序列训练、隐藏状态跨批保留与真实月末重置、分支隔离和 checkpoint 重放。当前单步打乱的 PPO 不能只替换一层网络就获得正确循环训练。详见当前报告；进一步修改和实验需另行确定范围。
