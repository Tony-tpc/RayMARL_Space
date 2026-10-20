# Critic 目标分离与逐采样批日志

本文记录目标分离和逐批日志的实现。随后已按原预算完成一次 Ausgrid 小规模训练与验证，实测结果和真实 24 批曲线见[固定预算对照](BATCH_CRITIC_PILOT.md)。测试集仍未加载或评估。旧报告缺少逐批 loss，不能由三个外层轮次的均值恢复旧实验的 24 个点。

## 1. 固定两套目标

Actor 继续使用原有 `lambda_gae=0.9025`（半小时场景）计算 GAE。Critic 新增独立 `critic_lambda`，默认 1.0；通用命令行参数为 `--critic-lambda`，合法范围为 [0, 1]，并写入 checkpoint 学习契约。

当 `critic_lambda=1` 时，每条记录的目标为：

\[
y_t^V=\sum_{k=0}^{n-1}\gamma^k r_{t+k}
       +\gamma^n V_{\mathrm{old}}(s_{t+n}).
\]

`n` 截至本批末尾或真实终止。真实终止的尾值为零；一批跨真实终止时，两侧回报分开递推。批末 bootstrap 仍沿用下一次合法公告状态；临时公告恢复环境和 RNG。目标在 NumPy 采样结果中固定，优化不重新计算目标，也不回传到旧 Value。设 `critic_lambda=lambda_gae` 可得到原目标数值，但不会自动放行缺少新契约字段的历史 checkpoint。

保持现有 MSE、Value 单位、Actor 优势归一化和两阶段优化顺序。原小规模配置仍为每批 64 步、Critic 4 epoch、Actor 2 epoch、minibatch 32；没有 MC 预拟合或额外采样。六维 UC 动作、消费者行动、网络输入和环境模型保持原状。

该目标减少对批内中间旧 Value 的依赖，增加回报目标方差。批末仍依赖旧 Value；它不能补足月末数据覆盖，也不能据此断言验证 RMSE 或供电率改善。

## 2. 每批立即写入

原生运行目录（`algorithm.logdir`）新增 `batch_losses.jsonl`。一次采样及其 Critic、Actor 更新完成后写入一条 `event=batch`，立即 flush 并关闭文件，不等外层轮次结束。核心字段如下：

| 字段 | 定义 |
|---|---|
| `batch_id` | 日志文件内递增编号；不进入候选或 checkpoint 快照，不随回滚重置 |
| `completed_updates` | 当前训练状态中的实际批次数；恢复旧 checkpoint 时可回退，与审计编号区分 |
| `outer_round`、`phase`、`candidate_index` | 外层轮次、消费者适应／UC 提案／候选响应／偏离搜索、候选索引 |
| `scenario_seeds`、`first_period`、`last_period`、`true_terminals` | 本批采样范围与真实终止计数 |
| `modules.<id>.critic_loss_before/after` | 同一完整批次、同一固定目标上，优化前后 `0.5 × mean(((V−target)/scale)²)` |
| `modules.<id>.critic_rmse_before/after` | 同一固定目标上的原奖励单位 RMSE；不是完整验证月 MC RMSE |
| `modules.<id>.policy_loss_before/after` | 实际更新 Actor 在固定行为概率和固定归一化优势上的完整批 PPO clipped surrogate；不含熵项 |
| `modules.<id>.policy_loss` | `policy_loss_after` 的直接字段；冻结 Actor 的三个 policy loss 字段均为 `null` |
| `branch_status`、`retained_updates`、`sampling_retained` | 分支决策及每个模块的 Actor／Critic 更新和采样轨迹是否保留 |
| `diagnostics_before/after` | 预测、固定目标、实际多步奖励、bootstrap、单期奖励的均值／标准差／最小最大值，以及两层 Tanh 的 `abs(h)>0.99` 比例 |
| `critic_gradient_norms`、`metrics` | 本批各次 Critic 优化的裁剪前梯度范数、采样环境指标；裁剪阈值保持 0.5 |

这里的更新前后 loss 是完整批次重新测量，不是最后一个 minibatch 的缓存指标，也不是 epoch 均值。诊断使用 `no_grad` 和确定性的已保存行动概率计算，不调用随机熵采样，不更新网络、Adam 或 Learner 指标。冻结 Actor 不继承历史 loss。原有 `metrics.jsonl` 另补入 `policy_loss/value_loss/value_rmse/value_explained_variance`，仍是外层聚合，逐批绘图只读取新文件。

候选尚未选定时，消费者前置适应、提案和响应均为 `pending`。选择完成后追加 `event=branch_resolution`，按 `batch_id` 关联原记录；不重写或删除已发生的更新。若运行在选择前中断，曲线保留 pending 标记。

- 消费者前置适应：整轮提交时保留；全部候选不合格、恢复轮初时也标为回滚。
- 选中候选响应：保留所有 Critic、消费者 Actor 和采样终点；未选响应的更新全部标为回滚，包括候选 0。
- UC 提案：候选是累积生成的。选择候选 k 时，提案 1 至 k 的 UC Actor／Critic 更新保留；所有提案的消费者 Critic 和提案采样推进均丢弃，故这些记录可能为 `partial`。
- 偏离搜索：记录其实际更新，但全部标记回滚。

保留标记表示该更新属于本轮选定分支的训练历史，不表示后续优化永远不再改变这些参数。日志不参与选优、状态恢复、预算计数或能源约束。

LayerNorm、UC 时间支路和供电／现金检查见[当前训练协议](TRAINING.md)。批内目标分解仅用于诊断；两套目标、MSE、更新次数及动作和环境模型保持不变。

## 3. 横轴直接使用 1–24

原小规模配置没有偏离搜索，每外层轮次 8 个采样批：消费者适应 2、UC 提案 2、候选 0 响应 2、候选 1 响应 2。三轮自然得到 `batch_id=1…24`。绘图不按 epoch 分组、不插值补点，也不筛掉回滚批；其他预算会使用其真实批编号。

有真实日志后执行以下纯绘图命令，不会训练或读取数据集：

```powershell
python experiments/sn_energy_market/plot_batch_losses.py <运行目录>/batch_losses.jsonl --output <图目录>
```

绘图 Python 需安装 Matplotlib。每个模块输出 PNG 和 SVG，包含 Critic loss、原单位 RMSE、Actor loss 三行。回滚更新用叉号标出，pending 用方框，冻结 Actor 留缺口。批间目标和策略分布会变化，曲线不能当作同一固定数据集上的单调收敛证明。

历史 `outputs/ausgrid-online-critic-pilot/20261020` 保持原状。目标与日志先通过单元测试和微型合成场景集成检查；新真实实验单独保存在 `outputs/ausgrid-batch-critic-pilot/20261020`，两类日志分别保留。
