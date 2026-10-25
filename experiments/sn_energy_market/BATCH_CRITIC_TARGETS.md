# Critic 目标分离与逐采样批日志

本文记录当前目标分离和逐批日志协议。季度轮转已进入主流程；阶段合批、SOC 72 轮、采样 × 门槛 2×2 和 D24 八轮对照分别保留实际编号与完成量，结果见[保留结果索引](RESULTS.md)。

## 1. 固定 Critic 目标，拟合后重算 Actor 优势

Actor 使用 `lambda_gae=0.97`，并与评价共用 `gamma=0.99`计算 GAE。Critic 新增独立 `critic_lambda`，默认 1.0；通用命令行参数为 `--critic-lambda`，合法范围为 [0, 1]，并写入 checkpoint 学习契约。

当 `critic_lambda=1` 时，每条记录的目标为：

\[
y_t^V=\sum_{k=0}^{n-1}\gamma^k r_{t+k}
       +\gamma^n V_{\mathrm{old}}(s_{t+n}).
\]

`n` 截至本批末尾或真实终止。真实终止的尾值为零；一批跨真实终止时，两侧回报分开递推。批末 bootstrap 仍沿用下一次合法公告状态；临时公告恢复环境和 RNG。目标在 NumPy 采样结果中固定，优化不重新计算目标，也不回传到旧 Value。在采样时设 `critic_lambda=lambda_gae` 可得到采样 GAE 加旧值的目标数值；Critic 拟合后的 Actor GAE 不再满足这一旧值恒等式，但不会自动放行缺少新契约字段的历史 checkpoint。

保持现有 MSE、Value 单位、Actor 优势归一化和两阶段优化顺序。当前小规模配置为同阶段两个 192 步单位连续采成一批 384 步（Ausgrid 8 天）、Critic 4 epoch、Actor 2 epoch、minibatch 32；没有 MC 预拟合或额外采样。六维 UC 动作、消费者行动、网络输入和环境模型保持原状。

Critic 拟合完成后，用新 Value 在保存的当前及下一决策输入上重算 GAE，整批归一化一次，再固定用于 Actor 的全部 epoch。原行为 logp、模式、潜变量、物理动作、固定 Critic 目标保持不变；消费者下一输入中的 UC 公告不重新采样。第 192 步没有人工截断，真实终止仍切断递推。

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
| `diagnostics_before/after` | 预测、固定目标、实际多步奖励、bootstrap、单期奖励的均值、标准差、范围及两层 Tanh 饱和比例 |
| `critic_gradient_norms`、`metrics` | 各次 Critic 优化的裁剪前梯度范数与采样能源指标 |

这里的更新前后 loss 是完整批次重新测量，不是最后一个 minibatch 的缓存指标，也不是 epoch 均值。诊断使用 `no_grad` 和确定性的已保存行动概率计算，不调用随机熵采样，不更新网络、Adam 或 Learner 指标。冻结 Actor 不继承历史 loss。原有 `metrics.jsonl` 另补入 `policy_loss/value_loss/value_rmse/value_explained_variance`，仍是外层聚合，逐批绘图只读取新文件。

候选尚未选定时，提案和响应记录为 `pending`。选择完成后追加 `event=branch_resolution`，按 `batch_id` 关联原记录；不重写或删除已发生的更新。若运行在选择前中断，曲线保留 pending 标记。

- 消费者前置适应：保留其更新和采样推进。
- 选中候选响应：保留所有 Critic、消费者 Actor 和采样终点；未选响应的更新全部标为回滚，包括候选 0。
- UC 提案：候选是累积生成的。选择候选 k 时，提案 1 至 k 的 UC Actor／Critic 更新保留；所有提案的消费者 Critic 和提案采样推进均丢弃，故这些记录可能为 `partial`。
- 偏离搜索：记录其实际更新，但全部标记回滚。

保留标记表示该更新属于本轮选定分支的训练历史，不表示后续优化永远不再改变这些参数。日志不参与选优、状态恢复、预算计数或能源约束。

## 3. 横轴使用真实学习批编号

历史小规模配置没有偏离搜索，每外层轮次 8 个采样批：消费者适应 2、UC 提案 2、候选 0 响应 2、候选 1 响应 2。三轮自然得到 `batch_id=1…24`。绘图不按 epoch 分组、不插值补点，也不筛掉回滚批；其他预算会使用其真实批编号。

本次同阶段合批后，每轮四批，3 轮为 12 批、10 轮为 40 批；样本及默认配置下的总优化步数不变。历史 24／80 批记录不改写。

有真实日志后执行以下纯绘图命令，不会训练或读取数据集：

```powershell
python experiments/sn_energy_market/plot_batch_losses.py <运行目录>/batch_losses.jsonl --output <图目录>
```

绘图 Python 需安装 Matplotlib。每个模块输出 PNG 和 SVG，包含 Critic loss、原单位 RMSE、Actor loss 三行。回滚更新用叉号标出，pending 用方框，冻结 Actor 留缺口。批间目标和策略分布会变化，曲线不能当作同一固定数据集上的单调收敛证明。

当前结果按 [RESULTS.md](RESULTS.md) 保留。早期三轮的重复日志与图表已清理，历史 72 轮保留完整 576 批，涵盖其前 24 批基线。
