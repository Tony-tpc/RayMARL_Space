# 能源市场实验

当前入口采用三模式储能、v9 六维 UC / 四维消费者物理动作、γ=0.99、Actor λ=0.97、Critic λ=1。同阶段两个 192 步采样单位合成 384 步学习批，每轮四批；先拟合全部 Critic，再重算 GAE 并更新选定 Actor。

季度轮转已接入主训练入口，四条季度流按完整外层轮次暂停、续接。最近完成的采样 × 门槛 2×2 为四组各 24 轮；SOC 版本已完成总计 72 轮。D24 的 z=1 / 始终保留候选两组各 8 轮及最终评价已完成：分别保留 4 / 8 轮，模式选择仅小幅改善，完整月 Critic 误差恶化。保留结果与原始路径统一见 [RESULTS.md](RESULTS.md)。

| 入口 | 用途 |
|---|---|
| [v9 建模](../../environments/sn_energy_market/docs/v9/modeling_complete_zh.md)、[实现审核](../../environments/sn_energy_market/docs/v9/implementation_audit_zh.md) | 数学定义、代码一致项及差异 |
| [TRAINING.md](TRAINING.md)、[BATCH_CRITIC_TARGETS.md](BATCH_CRITIC_TARGETS.md) | 运行协议、恢复、目标与日志 |
| `train.py` | 合成 / 数据场景训练、冻结评价、恢复和导出 |
| `rollback_comparison.py` | D24 同起点、两组各 8 轮候选保留对照及冻结评价 |
| `batch_critic_pilot.py` | 固定预算，默认 3 轮、12 批 |
| `hybrid_storage_study.py`、`study_recovery.py` | 储能训练、逐时段记录、预算内恢复 |
| `evaluation.py`、`training_audit.py` | 冻结完整月评价、独立目标和账本审计 |
| `profile_dataset.py`、`training_env.py` | 月场景、数据契约、两阶段训练适配 |
| `plot_hybrid_storage.py`、`plot_storage_round_comparison.py`、`plot_batch_losses.py` | 对已有结果绘图 |
| `frozen_critic_ablation.py`、`audit_frozen_critic.py`、`plot_frozen_critic.py` | 独立离线 Critic 对照及核验 |
| `validate_v9.py`、`validation_helpers.py`、`smoke_test.py` | 环境数值参考、机制检查与合法行动 smoke |

算法位于 `RLAlgo/src/raymarl/algorithms/stackelberg/`，环境位于 `environments/sn_energy_market/src/snmarket/`。数据获取与切分见 [AUSGRID.md](AUSGRID.md)，状态与后续事项见 [HANDOFF.md](HANDOFF.md)。

```powershell
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/train.py --smoke
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/smoke_test.py --consumers 3 --horizon 288
.\.venv\Scripts\python.exe -B experiments/sn_energy_market/validate_v9.py --plan-only
```

完整环境压力验证入口默认执行 100,000 个静态独立参考对照、10,000 个非法输入、10,000×96 期常规回合、100×672 期长回合、20×3,360 期跨月轨迹。计划量与已通过量分别记录；新运行写入新的 `outputs/snmarket-validation/<运行ID>/`，续跑要求来源哈希一致。环境生命周期见[环境说明](../../environments/sn_energy_market/README.md)。
