# RayMARL

基于 **RLlib 2.59 原生新 API 栈**的多智能体算法库。MAAC 按 **Ye 2021 能源论文**定义；其他算法有各自独立的配置、模型和学习器。

本目录是独立的 Python 算法项目。环境位于上一级的 `environments/`，工作区使用方式见 [主目录说明](../README.md)。下列源码路径以 `RLAlgo/` 为基准。

## 从算法目录开始阅读

```text
src/raymarl/
  algorithms/
    maac/       # Ye 2021，注意力单 Q 与反事实策略梯度
    masac/      # 联合双 Q SAC
    maddpg/     # 确定性连续策略与联合 Q
    isac/       # 局部双 Q SAC
    mappo/      # 局部 Actor、集中 Value
    ippo/       # 局部 Actor、局部 Value
    stackelberg/ # 独立双层 PPO 扩展，公告后响应与匹配候选评价
  common/       # 可复用网络、连接器、回放、优化器与运行管理
  distributions.py
  spaces.py
  inference.py
```

每个算法目录包含 `algorithm.py`、`config.py`、`rl_module.py`、`learner.py`。公共层不根据算法名称选择损失或 Critic 模式。

`stackelberg` 通过 `from raymarl.algorithms.stackelberg import StackelbergConfig` 显式使用，保持通用算法的导出和注册表独立。其采样协议、物理期 GAE、PPO 学习、配对评价、停止判据和 Actor 导出各自有单独接口；能源环境转换仅在实验适配器中实现。完整生命周期及实验参数见[能源市场训练说明](../experiments/sn_energy_market/TRAINING.md)。下文的通用算法配置示例与动作支持表仍描述原有六算法。

## 安装与训练

```powershell
# 在 RLAlgo 目录执行；在工作区主目录则安装 ./RLAlgo
python -m pip install .
python -m pip install '.[test]'
```

```python
import ray
from raymarl import MAACConfig

ray.init()
config = (
    MAACConfig()
    .environment(MyMultiAgentEnv)
    .env_runners(num_env_runners=0, rollout_fragment_length=4)
    .learners(num_learners=0, num_gpus_per_learner=0)
    .rl_module(model_config={"hidden_dim": 64})
    .training(batch_size=128)
)
algo = config.build_algo()
result = algo.train()
checkpoint = algo.save_to_path()  # 当前运行自己的 checkpoint 目录，默认保留最新两个
algo.restore_from_path(checkpoint)
algo.export_actors("./actors.pt")
algo.stop()
ray.shutdown()
```

`MyMultiAgentEnv` 由调用方提供，遵守 RLlib `MultiAgentEnv` 契约。固定智能体 ID 存放于 `possible_agents`；默认自动为每个智能体建立独立策略。也可显式配置一一对应的 `.multi_agent(...)` 映射。

一次 `train()` 按配置采样批量推进，可跨回合，也可只产生回合片段。联合环境步数和智能体步数使用 RLlib 原生指标；回报指标采用原生口径。

## 六种算法的区别

| 算法 | Actor 更新 | 训练用价值网络 | 信息边界 |
| --- | --- | --- | --- |
| MAAC | 熵正则 score-function + 反事实 baseline | 注意力单 Q + 目标 Actor/Critic | Actor 局部，Critic 联合 |
| MASAC | 连续重参数化、离散枚举期望 | 联合双 Q + 目标双 Q | Actor 局部，Critic 联合 |
| MADDPG | 确定性策略梯度 | 联合 Q + 目标 Actor/Critic | Actor 局部，其他 replay 动作固定 |
| ISAC | 连续重参数化、离散枚举期望 | 各策略私有局部双 Q + 目标双 Q | Actor/Critic 都局部 |
| MAPPO | 原生 PPO、GAE | 集中 Value | Actor 只读 local |
| IPPO | 原生 PPO、GAE | 局部 Value | Actor/Value 都局部 |

MAAC 是 Ye 2021 的算法实现与明确修订，包含有界 Gaussian/Bernoulli 策略和历史编码。通用 Discrete、Dict/Tuple 动作属于库扩展。ISAC 已改为独立 SAC，不再复用 MAAC 的反事实 baseline 或目标 Actor。

## 支持边界与输出

- 联合算法采用原生本地 EnvRunnerGroup 和本地 LearnerGroup；CPU 或一个 GPU。动态智能体、异步行动、提前独立终止和多 Learner 不在本轮支持范围。
- 支持异构向量观测、有限边界 Box、Discrete、MultiBinary 和 Dict/Tuple 组合。MADDPG 仅支持连续 Box 组合。
- 历史观测支持 `Dict(current=Box, history=Box[H,C], history_mask=Box[H])`，有效历史在前、零填充在后。精确离散枚举最多 256 种组合。
- 默认输出位于 `outputs/raymarl/<算法>-<唯一运行ID>`，保存精简指标，每 10 次训练自动保存 checkpoint，最多保留两个。用 `checkpoint_interval=0` 关闭自动保存；`detailed_outputs=True` 开启完整结果日志。
- `save_to_path()` 使用受控保留策略；显式指定路径的 checkpoint 由调用方管理，库不会删除。
- 新格式 checkpoint 包含模型、目标网络、注册优化器、联合回放、RNG 和计数。旧 v0.1 checkpoint 明确拒绝恢复。通用模拟器的内部状态没有统一序列化接口，恢复训练会重新开始环境回合，不保证复现未完成回合的逐步轨迹。
- `from raymarl import ...Config` 保持有效；旧 `training(hidden_dim=...)` 发出弃用提示，与新模型参数冲突时拒绝构建。原有私有 `JointModule/update_joint/critic_mode` 接口已移除。
- ActorBundle 导出包含局部 Actor 权重，不创建 Critic、Value 或环境。使用 `register_algorithms()` 显式注册具名 Tune 算法。

本库不包含能源论文的原始交易环境，不宣称复现论文收益、300 户结果、纳什收敛或正式隐私保证。注意力压缩单个 Critic 的输入；全连接两两注意力的总计算仍随智能体数量呈二次增长。

详见 [架构](docs/architecture.md)、[算法审计](docs/algorithm-audit.md)、[验证记录](docs/validation.md)和[清理记录](docs/cleanup.json)。

从环境定义到训练、评估、保存恢复及推理的完整代码，见 [单智能体与 MARL 运行示例](examples/README.md)。环境实现位于 [environments/energy_env.py](../environments/energy_env.py)。
