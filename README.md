# RayMARL

为 RLlib 2.59 提供通用的多智能体联合 Critic、混合动作和集中 Value 扩展。

```python
import ray
from raymarl import MAACConfig

ray.init()
config = (MAACConfig()
    .environment(MyMultiAgentEnv)
    .learners(num_learners=0, num_gpus_per_learner=0)
    .training(hidden_dim=64, batch_size=128))
algo = config.build_algo()
result = algo.train()
checkpoint = algo.save_to_path("./ray_results/checkpoint")
algo.stop()
ray.shutdown()
```

`MyMultiAgentEnv` 由调用方提供，是固定智能体集合的 RLlib `MultiAgentEnv`。一次 `train()` 完成一个环境回合；返回的步数是联合环境步数。

## 安装

```powershell
python -m pip install .
python -m pip install '.[test]'
```

基础依赖不固定 CUDA 构建。GPU 用户先按 [PyTorch 官方安装方式](https://pytorch.org/get-started/locally/)安装对应 PyTorch，然后设置 `num_gpus_per_learner=1`。导入库不会启动 Ray、下载数据或启动模拟器。

## 算法与边界

| Config | 行为 |
| --- | --- |
| `MAACConfig` | 私有 Actor、私有 Critic 编码器/头、共享注意力、反事实 baseline、score-function 更新 |
| `MASACConfig` | 联合信息双 Q、连续路径导数、离散分支精确期望、固定温度 |
| `MADDPGConfig` | 确定性连续 Actor、联合 Q、目标 Actor；只接受 Box 组合 |
| `ISACConfig` | 独立局部软 Critic，复用混合动作和 score-function 核心 |
| `MAPPOConfig` | 原生 RLlib PPO 更新，Actor 只读取 local，Value 读取训练用集中输入 |
| `IPPOConfig` | 原生 RLlib PPO 更新，局部 Actor 和局部 Value |

第一版支持异构向量观测、有限边界 Box、Discrete、MultiBinary、Dict/Tuple 组合动作。联合算法限定本地采样、一个本地 Learner、0 或 1 张 GPU，不支持动态智能体集合或提前独立终止。MAPPO/IPPO 使用 RLlib 原生 PPO 运行管理。

历史观测可使用 `Dict(current=Box, history=Box[H,C], history_mask=Box[H])`，有效历史在前，零填充在后。长度由观测空间决定，不硬编码设备或价格。

MAAC 是 Ye 等（2021）算法的一致性修订实现。原文缺少的网络和优化参数已明确补足，不宣称复现论文环境、成本下降比例、300 户结果、纳什收敛或正式隐私保证。共享注意力参数使整体计算随智能体数量线性扩展。

## 推理与恢复

联合算法支持回合边界 `save_to_path()` / `restore_from_path()`，包含在线/目标模型、优化器、Replay 和 RNG。仅加载可信的本地 checkpoint。

```python
from raymarl.inference import ActorBundle, export_actor_bundle

export_actor_bundle(algo, "./ray_results/actors.pt")
actors = ActorBundle("./ray_results/actors.pt", device="cpu")
actions = actors.actions(local_observations)
```

PPO Actor 导出要求策略 ID 与智能体 ID 一一对应，导出不包含 Value/Critic。使用 `register_algorithms()` 显式注册 Tune 名称。

## 主分支与实验

主分支只包含 `src/raymarl`、核心测试、通用文档及打包配置。CityLearn、数据下载、机器安装配置、多种子实验和绘图代码在独立归档分支保存。原始数据、模型、日志和 venv 不入 Git。

运行核心门禁：

```powershell
python -m pytest -q
python -m ruff check src tests
python -m build --no-isolation
python -m pip check
git diff --check
```

详见 [算法审计](docs/algorithm-audit.md)、[验证记录](docs/validation.md)和[归档规则](docs/archive-policy.md)。
