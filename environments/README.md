# 环境目录

将环境文件、环境包装器或完整第三方环境项目放在这里，与 `RLAlgo` 算法库分开维护。算法通过 `.environment(环境类, env_config={...})` 接收环境。

```text
environments/
  energy_env.py       # 已有电池教学环境
  sn_energy_market/   # v9 Stackelberg–Nash/PAB 独立环境项目
  aaa.py              # 可选：用户自己的 aaa 环境类
  my_energy_market/   # 可选：完整环境项目，保留其内部目录和资源
```

`energy_env.py` 提供 `BatteryCommunity`、`SingleEnergyEnv` 和 `MultiEnergyEnv`，从原示例目录迁入，物理逻辑保持一致。它演示接口与生命周期，不代表能源论文的原始环境或实验结果。

在工作区主目录运行时，导入方式为：

```python
from environments.energy_env import MultiEnergyEnv
from raymarl import MAPPOConfig

config = MAPPOConfig().environment(MultiEnergyEnv, env_config={"households": 2})
```

多智能体环境继承 RLlib `MultiAgentEnv`，定义固定的 `possible_agents`、每个智能体的观测和动作空间。`reset()` 返回 `(obs, infos)`；`step()` 返回 `(obs, rewards, terminateds, truncateds, infos)`，两个结束字典都包含 `__all__`。当前联合算法要求固定、同步行动的智能体。

读取数据或资源时，建议以环境文件的 `Path(__file__).resolve()` 为基准定位，避免依赖训练入口所在的目录。公共数据放在工作区 `datasets/`，实验入口放在 `experiments/`，训练产物放在 `D:/Codex/`。

`RLAlgo/examples/train_aaa.py` 是已有 MAPPO 入口，默认导入 `environments/aaa.py` 中的 `aaa` 类；该环境由调用方提供。

[sn_energy_market](sn_energy_market/README.md) 提供 v9 物理与市场内核及 UC→消费者两阶段接口，由 `raymarl.algorithms.stackelberg` 原生双层算法训练。能源实验的唯一正式训练入口是 `experiments.sn_energy_market.run`。
