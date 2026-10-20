# RayMARL 工作区

算法库集中在 `RLAlgo`，环境集中在 `environments`。主目录用于组织算法、环境、实验、数据和输出。

```text
RayMARL/
  RLAlgo/
    src/raymarl/
      algorithms/       # 通用六算法与独立 stackelberg 双层算法包
      common/           # 网络、连接器、回放与运行管理
    tests/              # 算法测试
    docs/               # 架构、算法审计与验证记录
    examples/           # 训练、评估、保存恢复与推理入口
    local_runtime/      # 本机安装与 Ray 启动辅助脚本
    archives/           # 本地历史归档工具与记录
    pyproject.toml      # 算法库的安装、打包与检查配置
    README.md
  environments/         # 独立环境及环境适配器
    energy_env.py       # 单住户 / 多住户电池教学环境
    sn_energy_market/   # v9 六维 UC / DER PAB 两阶段环境项目
  experiments/          # 实验配置与对照流程
  datasets/             # 本地数据集
  outputs/              # 训练结果与 checkpoint
  README.md
  .gitignore
```

虚拟环境及运行状态沿用主目录下的 `.venv*`、`.runtimes`、`.ray`、`.tmp`，避免已有运行环境的路径失效。数据集、输出、机器运行脚本和归档仍按原有规则排除在 Git 提交之外。

## 安装与运行

在 `D:\Yang\RayMARL` 执行：

```powershell
# 安装当前算法源码；Python 中继续使用 from raymarl import MAPPOConfig 等接口
.\.venv\Scripts\python.exe -m pip install -e .\RLAlgo --no-deps --no-build-isolation

# 单智能体 PPO 与双住户 MAAC 的完整生命周期示例
.\.venv\Scripts\python.exe .\RLAlgo\examples\train_compare.py

# 算法测试与静态检查
.\.venv\Scripts\python.exe -m pytest -c .\RLAlgo\pyproject.toml .\RLAlgo\tests
.\.venv\Scripts\python.exe -m ruff check --config .\RLAlgo\pyproject.toml --cache-dir .\RLAlgo\.ruff_cache .\RLAlgo\src .\RLAlgo\tests .\RLAlgo\examples .\environments
```

## 放入自己的环境

将环境文件或完整环境项目放到 `environments/`。例如 `environments/aaa.py` 中定义类 `aaa` 后，可运行：

```powershell
.\.venv\Scripts\python.exe .\RLAlgo\examples\train_aaa.py
```

该入口从 `environments.aaa` 导入环境，并用 MAPPO 训练。环境接口与目录约定见 [环境说明](environments/README.md)。

算法说明见 [RLAlgo/README.md](RLAlgo/README.md)，完整教学流程见 [运行示例](RLAlgo/examples/README.md)，本次整理验证见 [目录整理记录](RLAlgo/docs/layout.md)。

新版博弈环境的安装、接口和验证方法见 [sn_energy_market 环境说明](environments/sn_energy_market/README.md)，逐项依据见 [v9 完整建模](environments/sn_energy_market/docs/v9/modeling_complete_zh.md)。环境采用六维 UC、四维 Consumer 物理动作。双层训练由独立 `stackelberg` 算法包实现，能源场景和运行入口位于 `experiments/sn_energy_market`，现有算法与公共运行逻辑保持独立。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 experiments/sn_energy_market/train.py --smoke
```

该命令完成消费者适应、UC 候选、重新响应、偏离搜索、冻结测试和保存导出。正式实验、恢复及推理接口见[训练说明](experiments/sn_energy_market/TRAINING.md)，验收结果见[训练验证报告](environments/sn_energy_market/docs/v9/training_validation_zh.md)。

当前双层算法每批更新全部 Critic，仅对选中的 Actor 做策略更新。冻结 Actor 时保留其参数和 Adam 状态，候选响应恢复相同消费者 Actor／Critic／Adam。取消额外 MC 预拟合的一次固定预算实验入口为 `experiments/sn_energy_market/online_critic_pilot.py`，结果见[在线 Critic 小规模报告](experiments/sn_energy_market/ONLINE_CRITIC_PILOT.md)，测试集继续保留。

## 本地版本管理

整个工作区由根目录的 Git 仓库统一管理，保留既有算法历史。环境基线位于 `版本管理/环境建模基线` 分支，使用 `环境建模基线-20261018` 标签定位。代码、配置、测试、建模文档、HTML 和文档内的验证摘要均纳入版本；虚拟环境、数据集、运行输出、机器专用脚本及旧实验归档按 `.gitignore` 排除。本次新建分支、标签和提交说明均使用中文。

`.gitattributes` 保持文件当前字节，避免自动换行转换使验证清单中的 SHA256 失效。完整压力测试输出保留在本地 `outputs/snmarket-validation/`，可用已纳入版本的验证入口重新生成。当前仓库仅用于本地版本管理，尚未配置远程仓库。

```powershell
git status --short
git diff
git log --oneline --decorate -5
git show 环境建模基线-20261018 --stat
```
