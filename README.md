# Ray RLlib 本地环境

按[指定的 RLlib 文档](https://docs.rayai.org.cn/en/latest/rllib/)部署：Python 3.12 的 `venv`、Ray 2.59.0、PyTorch 和 Dashboard。这里提供的是 Windows 单机运行环境。

## 运行示例

在 PowerShell 中执行（无需先激活虚拟环境）：

```powershell
cd D:\Yang\RayMARL
.\.venv\Scripts\python.exe .\quickstart.py
```

示例沿用文档的 `Taxi-v3`、`PPOConfig` 和 `FlattenObservations`。为本机验证缩小了每次训练的采样批次、更新轮数和评估回合数；默认训练两次，CUDA 可用时使用一张 GPU，否则使用 CPU。可通过 `--gpus 0` 强制使用 CPU。

当前默认在本地采样、评估，减少 Windows 的进程和内存开销。内存充足时可用 `--env-runners 2` 启用文档中的两个并行采样进程；本次 GPU 验证采用本地采样。两个远程采样进程的 GPU 训练尝试曾发生内存分配失败，不视为验证通过。

```powershell
.\.venv\Scripts\python.exe .\quickstart.py --iterations 10
```

输出保存在 `artifacts/quickstart-result.json` 和 `artifacts/checkpoint/`。少量迭代用于确认环境、训练、评估和保存链路可运行，不代表策略收敛。

## 激活 venv

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip check
python .\quickstart.py
deactivate
```

如果当前 PowerShell 禁止执行激活脚本，可继续直接调用 `.venv\Scripts\python.exe`。

## Ray Dashboard

```powershell
.\start-ray.ps1
.\.venv\Scripts\python.exe .\quickstart.py --address auto
```

浏览器访问 <http://127.0.0.1:8265>。此节点配置四个 CPU 资源；训练完成后节点继续运行。独立运行示例时，示例创建的临时节点会自动关闭。

Dashboard 开启 token 认证。如果页面提示登录，在本机 PowerShell 中执行 `Get-Content .\.ray\auth_token`，将结果粘贴到登录框；不要分享该文件。

停止本机 Ray 节点（会停止本机其他 Ray 会话，请确认没有其他训练）：

```powershell
.\.venv\Scripts\ray.exe stop
```

## 重建环境

```powershell
.\setup.ps1
```

基础解释器当前使用 Codex 提供的 Python 3.12，路径记录在 `.venv/pyvenv.cfg`。移动到其他机器时重新创建 venv，并指定本机解释器：

```powershell
.\setup.ps1 -BasePython 'C:\Python312\python.exe'
```

`requirements.txt` 固定直接依赖；`requirements-lock.txt` 记录验证后的完整版本。

Ray 的认证文件保存在 `.ray/auth_token`，运行临时文件保存在 `.tmp/ray`，均已加入 Git 忽略规则。脚本保留 token 认证，使用 `RAY_AUTH_TOKEN_PATH` 指向本地认证文件。Ray 2.59 的日志目录通过 `StorageContext` 指向工作区，避免向用户目录写入；该接口为版本相关的开发者 API，升级 Ray 时需要重新验证。

## GPU

当前安装官方 `torch 2.14.1+cu130`，CUDA 运行时为 13.0，使用 PyTorch 官方 CUDA wheel 索引。示例在 CUDA 可用时默认使用 GPU，也可显式传入 `--gpus 1`；它会检查 CUDA 可用性及 Learner 模型参数的实际设备。

```powershell
.\.venv\Scripts\python.exe .\quickstart.py --address auto --gpus 1
```

参考环境 `D:\Yang\BaseModel\MARL\.venv` 的包文件记录为 `torch 2.14.0+cu130`，适配 Python 3.11。该环境当前无法启动，因为 `pyvenv.cfg` 指向的基础 Python 3.11 不可用。当前环境是 Python 3.12，因此使用对应的官方 `cp312` 包独立安装。

Ray 的 Windows 支持仍属 Beta，多节点 Windows 集群不在本次部署范围。参考[安装文档](https://docs.rayai.org.cn/en/latest/ray-overview/installation.html)。

## 本次验证

2026-10-05：Python 3.12.14，Ray 2.59.0，PyTorch 2.14.1+cu130，CUDA 13.0，Gymnasium 1.2.2，NVIDIA GeForce RTX 4060 Laptop GPU。`pip check` 无依赖冲突，CUDA 矩阵计算、前向/反向传播和 Adam 更新通过。

连接本地节点的 PPO GPU 示例已完成两轮训练、评估和模型保存，共采样 1,024 步，Learner 模型参数设备为 `cuda:0`。CPU 版安装阶段曾验证独立启动及连接节点的训练链路。当前 Dashboard 首页及带认证的版本接口均返回 HTTP 200，`torch.cuda.is_available()` 为 True。

Ray 自身仍输出新 API、内部 RLModule 弃用及 actor 重启提示。本次正常运行通过，未验证故障恢复或长期训练。
