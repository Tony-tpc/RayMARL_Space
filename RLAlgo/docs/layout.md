# 工作区目录整理记录

日期：2026-10-09。按用户要求，将算法项目集中到 `RLAlgo/`，在工作区根目录提供独立的 `environments/`。

## 路径对应

| 原位置 | 整理后位置 |
| --- | --- |
| `src/raymarl/` | `RLAlgo/src/raymarl/` |
| `tests/` | `RLAlgo/tests/` |
| `docs/` | `RLAlgo/docs/` |
| `pyproject.toml` | `RLAlgo/pyproject.toml` |
| 原算法 `README.md` | `RLAlgo/README.md` |
| `docs/examples/train_compare.py` | `RLAlgo/examples/train_compare.py` |
| `docs/examples/README.md`、`validation.json` | `RLAlgo/examples/` |
| `docs/examples/energy_env.py` | `environments/energy_env.py` |
| `train_aaa.py` | `RLAlgo/examples/train_aaa.py` |
| `local_runtime/`、`archives/` | `RLAlgo/local_runtime/`、`RLAlgo/archives/` |

根目录的普通目录为 `RLAlgo`、`environments`、`experiments`、`datasets`、`outputs`，普通文件为新的工作区 `README.md`。`.gitignore`、虚拟环境和运行状态目录保留在根目录。原根目录缓存归入 `RLAlgo` 或 `.tmp`。

## 路径与接口

`raymarl` 的 Python 包名及六种算法的公开导入方式保持一致。当前 `.venv` 已重新以可编辑方式安装 `RLAlgo`，实际导入位置为 `RLAlgo/src/raymarl/__init__.py`。

示例入口使用自身文件位置定位工作区，不依赖当前工作目录；环境通过 `environments.energy_env` 导入，输出仍归入工作区 `outputs`，Ray 状态写入工作区 `.ray`、`.tmp`。`train_aaa.py` 从 `environments.aaa` 导入用户提供的 `aaa` 类。

本机运行辅助脚本的虚拟环境、认证和临时目录路径已更新。实验脚本的源码扫描路径已更新为 `RLAlgo/src`，摘要仍使用原 `src/raymarl/...` 相对路径，以免仅目录迁移就改变算法摘要。历史验证 JSON 保留当时的路径和数据。

## 本次验证

- 迁移前登记的 **85 个文件全部保留**；算法包 **51 个文件**及电池教学环境的 SHA256 与迁移前一致。
- 完整测试：**46 passed**，含六种算法的原生训练、评估、保存恢复、Actor 导出和 checkpoint 重建。
- 从 `RLAlgo` 目录运行完整示例：单智能体 PPO 加六种 MARL 算法，全部完成训练、评估、保存、新建算法恢复续训和独立推理；每个推理回合均为 8 步。
- Ruff 与 `pip check` 通过；相关 Python 和 PowerShell 辅助脚本语法检查通过。
- wheel/sdist 构建通过；wheel 仅包含算法包和发行元数据，源码与 checkout 一致。安装到独立目录后，六个 Config 均可导入。
- 主目录结构及更新后的 README 链接检查通过。

示例验证参数为 `--mode both --algorithm all --iterations 1 --horizon 8 --batch 16 --history 4`。此次验证说明目录调整后的接口与生命周期可运行，不是长期训练收敛或论文结果验证；现有大型实验流程未重新执行。

机器可读摘要见 [layout-validation.json](layout-validation.json)。本轮临时构建、测试和示例产物集中在根目录 `.tmp/layout-*`，未写入既有实验输出目录。
