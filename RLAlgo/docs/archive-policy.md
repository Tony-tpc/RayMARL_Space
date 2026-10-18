# 主分支与验证归档

主分支内容：`RLAlgo/src`、`RLAlgo/tests`（核心）、`RLAlgo/docs`、`RLAlgo/examples`、`RLAlgo/README.md`、`RLAlgo/pyproject.toml`、工作区 `README.md`、`.gitignore`，以及本轮用户要求独立维护的 `environments/`。算法 wheel 只包含 `raymarl` 包；环境、机器安装脚本和实验代码不进入 wheel。

本地机器安装脚本与历史归档已迁入 `RLAlgo/local_runtime/`、`RLAlgo/archives/`，沿用 Git 忽略规则。工作区保留 `experiments/`、`datasets/`、`outputs/`；其中数据和运行产物不提交。历史验证 JSON 内的路径记录当时的布局，不改写既有证据。

v5环境位于 `environments/sn_energy_market/`，其源码、配置、测试及审查文档可以提交。`experiments/sn_energy_market/README.md` 与 `smoke_test.py` 作为该环境的验证入口单独允许跟踪；其他历史实验和全部结果仍被忽略。

`archive/bootstrap-20261005` 保存初始运行脚本及 CUDA 安装配置。`archive/ye2021-maac-validation` 保存 CityLearn sidecar/proxy、实验矩阵、配置、审计及证据清单；不整体合并入 main。

实验目录在开发工作区被忽略，只有在归档分支受控提交。原始数据、venv、checkpoint、日志、图表、凭据即使在归档分支也不提交。保存校验值与本地产物位置，避免把没有完成的实验记为完成。

实验证据记录算法源码 SHA256、算法提交、环境/数据版本、配置、种子、实际步数、结果文件校验值及失败状态。正式实验完成后追加证据提交和不可变里程碑标签，不移动现有标签。

提交门禁：pytest、ruff、pip check、wheel/sdist 清单、隔离安装的构建/训练/恢复、git diff --check，以及暂存区白名单检查。.gitignore 不会自动取消已有跟踪，需显式移出索引。
