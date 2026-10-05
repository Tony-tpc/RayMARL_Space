# 主分支与验证归档

主分支白名单：src、tests（核心）、docs、README.md、pyproject.toml、.gitignore。环境、机器安装脚本和实验代码不得进入 wheel 或主分支树。

`archive/bootstrap-20261005` 保存初始运行脚本及 CUDA 安装配置。`archive/ye2021-maac-validation` 保存 CityLearn sidecar/proxy、实验矩阵、配置、审计及证据清单；不整体合并入 main。

实验目录在开发工作区被忽略，只有在归档分支受控提交。原始数据、venv、checkpoint、日志、图表、凭据即使在归档分支也不提交。保存校验值与本地产物位置，避免把没有完成的实验记为完成。

实验证据记录算法源码 SHA256、算法提交、环境/数据版本、配置、种子、实际步数、结果文件校验值及失败状态。正式实验完成后追加证据提交和不可变里程碑标签，不移动现有标签。

提交门禁：pytest、ruff、pip check、wheel/sdist 清单、隔离安装的构建/训练/恢复、git diff --check，以及暂存区白名单检查。.gitignore 不会自动取消已有跟踪，需显式移出索引。
