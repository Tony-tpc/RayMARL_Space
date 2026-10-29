# 版本库与结果边界

正式实现和必要回归测试保存在当前算法实现分支，不合并main。算法wheel只包含raymarl包，环境和实验不进入wheel。

历史分支由Git保存；运行数据、日志、checkpoint、图表、机器缓存和过程报告统一存放D:/Codex，不作为算法源码提交。必要终态和基线保留，按可核对清单清理重复和中间产物。

目录和正式入口以[代码库维护说明](../../REPOSITORY.md)和[当前训练协议](../../experiments/sn_energy_market/TRAINING.md)为准。旧模型不能通过放宽学习契约或兼容入口恢复到新配置。