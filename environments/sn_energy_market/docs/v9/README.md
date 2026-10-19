# v9 六维 UC 调度环境与建模

提交 `33f27b7` 的[完整规模复验与一致性审查](full_validation_current_commit_zh.md)已完成：207 项测试、100,000 组静态独立对照、10,000 组非法输入、1,094,400 个动态物理时段全部通过；未发现 v5 兼容运行分支。[机器汇总](full_validation_current_commit.json)绑定此次提交、来源及全部批次哈希。当时列明的三处文档接口名称／引用差异已在当前正文中对齐。

当前已增加原生 RLlib 双层训练，详细接口和运行方式见[训练说明](../../../../experiments/sn_energy_market/TRAINING.md)，实际测试和训练验收见[训练验证报告](training_validation_zh.md)。环境内核保持既有机制，新增学习实现位于独立算法包。

当前运行接口与关键校验的最新调整见[兼容性与简化审查](compatibility_audit_zh.md)：只保留 `UCAction`、`ConsumerAction`，按字段名构造，不保留旧机制兼容层。

v9 环境已按完整建模文档实现，保留 UC 六维物理行动、Consumer 四维行动、联合可行域、一次公告与两层市场。正文按 UC 的 Observation—Action—Reward、Consumer 的 Observation—Action—Reward、市场和环境执行组织，保留八部分 26 章的详细论述。

- [完整建模文档](modeling_complete_zh.md)与[离线 HTML](modeling_complete_zh.html)。
- [实施与验证报告](implementation_validation_zh.md)与[机器可读记录](implementation_validation.json)：实际测试规模、残差、修复与保留问题。
- [公式—实现—测试对应表](formula_implementation_map_zh.md)：式（1）—（29）及环境工程契约。
- [环境使用说明](../../README.md)：动作、观测、配置、数据、终止和恢复。
- [HTML 检查](html_validation.json)：公式、目录、桌面与移动端离线显示。

当前生产验收在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m pytest environments/sn_energy_market/tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B -X utf8 experiments/sn_energy_market/validate_v9.py --workers 4
```

环境验证输出位于 `outputs/snmarket-validation/`，训练输出位于 `outputs/snmarket-training/`，分别绑定当时源码、配置和规格哈希。主合成组供给考核系数为 2.00，基础 DR 服务估值为 0。U02、A01、A02、A03 按现有数学定义保留，数值复现与经济解释分别列示。

## 历史文档审查档案

[U03 文档复算](u03_revision_validation.json)、[原参考结果](verification_results.json)、[机制审查](modeling_audit_zh.md)、[成本单位审查](cost_units_review_zh.md)和[全篇精简审查](simplification_review_zh.md)保留原始内容。历史 verifier 含“生产代码未改动”的旧哈希断言，记录的是当时仅修改文档的工作，不再作为已迁移环境的验收入口。旧保护清单和报告保持，当前实施证据以新验证报告为准。
