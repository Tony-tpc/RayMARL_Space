# v8 建模与核验

本版使用 **DER PAB＋UC 公告零售＋自动余量供电**。没有新增逐户签约流程，当前生产环境仍为 v5。正文已按 v5 的六部分、26 节汇报结构重写；这次重写保留 v8 的机制和 29 组核心公式。

- [完整建模文档](modeling_complete_zh.md)：本版唯一执行规格。
- [再审查报告](modeling_audit_zh.md)：旧问题逐项结论、规则取舍、验证与边界。
- [独立核验程序](verify_v8_mechanisms.py)与[结果](verification_results.json)：21 个确定性算例、5000 个静态样本、100×96 个连续物理时段。
- [文档验证](document_validation.json)：结构、链接、公式渲染和来源/生产文件保护。
- [重写一致性验证](rewrite_validation.json)：旧新公式编号对应、内容覆盖与核验程序未变记录。
- [受保护文件基线](protected_files_manifest.json)：核对本轮未改动环境和算法。

在 RayMARL 根目录重跑独立数学核验：

```powershell
.\.venv\Scripts\python.exe .\environments\sn_energy_market\docs\v8\verify_v8_mechanisms.py
```

该命令只写本目录的核验结果，不运行训练、不调用生产环境。它会核对本轮受保护文件快照；后续实施环境变更后，这项保护检查预期会失败，应另建对应新版本的环境验收，而不是改写旧快照。
