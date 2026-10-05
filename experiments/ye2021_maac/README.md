# 归档验证材料

所有算法通过主分支 raymarl 的 RLlib Config/Algorithm 运行。这里保存 CityLearn 双 venv、官方配置、真实博弈资格测试和串行实验计划，不属于可分发算法包。

训练侧 Python 3.12 / RLlib 2.59 / PyTorch 2.14.1+cu130。环境侧 `.venv-citylearn` 使用可移植 Python 3.11.17 / CityLearn 3.0.2 / Gymnasium 0.28.1 / CPU PyTorch 2.4.1。

```powershell
$env:PYTHONPATH='D:\Yang\RayMARL\src;D:\Yang\RayMARL\experiments\ye2021_maac'
.\.venv\Scripts\python.exe experiments\ye2021_maac\preflight.py
.\.venv\Scripts\python.exe experiments\ye2021_maac\suite.py --stages smoke
.\.venv\Scripts\python.exe experiments\ye2021_maac\pipeline.py
```

公开数据固定 v3.0.2 的 `citylearn_challenge_2022_phase_all_plus_evs`，原生 MARL 逐户奖励、不改设备动力学。36/8/8 周训练/验证/测试。排错 3 户，正式和参数选择 5 户。

pipeline 顺序：两轨道各算法 3 seeds×5000 联合步排错；等额 6 配置×10000 步验证集选择；5 seeds×50000 步正式比较；混合轨道 3 seeds×50000 步注意力、baseline、历史编码消融。Cen-SAC 单独作为集中信息参考。MADDPG 只跑连续轨道。完整回合使实际步数可能略大于预算，原生 168 行产生 167 次转换。

结果和模型在 outputs，数据在 datasets，均不提交。进度 `outputs/pipeline-progress.json` 明确 running/failed/complete。完成后 finalize 把结果摘要、配置、每个结果 SHA256 和来源哈希追加到验证归档分支并打不可变标签，不切换主工作区分支，不修改 main 索引。

成本与服务 KPI 同时保留，不把少服务带来的收益改善解释为更好的能量管理，不把短跑或稳定曲线当成机制完整、收敛或纳什证明。检查 case 结果而不是仅看进程启动。恢复时校验源码和配置签名。
