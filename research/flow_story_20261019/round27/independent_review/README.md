# R27 独立科学、身份与原数组质量复算

本目录不训练模型、不启动 GPU。原始训练成员和开发成员的 metadata-only 生成在 `build_membership_v1.py`；`membership_v1.json` 固定 train256/dev64、每臂512次更新的完整身份/crop顺序。既有 sealed8/H8 不进入本轮数组访问。原骨干见过这些训练场景，dev64只独立于本轮恢复与登记的旧诊断身份，不是官方验证集或新场景测试。

`ANALYSIS_FREEZE_V1.json` 在新开发质量前固定六状态、全部16区域、6144个帧区域指标、864个状态/作用域/区域汇总，以及四聚合门、双参考和支持分层。`analysis_core.py` 仅处理标量；`analyze_metrics_v1.py` 可以重新汇总已审的64帧六状态记录，不替代原数组来源审计。

`raw_quality_math.py` 从原 GT/mask/events 重建冻结的16区定义，边界与位移阈值沿训练D200，不在开发集重拟合。独立与旧冻结数学逐项合成核对；全部有效GT包含超过400px位移，无额外截断。事件字段 `voxel_nonzero_bins` 是非零体素bin数，不是原始事件数量。

`reduce_raw_v1.py` 先要求 root 独立执行审计器的实际完整PASS，核其源freeze、三个512步结果/ledger、所有192原始文件SHA及384实际head3原件，再首次读取 GT 数值。只用head3的像素单位float32预测重新算误差，不引用runner质量。GT、有效mask、体素支持和16区mask共享保存一次每帧；384预测保留原归档引用。全部中间PSN/整数张量的数值证明范围属于源审与运行时逐字节比较收据，不能声称本质量脚本重读了它们。

完成后的命令（实际目录由 root 确认）：

```bash
PY=/opt/anaconda3/envs/pytorch310_cpu/bin/python
$PY -B prepare_raw_inputs_v1.py --actual-root /absolute/completed/runs --audit /absolute/actual_execution_audit_v1.json --output ACTUAL_INPUTS_V1.json
$PY -B reduce_raw_v1.py --inputs ACTUAL_INPUTS_V1.json --output actual_quality_v1
```

上述两个程序都拒绝覆盖既有结果。尚未完整回收/审计时不能运行质量命令。可单独运行源核验，不加载实际预测或 GT：

```bash
$PY -B reduce_raw_v1.py --inputs unused.json --output unused --source-preflight-only
```

输出必须同时显著报告 sequence-balanced B/L 和高支持尾部警戒；`AGGREGATE_ORDINARY_RECOVERY_PASS_REVIEW_FLAGS_SEPARATE` 不是“全部帧/全部质量通过”。原R26结论不变，当前ordinary恢复不证明光流专用评分、完整训练收敛或实际硬件净加速。内部 TVLSI/TCAS-II 标准评审也不是正式期刊外审。
