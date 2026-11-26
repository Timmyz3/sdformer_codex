# R30 独立质量入口

[预结果协议](PROTOCOL_PRE_RESULT_V1.md)、[源代码/科学短审](SOURCE_SCIENTIFIC_REVIEW_V1.md)、[512步日程核算](SCHEDULE_AUDIT_V1.json)已写定。[纯数学冻结](ANALYSIS_FREEZE_V1.json)和[原数组IO冻结](IO_FREEZE_V1.json)均在R30实际质量前完成；[10项CPU检查](CPU_TESTS_V1.json)全通过，只有合成数值和旧元数据，未读新GT。

输出只增加 `initial_M/joint_Q4/joint_Q8/joint_M` 四态256预测。原始/匹配native及独立Q8/Q6/Q4以R27/R28已审标量和512原档SHA明确引用。原64份GT/事件支持/16region档复用，区域阈值不重拟合。`joint_M`双native聚合门是主质量结论；同W全Q8不是额外否决条件。硬件判定由独立合同提供，本脚本不自动放行后续实验。

实际执行完整PASS后，使用绝对结果/审计路径运行：

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B prepare_inputs_v1.py \
  --result <actual-formal-results.json> \
  --audit <actual-formal-execution-audit.json> \
  --output ACTUAL_INPUTS_V1.json
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B reduce_joint_raw_v1.py \
  --inputs ACTUAL_INPUTS_V1.json --output actual_quality_v1
```

运行前可用 `--source-preflight-only` 仅核冻结源码；必须同时提供两个CLI路径参数，但该模式不读取它们。任何不完整/重复state、smoke、源错位、mask合同错位或计数不符均在GT数值读取前拒绝。parity中间数组的执行完整性不由本adapter冒充重算。

[硬件配额短审](HARDWARE_QUOTA_SHORT_REVIEW_V1.json)独立核了11/40共同配额、真完整tile坐标、两成本分母与预留算术。它只复核来源、公式与表格，不是整网DAG或芯片测量；规划1%预测余量不是已实现预测器成本。

实现代理已只读核本IO的实际四state/schema映射及GT读取顺序，无阻断；未复跑本CPU或读取R30实际质量。实际结果齐备后再形成内部TVLSI/TCAS-II评审，不能用这些预结果检查宣称算法成功。
