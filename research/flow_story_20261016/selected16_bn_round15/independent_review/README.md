# R15 S8 独立工程审计

本目录只做 CPU 来源、执行工件与源码核验，不 GPU、不 SSH、不读取或排名 AEE。固定原 271482、selected16 固定/62current、native8；0训练/新校准/FF/dual。

`SOURCE_STATE_V1.json` 由真实原 PT 和 K 自身72统计中精确48子集独立重建525状态。`IMPLEMENTATION_REVIEW_V1.md` 与 `INDEPENDENT_CPU_REVIEW_V1.json` 给出源码与六项小模型/元数据独立复跑。`audit_selected16.py` 不导入作者runner/core，只复用已冻结独立R14/R13审计的原预测读取、几何及摘要函数；自身四项审计CPU测试包含真实W+48重建，以及8条完整记录、78 BN形状、16/62状态的破坏拒绝。

审计器在读取实际S8结果前冻结到 `PRE_RESULT_FREEZE_V1.json`；root执行源固定在 `EXECUTION_FREEZE_V1.json`。执行命令：

```
python -B independent_review/audit_selected16.py --stage audit --freeze independent_review/PRE_RESULT_FREEZE_V1.json --result run_v1/results.json --output independent_review/ACTUAL_EXECUTION_AUDIT_V1.json
```

实际路径由root完整回收通知确定，不改变来源SHA合同。报告不得覆盖，冻结审计器不得读结果后原位修补。若实现或审计失败，保留失败并另建明确版本。

实际审计应读取原PT、K72和保存48数组，核非BN参数/affine及完整525态；读取8原预测和16既有同W缓存原预测，逐条核实际native8、每FC4,464,000行、78 BN输入形状和每次调用。归档数量为8质量NPZ+1统计NPZ，不能把统计工件当第9帧。中间四头/八branch仅有摘要记录；没有CPU重演完整神经网络。

执行通过仅支持记录和来源闭合，S8本身无压缩动作；主质量分母为R14同W current78，由另一位内部审者按原GT全分区独立复算。新增62current图的8处统计屏障仍须成本化，不能套旧54-BN分母或宣称硬件收益，也不自动准入H0。
