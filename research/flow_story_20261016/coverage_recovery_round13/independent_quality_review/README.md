# R13 独立原始质量复算：运行前合同

本目录由独立质量评审代理编写，**不导入 R13 runner/analyzer，也不导入 R12 evaluator/analyzer 的函数**。依据固定的 `ANALYSIS_PROTOCOL.md` 与原指标定义重新实现区域、逐帧 AEE、总体/逐序列聚合、四类合取门。编写时正式恢复结果尚未生成；没有读取新 check 的事件、GT、预测或质量。检查输入的receipt元数据已经到齐，源数组留待全部八个正式结果完成后统一打开。

## 固定输入与范围

- manifest SHA：`eba4544fe99c082d7d099f90d03470d7abad22b01b68d3eb4d022e61e78151d1`。
- analysis protocol SHA：`5ec51175fbf18b9d2f4fab329247b011700f4cf109b50bf73e905708698c2d7b`。
- 新 D200 分区定义 SHA：`2e4934f6d82663f16ca1ae6c3aea238a31e699d4f49916052f6ea6b5e9c1d41b`。
- 原始24数组的 `input_snapshot/receipt.json` SHA：`2188cce3a09235dc5c96b793e94c025433b4154bd09ffd8ee15c2dcd9a3df8b6`；只读取过该receipt的metadata/schema，尚未打开数组。

预定正式目录：`raw/experiments_v1/{reference,diversity}/recover/{native,FF,CC,fine_s2b2}/results.json`，prediction NPZ 与相应 JSON 同目录。原始数组保持 receipt 的 `relative_path` 目录层级，位于 `input_snapshot/`。

所有八份结果必须存在且状态完成，才进行质量复算。完整范围是128 NPZ、320 float32 预测、40状态/布局、16区域、5120逐帧区域指标与1920总体/逐序列区域汇总；正式前向1964、正式更新1600。BN176和工程smoke另作执行审计，不能当独立样本或混入此正式质量矩阵。

## 独立计算与判断

原始事件 `(10,480,640) float32`、GT `(2,480,640) float32`、mask `(480,640) bool` 逐文件核SHA、大小、shape、dtype、finite及非空allGT；再与每份预测记录的 `input_evidence` 比对。预测逐NPZ和逐payload验证SHA、shape、dtype、finite。

所有有效GT以float64计算二维端点L2，记录数量/error_sum/AEE。B用有效四邻域向量最大跳变、严格大于固定阈值，未知邻域进入B_unknown；L按有效GT幅值严格大于固定阈值。voxel活动为逐bin `any != 0`，不把跨bin求和抵消解释为无活动，更不声称原始事件数。全部16区域保留。

聚合主量为非空区域的逐帧平均AEE，另报像素加权量和空区帧数。四类门逐条件、总体和两条序列分别计算：N/F正常；C/P相对两种匹配恢复参考；C/P相对自身N/F布局；C/P权重的N/F布局相对对应N/F训练参考。allGT≤+3%、B/L≤+5%，全部合取；空区不构成改善证据。

完整保留各臂自身布局相对初始的每帧增减、胜/平/负、signed gain集中度；同时输出所有20状态/布局的D−R和相对变化，不选择有利分母。跨八臂初始完整state SHA必须相同，跨条件32个同布局/同帧初始预测payload SHA必须相同。200步身份顺序逐项对照manifest。

与runner记录数值的复算核对容差在新质量前固定为 `atol=1e-10, rtol=1e-12`；输出最大绝对/相对偏差及位置。门判断本身使用原先精确阈值，不用复算容差放宽质量门。任何缺臂、缺帧、支持变化、SHA不符、数值超差或初始公平性失败都会终止，不能写PASS结果。

## 自检与执行

四项合成检查覆盖：严格阈值与边界未知、跨bin正负抵消的活动定义、3-4-5端点误差、空区域、帧均与像素权重差别、四类门各自失败、逐序列大位移失败，以及容差/空值拒绝。结果见 `synthetic_checks_v1.json`。它们不导入实际模型、数据或结果，不能替代原始端点审计。

全部正式结果完成后执行一次：

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B \
  research/flow_story_20261016/coverage_recovery_round13/independent_quality_review/reduce_raw_quality.py \
  --run-root research/flow_story_20261016/coverage_recovery_round13/raw/experiments_v1 \
  --input-root research/flow_story_20261016/coverage_recovery_round13/input_snapshot \
  --input-receipt research/flow_story_20261016/coverage_recovery_round13/input_snapshot/receipt.json \
  --regions research/flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json \
  --output-dir research/flow_story_20261016/coverage_recovery_round13/independent_quality_review/actual_reduction_v1
```

程序默认校验本目录 `freeze_v1.json` 的源码/测试/文档SHA；输出新目录拒绝覆盖。最终内部TVLSI/TCAS-II评审应在全部真实结果、复算和执行审计完成后撰写，当前不提前判定科学成功。原骨干见过全部train，仍不称官方测试、独立场景、统计显著性、光流增量或硬件净收益。
