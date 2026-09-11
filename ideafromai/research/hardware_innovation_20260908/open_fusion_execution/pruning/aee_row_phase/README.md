# 水平 P2 共掩码、纵向两相位：实际 AEE

2026-09-12，A800，未训练。两个既有 fixed 学生各跑非校准帧 smoke，再跑原 diverse10 的 `global_joint_pair` 与 `row_phase_joint_pair`。后者在同一水平 P2 内使用相同 H8 mask，仅随源像素行奇偶改变。这是新的执行接口；本轮没有继续调整此前恢复训练的步数或参数。

| 学生 | mask | 10 帧均值 AEE | 相对未剪枝 | 相对本轮 global | 排除校准源帧的 9 帧 ΔAEE（对 global） |
|---|---|---:|---:|---:|---:|
| ordinary | global | 1.214952869 | +0.053765794 | 0 | 0 |
| ordinary | row phase | 1.217386459 | +0.056199384 | +0.002433590 | −0.004976398 |
| lifting_raw | global | 1.254435748 | +0.067850231 | 0 | 0 |
| lifting_raw | row phase | 1.247093637 | +0.060508120 | −0.007342111 | −0.002725703 |

目前两臂 row mask 相对未剪枝都未达到 +0.005。9 帧对 global 有小幅改善，但这不能替代精度准入。四窗口执行原型中消除水平广播税的结果由旁边 `paired_phase_execution.json` 单独记录；本表不把软件运行时间或发放比例当作周期收益。

均值权重会改变相对方向，不能只挑有利的一种：ordinary 的像素加权 AEE 为 global **1.233881021**、row **1.227750905**，与帧均值方向不同；lifting 的像素加权 AEE 为 global **1.261962663**、row **1.267011434**，也与帧均值方向不同。原 evaluator 两种指标均保存。

两轴重测 global 的每个逐帧 AEE 都与旧 `aee_results/global_group2` 完全相等（最大差 0），新旧 global mask 也逐元素相同。未剪枝引用此前实际同学生、同帧重测结果：ordinary 1.161187076，lifting 1.186585517。施加 mask 前的逐帧 sn2 发放数同样完全复现。这里使用原始两个 fixed 学生，没有使用 `paired_recovery/stage64` 的恢复权重。

实际 sn2 `{0, θ}` 输出按 `2*(source_y%2)+source_x%2` 施加返回值 mask；原始 I24、真实 fixed Conv2 和两个后续 gate/PED 消费者、后续动态 BN 与粗头网络继续完整执行。未训练、未更改量化、未覆盖任何旧结果。diverse10 只有 516,735 个有效像素，包含一帧 mask 校准源帧，因此额外列出余下 9 帧；不是 valid825 或独立数据集。

入口支持可选 `--modes`，默认仍为原三个模式；mask 无 `drop_groups` 字段时从实际连续 H8 位图导出组号：

```bash
env312/bin/python3.12 -u open_fusion_execution/pruning/evaluate_aee.py \
  --root /root/private_data/work/hardware_innovation_20260908 \
  --mask open_fusion_execution/pruning/paired_phase_masks.json \
  --output open_fusion_execution/pruning/aee_row_phase \
  --modes global_joint_pair row_phase_joint_pair
```

`run.json`、各学生子目录保留实际逐帧输出；`summarize.py` 生成 `comparison.json` 并核对旧 global 复现。当前结果限制本轮免训 mask，不否定水平 P2 共掩码的执行接口。
