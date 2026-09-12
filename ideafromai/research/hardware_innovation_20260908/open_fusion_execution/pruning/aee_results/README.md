# 相位 H8 剪枝：实际网络 AEE 探针

2026-09-12，A800，未训练。两个既有 fixed student 都完成：先 1 个非校准帧 smoke，再按原 diverse10 清单分别跑 unpruned、global_group2、phase_joint。每种模式的 10 帧共 516,735 有效像素。不是 valid825，也不是冻结 ep34 原模型的论文精度表。

| 学生 | 模式 | 10 帧平均 AEE | 相对同学生 unpruned | 不含校准源帧的 9 帧 ΔAEE | 删除的实际非零 sn2 发放 |
|---|---|---:|---:|---:|---:|
| ordinary | unpruned | 1.161187076 | 0 | 0 | 0 |
| ordinary | global_group2 | 1.214952869 | +0.053765794 | +0.055067153 | 14.003% |
| ordinary | phase_joint | 1.211913715 | +0.050726640 | +0.039300413 | 14.405% |
| lifting_raw | unpruned | 1.186585517 | 0 | 0 | 0 |
| lifting_raw | global_group2 | 1.254435748 | +0.067850231 | +0.062564069 | 13.879% |
| lifting_raw | phase_joint | 1.229378978 | +0.042793460 | +0.043195785 | 13.953% |

相位方案比全局方案 AEE 更好：ordinary 改善 0.003039154，lifting 改善 0.025056771。按2026-09-12用户[新精度规则](../../ACCURACY_POLICY.md)，本表全部臂均优于同diverse10的原SDformerFlow本地复现NB0（1.454602861），恢复继续考虑资格。相对各自unpruned的损失照报，旧+0.005不再作为淘汰理由。剪掉发放的比例不是周期收益；相位相对全局的实际执行增量由旁边原型独立测量。

实际改动是 `sn2.spiking_neuron` 返回的 `{0, θ}`：按 `phase=2*(source_y%2)+source_x%2`，每个相位去掉固定 2 个连续 H8 组，所有 T10 一起归零。`θ` 不重新量化。固定 helper 的 Conv2 读取这个返回值，随后原始 I24 残差、更新后 I24、投影 gate 和连续 PED 两个消费者、后续动态 BN 和整个原粗头路径均执行。没有只修改 FP shadow，也没有跳过真实消费者。

两轴 unpruned 的每个逐帧 AEE 都与原 `fixed_sources_diverse10` / `fixed_lifting_diverse10` 精确相等（最大差 0）。两个剪枝方案每帧施加前的 sn2 非零数量也与本轴 unpruned 相同。mask 来源是 `../../review/phase_group8_masks.json`，在 `zurich_city_09_a_0001.npy` 的小网格上选取；diverse10 包含这 1 帧，因此额外报告排除它的 9 帧配对结果。其余帧仍来自已有场景和同一验证目录，这不是独立数据集或训练后的泛化结论。

运行入口：

```bash
env312/bin/python3.12 -u open_fusion_execution/pruning/evaluate_aee.py \
  --root /root/private_data/work/hardware_innovation_20260908
```

`run.json` 含全部 smoke/10 帧结果、mask 组号和逐帧实际发放；`comparison.json` 汇总均值、配对变化和 unpruned 复现检查。子目录保留原 evaluator 的逐帧与汇总输出；其中 GPU wall time 仅用于记录执行完成，不是硬件速度。
