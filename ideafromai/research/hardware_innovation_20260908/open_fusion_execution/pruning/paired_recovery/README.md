# 相位 H8：一次同预算定点恢复

2026-09-12，A800。两臂各完成 64 步，固定同一 train16、seed 912 的四个排列、Adam 1e-4 和真实 GT robust EPE；没有扫学习率、步数或按验证集选 checkpoint。两臂从同一个 ordinary 学生开始，分别保留既定 global_group2 / phase_joint mask。

| ordinary 学生 | diverse10 AEE | 相对未剪枝 | 相对本臂免训 mask | 排除 mask 校准源帧的 9 帧 ΔAEE（对未剪枝） |
|---|---:|---:|---:|---:|
| 未剪枝旧基线 | 1.161187076 | 0 | — | 0 |
| global_group2，64 步 | 1.220646668 | +0.059459592 | +0.005693798 | +0.053969769 |
| phase_joint，64 步 | 1.201610389 | +0.040423313 | −0.010303326 | +0.033111998 |

相位方案通过这一次恢复得到改善，比同预算全局方案低 0.019036279 AEE。按2026-09-12用户[新精度规则](../../ACCURACY_POLICY.md)，两臂均优于同diverse10的原SDformerFlow本地复现NB0（1.454602861），可以继续融合；旧+0.005不再阻止精度准入。相对未剪枝的损失与本次恢复收益保留，尚不能据此认定投稿优势。不继续扫步数/学习率。未试的接口仍包括用多帧真实flow目标选择相位组，而不是当前单帧局部PED误差与门翻转评分。

两臂均只训练 Conv2 的 U_R16 / V_R16，共 15,360 个参数；BN gain 合入实际 F。源时间矩阵、源偏置/θ、preview U/V、sn2 时间算术、消费者比较阈值、PED U/V、所有 BN 常量及其余网络全部冻结。实际 sn2 返回值按源像素 phase 施加 mask。原始 I24 残差不受 mask 改写，更新后的 I24 仍进入真实 gate 和连续 PED 两个消费者。

已有训练脚本不能直接用于这个 fixed 学生：fixed helper 的 `@no_grad` 和 `x.ne(0)` 会阻断 preview 的梯度。因此这里新增 `fixed_qat.py`，只给真实定点 Conv2→F→I24→PED 链增加反向 STE；硬前向仍为 q16 系数、原固定指数、Acc48 精确整数运算和每次完成时的 RNE/saturate24。反向使用截断范围内恒等 STE，投影 gate 使用明确的三角 surrogate。没有浮点替换硬前向，也没有把 FP shadow 梯度当成实际消费者梯度。

两个梯度 smoke 都在第一个 TRAIN 帧完成，零次优化更新：

- 整帧 flow、18,432,000 个连续 PED 值、73,728,000 个 projection gate 值，对原 fixed helper 全部零差。
- 两臂 U 都有 11,520 / 13,824 非零有限梯度；V 都是 1,536 / 1,536。mask 禁用的源列自然不产生 U 梯度。
- 最终权重从保存 NPZ 重新加载到原 `FixedTemporalForward`，U/F 的部署指数仍为 17 / 14，所有量化系数对训练硬前向零差，整帧输出也零差。
- 对原部署常量逐字段比较，只有 `U_conv2_theta_q16` 和 `F_q16` 发生变化。源矩阵、θ、阈值、PED 和 BN 的部署常量未改变。

最后的 AEE 来自这个重新加载的原 fixed helper。`diverse10` 只有 10 帧 / 516,735 有效像素，包含 mask 校准源帧，另报告其余 9 帧；不是新 valid825，也不能继承旧模型完整验证精度。后续投影门密度可能随恢复改变，旧硬件时序不能直接继承。没有计时、RTL 或 PPA 的新优势声明。

运行：

```bash
env312/bin/python3.12 -u open_fusion_execution/pruning/paired_recovery/train_pair.py \
  --root /root/private_data/work/hardware_innovation_20260908
```

`stage64/run.json` 保存训练日程、实际梯度与部署复现检查；`stage64/*.npz` 是两个新学生和部署常量；`stage64/*/*frames.json` 是实际部署逐帧 AEE。`summarize.py` 生成 `comparison.json`，对齐原未剪枝和免训 mask。生产树及原训练/推理脚本均未修改。
