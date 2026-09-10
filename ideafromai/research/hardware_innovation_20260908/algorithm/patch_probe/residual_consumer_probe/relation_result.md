# 两个真实 T10 神经元的廉价关系诊断

只分析 A 与 identity-only 默认膜；没有拟合 Conv2 残差，没有许可或硬件加速比。train4 拟合结果是同样本诊断，不能当验证精度。

| 拟合/映射 | 矩阵相对 Frobenius 差 | native 膜重用 RMSE | 默认门差 | 丢失目标非零比例 | 保存膜行数 |
|---|---:|---:|---:|---:|---:|
| static_A_unweighted/identity | 1.56622 | 1.45179 | 63587/983040 | 87.6717% | 10 |
| static_A_unweighted/diagonal_scale | 0.972723 | 0.887375 | 27599/983040 | 99.1743% | 10 |
| static_A_unweighted/one_row_scalar | 0.807336 | 0.750952 | 25967/983040 | 88.5443% | 5 |
| train4_identity_second_moment/diagonal_scale | 1.0317 | 0.819919 | 27598/983040 | 96.5168% | 10 |
| train4_identity_second_moment/one_row_scalar | 0.873335 | 0.657238 | 25148/983040 | 83.1248% | 5 |
| dense_static_inverse_control | 7.32458e-16 | 2.34383e-07 | 0/983040 | 0.0000% | 10 |

源 A 秩 10，条件数 24.421；目标 A 秩 10。稠密逆映射需完整 100 个标量乘法及归约/偏置，不能算便宜变换。

预测显式使用 native sn1 膜以及折合的 bias/center 修正，目标是 CPU FP64 的 Aproj(identity) 默认膜；与真实 residual 后的 proj 输出不是同一个对象。全部来源和 native 数值误差见 JSON。

目标默认门中 27733 个非零；全零默认本来就会得到很高的总体一致率。因此低总门差不能掩盖丢失大多数非零门。当前固定廉价映射没有显示可直接代替 Ap(identity) 的关系；若后续联合约束两套 A，属于新的训练学生，仍需精度、许可和费用验证。

实际 SpikingPEDLayer 的偶数行/偶数列还有连续消费者；即使默认门相同也不构成取消 Conv2 的许可。必须再证明残差界、依赖、相同存储/端口下的净费用及新学生 AEE。
