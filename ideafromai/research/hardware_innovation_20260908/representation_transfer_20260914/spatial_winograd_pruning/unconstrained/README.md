# 固定 U_m2 置零：有效的 phase3 强控制

已完成固定系数、静态界、135 个原 tile 与 36 个 live 记录的 raw/J/wide/I24 gold；656640 个 raw p2 全部与独立两相位 expanded-W 相同。没有训练、活动选择、比例扫描、GPU 或 RTL 运行。与 moment/native_tap 的局部比较已到齐：这条自由 U 置零控制更接近母 Q11，应保留；网络 AEE 和实际周期分别由根及硬件负责人验证。

| 同一输入，对 Q11 母函数 | unconstrained | moment | native_tap |
|---|---:|---:|---:|
| 135 tile：BN尺度局部 RMSE | 0.166056 | 0.211799 | 0.326990 |
| 135 tile：I24 relative L2 | 0.115428 | 0.147225 | 0.227297 |
| 36 live：BN尺度局部 RMSE | 0.123452 | 0.174268 | 0.268402 |
| 36 live：I24 relative L2 | 0.095584 | 0.134929 | 0.207814 |

模型 `factors.npz` 明确 `function_type=phase3`，**没有 q2 字段**。`physical_coeff3[96,16,3]` 是母 Winograd 分量 `[U0,U1,U3]`，不是普通水平三tap；固定清除全部 192 个 `(rank,N8)` 的 U2，剩余 576 个 N8 向量均非全零。实际 coeff 范围 ±2981，在 signed13 内。q1、θ、bias、BN_gain/offset、b_q20、Z界保持母模型；`output_scale=母/2`，`a_q40=RNE(output_scale×BN_gain×2^40)`。母 a 有51个奇数，新 a 有26通道不同于母 a 整数向下除2，因此必须加载新配置。

令 `D=[z0−z2,z1+z2,z1−z3]` 对应原分量0/1/3，`Mk=Σr Dk×Uk`。硬件 raw **p2_even=M0+M1，p2_odd=M1−M3，不除2**。gold 的 `p_int` 存 raw p2，`physical_M_int[...,3]` 同样按0/1/3；最后仍只执行一次既有 RNE26/I24。普通 factor 不能消费这份系数。两相位水平权重分别是 `[U0,U1,U1−U0,0]` 与 `[0,U1−U3,U1,U3]`；与 q1 展开的 `expanded_phase_int32[2,96,96,3,4]` 保存 phase/output/input/ky/kx。phase=tile 内 x=0/1，原输入输出 tile 的全局起点均为偶数；整图实现必须保持这个相位锚点。

独立母函数验证还给出精确误差式：`p2_even−2p_even=−M2`、`p2_odd−2p_odd=+M2`，每对水平 raw 输出和严格守恒。这是带固定两相位的近似线性算子，不能假装成平移不变1×3卷积；也不能因为它不是这种卷积就排除。135/36 的两个相位 I24 RMSE 分别为 2720.654/2720.656 与 2022.632/2022.630；完整逐相位误差在 `SUMMARY.json`。raw p2 的奇数值为221362/35198，均合法，无需除2或新增中间RNE。

| 静态算术合同 | 保守绝对界或范围 |
|---|---:|
| 固定 q1 的 Z signed15 | [−8562,7427] |
| D（送现有 signed19 乘法端） | 17124 |
| M 任意 rank 前缀 signed32 | 239174076 |
| 恢复与跨 stripe p2 任意前缀 signed32 | 424629054 |
| 两相位 binary 展开最终 p2 | [−65823737,80854640] |
| wide signed64，含全 signed32 J 与固定 b | 2284193185108974 |

证明不依赖这171个输入：对每rank有 `|D|≤2max(−Zlo,Zhi)`，M前缀按绝对乘积全和界定；任意恢复或条带累计由 `max(M0abs+M1abs,M1abs+M3abs)` 覆盖。独立展开的正/负系数和给最终相位界，`p_lower/p_upper[96]` 取两相位 envelope 供旧消费者复用。样本观测与静态界分别保存，不混用。

`gold_tiles.npz` 是原135输入；`gold_sequences.npz` 是 `../../quality/q11/sequence_tiles.npz` 的同36记录，来源元数据在 `sequence_tiles.json`（18序列×tile128/9664）。母模型的P/I24只作误差分母，候选gold全部重算。另新增 `../moment/gold_sequences.npz` 与 `../native_tap/gold_sequences.npz`，不改变其冻结系数或原135 gold。每臂各自的三种数值路径对应各自函数；静态U组比例不等于周期收益，也不等于网络误差预算。

可机读合同 `manifest.json`，三臂同输入表 `comparison.csv`，界 `static_bounds.json`，完整结果 `SUMMARY.json`，文件读回核验 `saved_artifact_check.json`。复现依次执行 `/opt/anaconda3/bin/python3.12 export.py` 和 `/opt/anaconda3/bin/python3.12 check_saved.py`。这份导出是成熟 Winograd 域剪枝的必要强控制，不主张新 WINS。
