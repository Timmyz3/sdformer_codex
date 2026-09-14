# 下一接口：保持整数卷积约束的 Winograd 广播组剪枝

**结论：留一个可实施候选，暂评 3/10；成熟 A 占主体，不作为首创新标题。** 本轮仅文献/源码审阅，未跑 GPU/RTL。对象是 θg→竖3×1 R16→Z15→连续横1×3/Winograd→I24；Q13 的 AEE≈1.2598 对 NB0≈1.4479 只是动机，不能视作可消耗的误差预算，也不能替代 Q11 剪枝后的 valid825。

**B：** 原普通 Q2 占 core 约55%；现硬件一次实际服务是 `(rank r, component m, output-group og)` 的 **整个 N8**：只剪其中几 lane，`q2_live` 仍真，周期不降；D 已连续，Bishop 的 binary Q/K 计数误差界不适用。更关键的是，任意单独置零 U 分量可能破坏 `U0,U3∈2Z`、`U1+U2=U0+U3`，令现合同的精确 `/2` 变奇数并失去同一平移共享1×3卷积；这只说明不能直接沿用原表示，**不表示有损 U 剪枝无效或必然更贵**。

**最近邻 A 已覆盖：** [WINS §4/5](https://openaccess.thecvf.com/content/ICCV2025/papers/Park_WINS_Winograd_Structured_Pruning_for_Fast_Winograd_Convolution_ICCV_2025_paper.pdf) 已做逐变换子GEMM向量剪枝、平衡和精度/速度选择；[Spatial-Winograd](https://arxiv.org/abs/1901.02132) 已做空间结构向变换域传递稀疏；[TASD §3/4](https://arxiv.org/html/2403.07953v2) 已做结构主项+残差项及硬件延迟约束选择；[HiNM §3/4](https://arxiv.org/html/2407.20496v1) 已做向量+N:M与通道置换；[Bishop §4/5](https://arxiv.org/abs/2505.12281) 已做TTB训练、权重复用与binary attention误差约束；[量化 Winograd](https://openaccess.thecvf.com/content/WACV2024/papers/Mori_Wino_Vidi_Vici_Conquering_Numerical_Instability_of_8-Bit_Winograd_Convolution_WACV_2024_paper.pdf) 已处理低位宽变换数值问题。不能把这些分别改叫“组发放率”“成本感知”就算 X。

**具体 X 候选：** 在原 Q11 整数三tap `g` 上施加整N8共享约束，使目标 U 分量自然为零：m0=`g0=0`，m1=`g1=−g0−g2`，m2=`g1=g0+g2`，m3=`g2=0`；每 `(r,og)` 至多选一条，且 g 保持±1023。U 始终由 g 重导出，因此普通/Winograd仍同函数、D16/U13及偶数恢复可重新证明。用校准真实 `D[r,m]≠0` 的服务次数与重建后的 p/I24偏差联合选组；残差只在**保留自由度与上述约束内**作一次离线回拟合，避免重开刚删的广播服务。直接运行第二残差项只是 TASD 借入，额外支持/权重/合并必须重计，本次最小试验不加该旁路。真正待验证的增量是“整数卷积约束＋实际执行并集＋消费者误差”的联合选择；约束和代价优化各自都不是新数学。

**不可省的未实现强控制：** unconstrained U-zero 可直接保存未除2的重建 p2，以 output_scale/2 重定 a_q40，界够则仍用 p32、末端单次 I24 RNE；须独立导出/AEE/相位误差，不能因奇数或非平移共享就排除。moment 的增量只能说保留整数三tap/平移共享表示；之后有优势也须对这条直接 U 稀疏成熟 A 才能归因。

**完整 A 与最小试验：** 冻结0–31校准、只取一次10% Q2实际服务削减目标；做无剪枝、整`(r,og)`三tap块剪枝（完整普通A）、相同合法候选/相同回拟合但按幅值排序、上述实际成本排序四臂。后两者固定相同静态组预算，以隔离成本选择；每个新 g 分别跑同函数ordinary/Winograd，held128–191、4000–4063与valid825独立评估。须测：各组D活动/跨样本迁移、N8并集、真实MAC/cache/W/Z/p访问、恢复/配置/BP/I24总服务、signed界和奇数数、raw/wide/I24偏差及AEE。当前VLOAD即使U零也仍付一拍/一次cache写，Q1与D变换也仍全做；不虚计这些为省下的周期。

**原放置评分校准：** 定向检索 Winograd+spiking/binary/factor/second-stage，定位到 [mlGeNN 原论文](https://www.researchgate.net/publication/359035109_mlGeNN_accelerating_SNN_inference_using_GPU-enabled_neural_networks) 对稀疏脉冲无乘法传播与ANN Winograd/FFT的成本比较；已核段落没有“binary竖因子保AAC、仅连续横因子快卷积”。[Qin的3D低秩Winograd](https://arxiv.org/abs/2301.11180) 也不是该空间接口，前述DSC/SKC摘要不能替代它。本次未定位直接覆盖该精确放置的primary证据，因此原“组合2/10”只评借入算法、不能解释为接口已被完全覆盖；接口3/10仍是保守暂评，既不据此杀掉实测，也不以未搜到宣称首创。
