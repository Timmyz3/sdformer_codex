# 七篇原始来源的有界核验

核验日期：2026-09-13。先完成独立五想法，再阅读指定 AI 例子，最后针对候选查原始来源。下表的“迁移判断”是本项目的推断，不是原论文替本项目证明的结果。没有重新全扫已有775实体，也没有据检索未见就宣称新颖性。

| 原始工作与本次核对版本 | 实际核对位置／已找到证据 | 对本次候选的约束或借入 |
|---|---|---|
| Jacob 等，*Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*，arXiv v1，2017-12-15。[原文](https://arxiv.org/html/1712.05877) | §2.2–2.4，式(4)、(7)–(10)：仿射整数映射、尺度外提、零点相关行/列和、融合偏置与量化边界。 | **C1 的 affine/常数项外提属于强先验 A。** 必须给普通 affine 同样权限；不能把“不先反量化再乘”申领成新机制。原文也强调训练/推理量化位置一致。 |
| Li 等，*LUT-DLA: Lookup Table as Efficient Extreme Low-Bit Deep Learning Accelerator*，v1，2025-01-18。[原文](https://arxiv.org/html/2501.10658v1) | §III–IV，CCM、IMM、连接FIFO、LS数据流与Algorithm 1；§V LUTBoost。 | C1 的查表对照必须包括索引/匹配、查表响应与累加；已有门码可免额外聚类索引计算，但门本身不能免费提前就绪。未声称复现整套 LUT-DLA。 |
| Huang、Yuan、Shao、Zhang，*MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators*，MLSys 2025；arXiv v2，2025-04-07。[会议原文](https://proceedings.mlsys.org/paper_files/paper/2025/file/9032e5c9ec394ce768a2fa9bdc56af6c-Paper-Conference.pdf)，[可检索原文](https://arxiv.org/html/2504.02658v2) | §3.2.1–3.2.6、Algorithm 1：量化与低秩残差交替更新、rank选择与补偿器量化；§3.3：INT3打包、实际位操作解码、重排与流水。 | **C2借执行格式及付费解码，C3借量化加低秩补偿。** 此处不是MoE，不照搬专家频次。它的CUDA加速数字不能搬到单issue定点机，且“低秩补偿量化误差”已非新概念。 |
| Wei 等，*Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency Spiking Neural Networks*，ISCA 2025；v1，2025-05-16。[原文](https://arxiv.org/html/2505.10909v1) | §3.1–3.2：Level1模式与权重预计算、Level2的{+1,−1}稀疏修正、分块归约；当模板不合算时保留原行稀疏。 | 约束独立 I02/I05：模板+差分不是新贡献。C1全D解码也不能假借另一组门卷积的预计算；PED U是独立权重，实际`Ug`必须计费。 |
| Sun、Que、Loncar、Luk、Spiropulu，*da4ml: Distributed Arithmetic for Real-time Neural Networks on FPGAs*，TRETS 19(1), Article13，2026；arXiv v2，2026-04-24。[正式DOI](https://doi.org/10.1145/3777387)，[原文](https://arxiv.org/html/2507.04535v2) | §4.1–4.3，量化区间、CSD、图分解后对两矩阵做CSE、输入/输出二次幂归一化。arXiv记录明确2026 TRETS信息。 | 普通与lifting共享CSE/PoT/尺度消除权限；C1 fixed零基值的`scale→RNE`应精确合并。原作全展开FPGA CMVM不等于本机有限端口槽数。 |
| Fang 等，*Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies*，NeurIPS 2023；v4，2024-01-09。[原文](https://arxiv.org/html/2304.12760) | §3.2–3.4，式(9)、(11)–(15)：全连接时间映射；masked和sliding变体为处理未来输入依赖而另定义。 | 固定T10 PSN的某个门不能默认只依赖过去。C1全D与门编码必须尊重已完成T10词；本次没有换成masked PSN，也不改变AT-LIF的固定{0,θ}身份。 |
| Parger 等，*DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos*，CVPR2022；v2，2023-09-02。[原文](https://arxiv.org/html/2203.03996) | §3.1：线性卷积差分恒等式；非线性层需要历史累计输入，输出差分为`f(x+δx)−f(x)`；§2.3指出收集/散射开销。 | 限制I03及舍入复用叙事：差分线性展开与缓存非线性状态已有先例。C1跨过sat24之前必须证明无饱和或补偿，不能把线性恒等式越过RNE/sat直接用。 |

检索使用的主要查询是 `MiLo mixed precision low rank quantization accelerator paper`、`LUT-DLA 2501.10658`、`da4ml constant matrix`、`Phi 2505.10909`，随后直接打开上表 arXiv/会议全文并定位章节。`ReverB low rank binary residual quantization paper` 的这一小轮结果未定位到所需原始论文，因此本轮不以未确认的ReverB机制支撑候选，也不宣称该工作不存在；C3明确只以已核实MiLo为A。

证据状态：C1的仿射代数和非线性边界为 **support-located**，其标题新颖性为 **challenge-located**；C2/C3有清楚最近邻，但本项目性能/精度仍为 **not-validated**。最直接的新接口是已有Q8表示到真实连续U消费者；是否值得当论文主线，仍取决于超越这些共同优化权限的实际证据。
