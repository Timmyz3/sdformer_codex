# CICC/JSSC/ISSCC：围绕当前三项真实探针的有界补读

2026-09-12。**一个目录遗漏已经找到；本轮只深入三项。** 原来的 [CICC_SUPPLEMENT.md](../../../audit_comparison_20260909/CICC_SUPPLEMENT.md) 已列 16 条 CICC，新汇总表未吸收该文件，才显示 CICC 只有两条。这是汇总缺口，不是本轮新发现了 14 篇，更不是 16 篇已全文读完。已通知主线程补入口，未改别人的表。

这里的三个原作家族对应：动态神经元剪枝；时间并行的非零供数；源／操作数留在 RF。阅读结果为 **一篇 ISSCC 全文及图 30.6.1–7，两项 CICC 仍有原文缺口，其中 Fang 增补了 JSSC 作者机构摘要**。尚未找到官方 RTL，不能称开源芯片复现。没有新 EDA、训练或性能实验。

| 原作家族 | 本轮确实读到 | 对当前推进的处置 |
|---|---|---|
| Yang 等，样本自适应动态神经元剪枝，CICC 2025 11-5 | 官方节目、作者机构论文页；无方法正文 | 补为共同门字预测／删生产者的近邻；不能据题名复制细节，也不能缺全文就杀思想 |
| Fang 等，非结构稀疏注意力／卷积 3D 阵列，CICC 2024；JSSC 60(3)，2025 | CICC 及 JSSC 一手机构摘要、作者伴随演示摘要；无电路全文 | 作为 Gustav 外的强供数对照；须补并行取数及实际 bank／T 资源 |
| Vecim，ISSCC 2024 30.6 | 原刊全文及完整三页图表；见下方来源 | 立即完善普通驻 RF MAC／队列基线，不把驻留或预取本身冒称 X |

## 1. Yang：样本自适应动态神经元剪枝

**身份。** *A 40nm 0.05–1.4uJ/inference Sample-Wise-Adaptive Spiking Neural Network Processor with Dynamic Neuron-Pruning and Unstructured-Model-Aware Architecture*，Jinqiao Yang 等，CICC 2025。原作题名及作者由 [CICC 最终节目](https://www.ieee-cicc.org/wp-content/uploads/2025/04/CICC-2025-Program-4-8-25.pdf)和[复旦作者机构页](https://fit.fudan.edu.cn/Data/View/6272)核实。

**A 已知与缺口。** 可确认它把样本自适应、动态神经元剪枝和非结构模型映射结合。未取得正文，无法确认触发特征、阈值计算、训练、稀疏格式、是否预测/回退、临时状态或每级收益。仅据题名编造一个比较器，再称“完整照搬原作”不成立。其 0.05–1.4uJ 是原作题名，不是我们的预测能量。

**B 是当前哪里。** 本地早停／组完成曾只有约 1% 的服务增量；更深原因是 gate 完成后真实 I24/PED 仍有消费者，不能将神经元不发放当成连续生产者可删除。AT-LIF 的 θ 可折权不改变这一点。非因果 T10 又使“当前 t 不发放”不能自动推出未来时间行无用。

**能试的一个接口（本地假说，不归给原文）。** 在已有 phase／广播组剪枝试验上，仅增加一个 **T10 全门字＋PED 残差预算共同判据**：使用已经算出的低成本 preview 特征决定一个物理源组是否进入精算；两个消费者共同使用同一组选择。训练损失同时比较真实门字和真实连续 PED，选择器、未选组替代值及供数目录全部算成本。允许有损，必须与相同恢复预算的普通组剪枝比较。不要再用“gate 零”删 PED，也不要截断原非因果 PSN。

**借全清单。** 若要把它写成基于 Yang 的具体硬件改进，先补出原作预测器／剪枝策略、非结构格式与调度、代价与精度恢复；本地独立探针可以先做，并明确它尚非原作完整迁移。不是以全文缺口暂停一切尝试。

## 2. Fang：3D 阵列＋并行非零供数

**身份与实际证据。** CICC 2024 为 *A 0.078 pJ/SOP Unstructured Sparsity-Aware Spiking Attention/Convolution Processor with 3D Compute Array*，DOI `10.1109/CICC60959.2024.10529019`。[HKUST 原作摘要](https://repository.hkust.edu.hk/ir/Record/1783.1-138276)明确问题是跨时间的 W/psum 重复访问、逐个非零取数吞吐不足、算子统一排程不合适。[作者伴随演示](https://epapers2.org/biocas2024/ESR/paper_details.php?paper_id=2354)确认三项方案：3D 阵列、并行非零取数器、多模式调度。

相关 JSSC 长文是 *An Energy-Efficient Unstructured Sparsity-Aware Deep SNN Accelerator With 3-D Computation Array*，**2025 年 3 月，60(3):977–989**；DOI 中 2024 是先行出版身份，不宜将刊期写成 2024。[HKUST 长文摘要](https://repository.hkust.edu.hk/ir/Record/1783.1-152063)确认多 T 并行与 SCONV、Q/K/V、SSA 模式。IEEE 页面遇到访问验证，机构页仅摘要；没有取得其 bank 仲裁／索引格式／累加器位宽正文。

**对本网络的边界。** 二电平 θg 和折权后 AAC 很适配该数据供数方向；T=2/T=10 应重新分配状态，不等同照抄时间维展开。原算术以脉冲驱动为核心，不能把真实连续 PED_I24 放进同一 AAC 而维持精确。它的 SSA 也不能当 Motion-XOR。原论文没有从当前摘要证明解决了非因果 T10 的 gate/PED 双消费者释放。

**一个可以接着写的小接口。** 从已有全 K864 原始门字捕获生成按实际 H8/CR256 布局分组的非零请求，做 **并行取数完成位图＋有界两 P2 源驻留**。与普通规则 3×3 地址生成和 Gustav NRV 都用同 SRAM 口、同 outstanding 数、同 T10 状态。输出不提前删：gate/连续支路分别登记完成后才回收。先测被串行目录／请求组织浪费的槽；不将多开取数口或多开 T lane 的收益算作 X。

**借全清单。** 3D 数据映射、原非零索引、并行取数与权重布局、SCONV/SSA 各模式、buffer/psum 容量、bank 冲突、神经元完成协议，缺哪项就标哪项。该建议加强供数分母，不恢复旧 TSBG 换序为标题。

## 3. Vecim：直接约束本轮“源驻 RF”的新颖性

**全文。** *Vecim: A 289.13GOPS/W RISC-V Vector Co-Processor with Compute-in-Memory Vector Register File for Efficient High-Performance Computing*，Yipeng Wang 等，ISSCC 2024，30.6，pp.492–494，DOI `10.1109/ISSCC49657.2024.10454387`。[ISSCC 官方节目](https://www.isscc.org/s/ISSCC2024AdvanceProgram-Final.pdf)核身份；本轮读了[原刊 Session 30 PDF 镜像](https://iccircle.com/static/upload/img20240529102305.pdf)的正文及图 30.6.1–7。PDF 暂存 `/tmp/ped_followup_isscc2024_session30.pdf`，不把全篇复制进 Git。

**A 完整到哪些关键段。** Vecim 基于 Ara 的向量通路，将 INT8/BF16/FP16 乘加放在全数字 1R1W VRF 附近／内部。它不仅“缓存 A”：分离 MEM、CIM、ARITH 三队列，按实际依赖和读写可用相位发射；累加目的操作数在真正需要前写回即可，不能把整条 MAC 的所有阶段一律当 RAW 停顿。块 MVM 指令把 A 装成向量并在多个输出上复用，另外支持 8b×8b→32b。其定制 SRAM 计算、时钟与宏面积不能转借给 TS1N28；本地可借的是控制与供数对照。

**与本次结果的关系。** [PED 探针](../ped_bitplanes/README.md)的普通驻 RF MAC 减少 26.84% ready 服务，是原有一源反复取数被修正；Vecim 说明源复用、块指令和依赖感知发射都有成熟先验。它不支持把这个数字单独作为新机制。我们保留完整 I24×INT16、48bit 累加和原 RNE/sat，不能套 Vecim INT8 的周期或能量。

**一个可执行接口。** 将现有 `resident_mac()` 接入双方完整 U32→V96 连续链，然后给双方增加 **固定两个源槽的 load/MAC 重叠＋精确 RAW 可见时刻**。槽内保留当前 k 的 T10；下一 k 只在共享 SR64 许可时预取，所有输出 H8 消费完成才覆盖。相同 RF96 和 CR256，不能给候选额外 20 源寄存器。当前已用 `0..79` 存 P2×T10×H32 psum、`88..91` 存源，因此第二槽能否真正驻留必须先排一个具体预算；放不下就用分组或回退，不能口头双缓冲。

**待证的 X。** 普通 Vecim 式控制做到后，如果双消费者所需表示／完成边界仍让一组源产生实质重复状态或等待，才研究该差分。现在的具体任务是把普通底座做强；非因果 PSN 不能像逐 t LIF 那样提前判未来完成，PED 也不会因 g=0 自动失效。

## 本轮改变执行顺序的结论

优先把普通驻 RF MAC 接双方完整 U/V 并恢复原 RNE/bias，再研究生产者组共同剪枝；Fang 的并行取数补作供数强对照。三项都未被宣判思想失败。没有从有界检索未找到同接口，推导“首个”或“强接收”。SparseTrim 等旧补充仍保留在目录里，本次不再产生更多卡。
