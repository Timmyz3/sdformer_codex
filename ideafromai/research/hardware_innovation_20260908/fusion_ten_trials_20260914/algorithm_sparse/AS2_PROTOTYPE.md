本项完成了“完成latent K4原型+单rank残差”的实际表示接口，编码、weighted距离、最近原型选择、最大加权残差选择、原型Q2表读、残差乘加均在RTL。8块暖拍比完整R8少7.69%；diverse10为1.471894678700，略差同集合NB0门1.469968114434。按预先约定的≥10%净周期拓展条件，本工作点不再跑825，也不靠换K/残差数/重选格式续命。它比zero+单rank控制质量好，但并未因此胜过完整R8。

A是[LUT-DLA](https://arxiv.org/html/2501.10658v1)的在线距离+查表与[Phi](https://arxiv.org/html/2505.10909v1)的pattern/weight product加双向残差。B是这里完整Q1照付，Q2只有8rank，encoder成本很容易吃掉算子削减。本次仅测试从二值模式迁至完成signed latent，并只保留一个最大加权残差的新有损工作点；这种迁移与限定残差本身没有建立足够X。

code0固定全零，另3原型用训练窗口weighted L1最远初始化与8次固定kmedian获得，不使用验证或GT。运行argmin Σ_r 2^shift_r |z_r−c_r|，低index解tie；残差rank用argmax同weighted绝对值。输出p=Q2*c+Q2[:,r]*(z_r−c_r)，其余七个残差删除。z_hat各坐标仍来自真z或训练原型，signed13可存；RTL按早先抽象界给残差保守留signed14，共同乘法器为16×14；固定Q1精确L1界最大667，实际13bit已经足够。完整整数gold同时核对“重建z_hat后dot”和“prototype结果+残差”相等。

非零vector encoder29拍，全零同读检测后2拍；本8块encoder实际6229拍；共660个256bit原型结果字读、2484次Q2 MAC。zero-prototype+单rank控制非零vector5拍/全零2拍，encoder实际1261拍、无非零原型读、同2484 MAC，但AEE1.714547411881。原型结果表1536B，codebook52B及元数据另计。所有参数冷装入也计；没有把encoder放TB或把表当免费组合ROM。codebook小寄存器有独立8lane读口，公平资源中明确列出。

| 执行模式 | 暖计算总拍 | 冷配置+计算 | 比exact暖拍减少 | diverse10 AEE | valid825 AEE |
|---|---:|---:|---:|---:|---|
| 0 exact | 107044 | 134644 | 0.000% | 1.390331540132 | 1.327635022608（同环境封存值，未重跑） |
| 3 K4prototype_residual | 98813 | 126413 | 7.689% | 1.471894678700 | 未运行 |
| 4 zeroproto_rank | 93185 | 120785 | 12.947% | 1.714547411881 | 未运行 |

共同合同：固定Q1[8,864]、Q2[96,8]、θ=1和aQ40/bQ20；完整Q1、T10、C96、N96、P4都在RTL。入口是实际r0.sn2脉冲与真实FP32 identity，出口完整I24。所有模式每tile480个8lane消费者字，包含FP32转Q20、乘加、RNE/saturation和背压。九执行模式共用全部编码状态、端口与8路加法器/8路signed16×14乘法器；固定核中的完整R8是mode0。参见[资源合同](RESOURCE_CONTRACT.md)、[执行账](EXECUTION_LEDGER.md)和[原始收据](results.json)。396次运行，三个检查点raw/J/I24各1,520,640值零差；独立源公式11088项通过。

训练校准使用实际train列表的thun_00_a_0002，336规则栅格窗口，输入183322个spikes；不属于valid825。硬件评价是另一帧zurich_city_09_a_0001的8个固定内部/边界块，没有活动筛选。每块连续4×4源、2×2输出；不是整层。校准参数在AEE之前冻结，不训练模型、不扫描比例。完整I24消费者同环境diverse10 baseline精确复现1.3903315401317662；同集合NB0十帧1.4699681144337489，官方825门1.447936665574317。十帧过门不等同825过门；共同全零encoder旁路前的数只存在pre_zero_bypass快照，最终396runs没有重复计数。旧历史十帧门1.45460286107也单独保留。AEE是此整数线性与I24函数接后续浮点网络，不能称全网bittrue。

表中暖拍从go到core+consumer全部退休；冷拍另加每tile3450拍完整source/参数配置，共8tile27600拍。跨其他代理的源快照、端口或整层结果不直接拼周期。背压/二次命令数据见SUMMARY.json。

实现与重现：[reference.py](reference.py)、[lossy_r8.sv](lossy_r8.sv)、[run.py](run.py)、[verify.py](verify.py)。完整primary来源边界见[SOURCES.md](SOURCES.md)。
