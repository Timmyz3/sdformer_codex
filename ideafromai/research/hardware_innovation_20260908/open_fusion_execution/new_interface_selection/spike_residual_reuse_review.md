# 脉冲解码 + 连续残差共同服务 PED：有界可行审查

2026-09-12。**允许留一个真实残差探针；目前不能宣称复用了现有 gate 卷积，不能据分解代数启动新标题。** 本次只核真实图与已有端点，未用 GPU、未写新执行器。

**A / B / X。** A 是固定预测值加精确残差、线性投影分解，以及既有稀疏供数/低秩 PED；这些都不是 X。B 是已经计算过的 T10 门字是否携带足够的连续源信息，能让 PED 的必要连续源请求变少。尚可检验的 X 只在**同一门字的解码残差表示，确实替换连续源请求，并在完整双消费者、原舍入及相同端口下留下净服务**；当前没有该证据。普通压位宽、双路径或已有卷积代数展开不足以构成增量。

**真实图先排除三个误认。** [SpikingPEDLayer](../../../../../SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py) 的 `conv_res` 是独立 1×1/stride2 连续投影，`sn→conv` 是另一组 3×3 权重；本学生把前者改为 U96→R→V96，后者并未因此成为同权。时间 D 与通道 U 可交换，不等于 U=D 或 U 与 gate 权重相同；需核系数、源、目的、空间偏移和数值尺度后才能复用乘积。其次，`full_sn1_words` 属于更新前 `full_I24`；PED 消费的是经过 Conv2/BN2/残差及锚点舍入的 `full_updated_I24`。用前者预测后者时，r 包含新残差支路，不能叫纯神经元量化误差；`full_proj_words` 是另一个下游门。最后，T10 PSN 非因果，不能把某个 t 的门当作提前完成整个解码的许可。

**精确算术边界。** [fixed_temporal_coordinates.py](../../algorithm/patch_probe/residual_consumer_probe/projection_chain/fixed_temporal_coordinates.py) 的 PED 实际是
`Q24(Vq·Q24(Uq·x/2^eu)/2^ev)+原 bias/sat`。
一般不等于两条独立 U/RNE/V/RNE 支路之和。合法精确候选是先定义整数 `b=decode(g)`、`r=x−b`，把 `Uq·b+Uq·r` **在原 U RNE 前合并**，随后只执行一次原 V/RNE/bias；中途不得各自截断。若 b 是 `RNE(Dq·g)`，把 D 换到 U 后面会跨越该 RNE，必须另证，不能免费交换。残差最坏需要25位；即使 x/b 都在24位内也不能截 r。要计解码、偏置、差分生成、额外累加/状态及端口；`As_inverse` 不会恢复阈值化丢失的信息。幅值变小而 r 仍非零，在当前24位 MAC上没有天然发射数收益。

| 已有工作/端点 | 已覆盖什么；此次还能留下什么 |
|---|---|
| [三源投影及共享潜空间审阅](../../algorithm/patch_probe/residual_consumer_probe/projection_chain_prior_review.md)、[残差消费者审阅](../../algorithm/patch_probe/residual_consumer_probe/novelty_review.md)；CFMP/线性块折叠 | 已有 `C(xstem+Y0+Y1)` 沿生产链展开及共同/私有潜空间提案，需新投影核、动态 stem BN、原门链和量化边界。若本方案退化为这条展开，停止复写；但这些记录未完成本次“门字解码后精确 r 是否便宜”的探针。 |
| [shared-Q+inverse](../../algorithm/patch_probe/residual_consumer_probe/projection_chain/README.md) | 已说明共享坐标仍需 BZ、逆变换、常数及额外 RNE，不能以共用 Q 宣称省连续线。它不直接证伪整数门字残差编码。 |
| [私有 R56 实际请求](../../algorithm/patch_probe/factor_completion_20260909/latent_stage_train16/physical_u_requests.md)、[连续位面探针](../ped_bitplanes/README.md) | 前者逻辑取消被物理字合并吞掉；后者串行位面输给驻源 MAC。故必须算真实源字与执行器，不能用残差位宽/加法数冒充周期；未试的廉价低位 PE 属另一资源点。 |
| [SymbolicLight V1，2026预印本，§3.1–3.3](https://arxiv.org/html/2605.21333v1) | 已有二值门与连续表示双路径；连续 Q/K/V 另投影，非本网精确残差解码或双分支同权。它占据“双路径切法”先验，不是本接口已闭硅证据。 |

**已转为一个固定真实探针。** 主线程要求实际执行，代码在 [spike_residual_probe/probe.py](spike_residual_probe/probe.py)。输入固定为同生产者 **full_updated_I24 + full_proj_words**；sn1/sn2 不代替该门。两个学生分别使用校准锚点 y/x=32:96:4 拟合一个全通道共用10×10仿射 D，一次将系数RNE为整数状态单位；对照为 raw I24 和同校准逐时间常数偏置残差。只用原 corner/interior 留出，完整原 U32/V96/RNE/bias 不变、rank不扫。原 U RNE 前合并整数二值项与精确 r；零率、负符号位、最大位宽、字节和两种独立二值投影成本均计入。Dg 与现有 gate 卷积没有同权抵扣。另检查本帧所有 stride2 锚点的 b+r 重构和残差范围，不能将该统计称为完整层周期重放。结果见同目录 results.json/README.md；若 r 基本密集且完整费用不降，只停这个固定编码，不否定未执行的低位 PE/不同表示接口。
