# 体系结构 ANN/DNN → 本网按需迁移（2026-09-16）

约束：光流 SNN-Transformer，输出二值 {0,θ}，准入门同端口净服务 ≥15% + AEE 优于 NB0。
方法：完整抄 A（有代码用代码，没有就写周期模型/RTL），好再改 X；差就当问题发现。
不限 SNN。下面每条都是**可执行挂点**，不是「和 C1 同一句话换皮」。

状态：`copy` = 本目录正在抄；`queue` = 下一刀；`blocked` = 缺全文/缺接口。

## 必须先抄全（本目录）

| ID | 论文 | Venue | ANN 机制 | 挂到本网的 A | 抄完才允许的 X | 状态 |
|---|---|---|---|---|---|---|
| A1 | BitFair | JETCAS'26 | 逐 PE 独立早停 + 学习阈值 + 自适应位序，预测 ReLU=0 | 10 路 t **独立**停；Y 符号幅值 bit-serial；theta=thr（BN 初值）；贪心 ABO | ① 把 ReLU-0 预测换成精确区间但仍逐 t 退休（不要等最慢）② sd5ai 上学 theta ③ 生存正则对着供数 | **copy** |
| A2 | BitL | MICRO'25 | 子 tile 横/纵查找 + dynamic pivot，缩短零比特关键路径 | 10×24 Y 比特阵，8-bit 子 tile，一行或一列一拍 LUT | 证书锁定后不再查剩余 tile；和 BitFair 逐 t 停叠 | **copy** |

Claude T10 的 5+5 LUT **不是** A2。Codex FUSION_STATUS 第 47/W02 已写明。

## 降维打击候选（ANN 加速器机制 → 更窄的本网接口）

本网比通用 DNN 简单的地方：消费者是二值门、Y 10 维、S 已经是 0/1、A 是 10×10。
把别人在「大 GEMM / 大卷积 / 长序列 attention」上成立的机制，压到这个小接口上，数字要么立刻很大、要么立刻被粒度税杀死——这就是降维。

| ID | 论文 | Venue | 抄什么 | 为什么可能打得过 C1 | 杀门 |
|---|---|---|---|---|---|
| A3 | **BitWave / SparseCol** | HPCA'24 / 后续 NPU | **bit-column-serial**：一列权重比特横切所有通道，跳结构化零列 | 生产边 W 的 bit 列稀疏；S 已是 1-bit，列全零直接免 MAC。T14 证书界松，但**跳零列不是证书** | 相对「基线也跳 S=0」的公平口径额外省 <15% 则杀 |
| A4 | **AdaS-Pro** | TCAD'25 | 运行时选 **更稀的那个操作数** 做 bit-serial + Booth | 门核 A 和 Y 谁更稀就串谁；C1 锁死串 Y | 两操作数都密或切换元数据 ≥1 拍/组则杀 |
| A5 | **PADE** | 2025 加速器（attention bit 级早停+比特复用） | QK 分数 MSB→LSB，top-k 范围一确定就停；比特复用免预测器 | **注意力边**，C1 根本没碰。本网有二值 Q/K | 同端口相对 dense QK <15%，或必须预测器税 |
| A6 | **MCBP / BGPP** | MICRO'25 | bit-slice 稀疏+重复；MSB→LSB 估 attention，出 top-k 范围即停 KV | 同 A5，挂 KV/分数边 | 光流窗远小于 LLM 序列则收益不够 |
| A7 | **CGNet** | MICRO'19 | 通道前缀先算，部分和决定要不要算尾巴；跳尾仍保留非零部分和 | 生产边 C=96/768 通道前缀；门不需要完整 Y 时停产 | 前缀后仍要完整值消费者（fc2 连续边）则不能丢尾巴 |
| A8 | **SnaPEA** | ISCA'18 | 值级部分和预测负则跳剩余权 | BitFair 的值级祖先；先当 BitFair 消融 | 被 BitFair 支配则只留对照 |
| A9 | **LUT-DLA** | HPCA'25 | 把 MAC 收成 LUT 数据通路 | 门核 A 10×10 本来就可整表；比 T10 半模式 LUT 更极端 | 面积 vs FX 10 个乘法器；FPGA LUT 爆炸则杀（T24 已见） |
| A10 | **SeaCache** | MICRO'25 | 变长 fiber 合装 + 重用感知替换 | Gustav/FC1 源 fiber；生产边缓存 | 要接真实 bank，不是门核拍比 |
| A11 | **UCNN** | ISCA'18 | 输入相似复用 | 相邻窗/相邻 h 的 Y 或 S | T2 已杀运动对齐；换 **通道邻域 / 同 patch 多 t** 再测一次 |
| A12 | **DeltaCNN / MotionDeltaCNN** | 训练侧差分 | 帧差只算变化 | 事件源已是差分；挂**深层特征**不是 g | 深层 toggle 相对降 <50% 则停 |
| A13 | **Single-Spike ANN** | ISCA'25 | 权当延迟、激活当到达时间 | 二值 {0,θ} 几乎已经是单脉冲；把 θ 编码成 delay | 光流 AEE 崩则杀 |
| A14 | **DAC'25 bit-sparse MAC FPGA** | DAC'25 | LUT 友好编码 + 乘法时间不定的 PE | 直接对着 KV260 门核，补 T24 的 LUT/DSP 反差 | 不定延迟在组同步下被最慢 PE 吃掉则杀（又是粒度） |
| A15 | **R-Sparse** | ICLR'25 | 幅值分流精/粗两路 | 原 C2；lane 粒度再抄一次生产边，不抄块级 | keep≥50% 才能过 AEE 则空间不够 |
| A16 | **ScanNow** | ICCAD'25 | 窗口调度 | 被付费墙挡 | blocked，要全文 |
| A17 | **Phi / Prosperity / Bishop** | ISCA/HPCA'25 | 模式复用、前缀、TTB、ECP | Codex 已部分碰；**完整 A**（ECP 界、packer、恢复训练）仍未迁到本网门+注意力 | 完整抄一次 ECP 到二值 Q/K，不要只写叙事 |

## 三条应并行的主线（不要再只养门核）

1. **门核 A1+A2**：先出 BitFair 逐 t 停 vs C1 组锁的拍数/翻转；再叠 BitL pivot。这是把 C1 从「串行+区间」拆开，看收益到底来自哪一块。
2. **生产边 A3+A7**：FC1 的 Y 还没生产出来。跳 W 的零 bit 列 / 通道前缀停产。这才碰完整链分母。
3. **注意力边 A5+A17**：QK 二值，Bishop ECP / PADE bit 早停是 ANN→SNN 最干净的迁移。和门核正交。

训练（sd5ai）只跟 A1 的可学习 theta 走，不把所有线绑死在微调 A 稀疏上。
