# T12 七篇标杆精读：方法论萃取 + 非SNN→SNN移植扫描 + 硬件idea（2026-09-15）

> 任务（用户指令）：精读 Bishop、Phi、FireFly 系列、GustavSNN、COMPASS、C-Transformer、
> Prosperity——重点学他们**怎么发现问题、怎么解决问题**；看重点引用文献；找
> **非 SNN 领域已有技术沿用到 SNN** 的机会（GustavSNN = Gustavson ANN→SNN 适配的范例）；
> 根据当前工作（C1 存活形态）产出硬件 idea。
>
> 诚实边界：COMPASS（W0056，"SRAM-Based CIM SNN Accelerator with Adaptive Spike
> Speculation"）与 C-Transformer（ISSCC24）**无全文**，仅有题名/二手摘要，本文只能给
> 题名级判读并维持 APPLICABLE_BLOCKED；其余五篇（GustavSNN/Bishop/FireFly-S/
> FireFly-T/Phi）本地有全文，Prosperity 有完整笔记+一页卡。

## 1. 七篇的问题发现→解决 recipe（学的是什么）

### 1.1 GustavSNN（HPCA26，全文已读）

**怎么发现问题**：三步——(a) profile 确认 SpMM 是 SNN 能量瓶颈；
(b) 指出前人（PTB/Stellar/LoAS）虽然"用了 Gustavson product"，但都是 OP-based
数据流，潜在状态驻留导致重复读 spike/权，**"引用了机制≠继承了机制的收益"**；
(c) **先建解析能量模型再写 RTL**：IP/OP/ex-situ GP/in-situ GP 四数据流的
Dr/Dw 读写计数对比 + Nsilent ∝ 1−ρ^T 解析式（FTP 类收益随 T 衰减的定量论证）。

**怎么解决**：CPTB（列并行 tick 批数据流，每 PE 持 P=N/K 个电位在本地寄存器
in-situ 累加）+ NRV 非零行向量格式（行跳过）。结果 11.8× vs naive GP、
1.43× vs SOTA（GOPS/W）。

**可学的纪律**：①"前人用了 A 但没得到 A 的收益"是发现问题的独立入口；
②解析模型先行，RTL 只验证胜者；③把"收益随某参数衰减"写成闭式（Nsilent∝1−ρ^T）。

### 1.2 Bishop（全文已读，NUL 字节文件用 grep -a）

**怎么发现问题**：FLOP profiling——attention 占 66.5–91.0% 总负载；
再列四个具名挑战（多 bit 权重复用/训练感知稀疏/负载异构/乘法开销）。

**怎么解决问题**（三招各有独立价值）：
- **TTB 打包**（token×时间 bundle）复用多 bit 权重；
- **BSA 训练**（bundle 级 L0）让硬件友好的稀疏"长出来"；
- **ECP（Error-Constrained Pruning）——最值得学的一招**：利用尖峰 Q/K 的
  **二值性**，若活跃 bundle 数 n_ab < θ_p,Q，则该行分数**以 100% 置信度**
  低于阈值、可精确剪除——**ANN 连续 Q/K 下此界不存在**。这是"用 SNN 独有
  结构性质换精确界"的范式。5.91×（vs PTB）/6.11× 能量。

### 1.3 FireFly-S（2408.15578，全文已读）

**怎么发现问题**：前人只挖了 spike 稀疏这一侧，**权侧稀疏被忽略**（双侧重叠区）。
**怎么解决**：>85% 权重剪枝（梯度重布线）+ 4-bit LSQ，解码用
Bitmap 按位 AND + one-hot 剥离（y = x ∧ ¬(x−1)）计数替代乘加。
**纪律**：把"operand 的另一侧"当作独立攻击面。

### 1.4 FireFly-T（2505.12771，全文已读）

**怎么发现问题**：点名前人三个具体缺陷（而非泛泛"还有提升空间"）——引擎耦合、
bank 冲突、负载不均。**怎么解决**：双引擎 overlay（稀疏引擎管 conv/linear +
二值引擎管 AND-PopCount attention）、LUT6 优化 popcount、字节级 SRAM 写、
乱序执行消除 bank 冲突。1.39×/2.40× 能效、4.21×/7.10× DSP 提升。

### 1.5 Phi（一页卡+全文笔记）

**怎么解决问题**：表示侧离线预计算——权矩阵行重复 → k-means 码本（q=128 个
16-bit 二值模式），**模式×权重的乘积离线算好**（Pattern-Weight Products），
运行时只剩查表 + L2 逐元素 {1,−1} 校正。
**纪律**：静态数据（权重）上能离线做的绝不运行时做。

### 1.6 Prosperity（完整笔记 + 一页卡）

**怎么解决问题**：乘积稀疏——不同 spike 行的支持集呈子集关系时，
已算好的内积可**精确复用**（Partial/Exact Match）；Intersection 情形明确弃用。
Codex 侧 G4 融合试验为负结果，家族已实测覆盖。

### 1.7 COMPASS / C-Transformer（无全文，题名级）

- COMPASS：题名含 "Adaptive Spike Speculation"——**推测+恢复**家族。
  注意与 T7 杀的"静态锁深表"不同：推测带恢复路径则零差可保，误推测只付
  重放代价而非精度。这是**唯一未被 T7 结果覆盖的同族变体**，但无正文无法
  继承，维持 BLOCKED（T11 第一优先清单之外，建议加入索取列表）。
- C-Transformer（ISSCC24）：SNN Transformer 芯片平台，必引对照，域失配维持。

## 2. 重点引用文献覆盖核查

七篇的核心引文——LoAS、Stellar、PTB、Spikformer、Spike-driven Transformer(SDT)、
Cambricon-X、Eyeriss、SyncNN、SpikeThrift、ExSpike——**全部已在 698 篇判读集内**
且均已分桶（INCORPORATED/FAMILY_COVERED/NOT_APPLICABLE），无漏网新对象。
这与 T11 结论（APPLICABLE_NEW=0）交叉一致：标杆论文的邻域就是我们已经
系统筛过的邻域。

## 3. 非SNN→SNN 移植扫描（GustavSNN 模式的推广）

GustavSNN 的 recipe = 找一个**非 SNN 领域成熟机制**，分析它**为何直接适配不了
SNN**（本例：N×T 稠密 spike 矩阵 + 电位时间累加使 ANN 的 GP 数据流失配），
**重新推导数据流使其适配**。对当前 C1 语境逐一扫描候选源领域：

| 源领域机制 | 移植到 C1 语境 | 判定 |
|---|---|---|
| Gustavson SpMM（ANN/HPC） | 稀疏供数家族 | 已覆盖（Codex TSBG/GustavSNN 本体） |
| 迭代精化/MSB-first（数值计算） | 位平面渐进供数 | **就是 C1**，已移植完成 |
| 区间算术（verified numerics） | 逐判决区间证书 | **就是 C1**；T10 RTL 已含符号分裂（P/N 分离区间，比朴素区间紧） |
| 仿射算术（区间算术的精化） | 用相关性项收紧证书界 | **解析杀死，无需实验**：未知 bit 是真正独立未知量，无线性相关结构可利用；符号分裂已捕获唯一可静态利用的结构（正/负词极端可达）。T9 的 oracle 词序结果（k=8.3–9/10）反向印证：词级消歧必须供到那个深度，界本身已紧 |
| Booth/CSD 编码（乘法器设计） | 稀疏化 T10 LUT 的索引 | 死：LUT 覆盖全部 2^10 模式，权侧静态化后 CSD 无节省对象 |
| 值预测+验证（CPU 体系结构） | BFP 指数预测免 sop 拍 | **解析杀死**（见 §4.1）：当前 sop 拍已把 h+sign+e 免费打包，sign 本身就是最高位平面，协议已拍数最优 |
| 推测执行+恢复（CPU） | COMPASS 式 spike 推测 | 决策空间静态预测已被 T7 杀（误 fire 37–38%）；**推测+恢复变体是唯一存活可能但无全文**（§1.7） |
| 分支定界最优序（OR/优化） | 按词幅值排序收紧界 | T9 已实测：词序 59–61% vs 位平面 17.4%，轴已定论 |
| 码本离线预计算（Phi） | 权侧静态展开 | T10 dot10→12.8kb 子集和 LUT 即本家族实例，已完成 |
| 冰山/阈值查询（数据库） | 只物化过阈值聚合 | 与证书同构，已被 C1 覆盖 |
| FireFly-S 式"另一 operand 侧" | 权/阈值侧再挖 | 权侧=T10；阈值侧=tau 静态表已离线（tau0–9 ROM）；两侧均已静态化 |

**扫描结论**：GustavSNN 式的移植机会在我们这条语境上已经被 T1–T10 实质穷尽——
这本身是对 T11 "APPLICABLE_NEW=0" 的第二条独立证据链（方法论级而非文献级）。
**没有被穷尽的只剩两个**：(a) COMPASS 式推测+恢复（被全文挡）；
(b) GustavSNN 的**方法论**本身（解析模型先行）——它不是机制，是评估纪律，
可直接移植到 C1 收尾（见 §4.2）。

## 4. 产出：对当前工作的硬件 idea

### 4.1 C1 供数协议拍数最优性（本次精读的解析收获，写进论文）

对 t10_rtl/cert_gate_bitl.sv 逐行核对后确认：sop 拍一次性打包 h(12b)+sign(10b)+e(5b)，
其中 **sign 在数学上就是最高位平面**（2's complement 中 vtop 初始化
= −lut10(sign) 恰是位平面递推在 m=e 层的实例）。因此：

- 当前每组拍数 = 1（sign+e+h sop）+ ~3.2 数据平面 ≈ 4.2 拍（17.3–17.7%）；
- 任何"指数/符号预测免头拍"方案都无法低于此——预测 e′≥e 造成低 bit 截断，
  截断误差必须折进证书界（可保零差但等效抬高 m 起点，拍数不降反升）；
  e′<e 则表示溢出，不可用；
- 同端口侧带（把 e 挤进 sign 平面）与现状等拍。

**结论：C1 的 4.2 拍/组在同端口口径下已达协议下界**（1 拍符号/指数 +
锁定所需最小平面数）。这给 TCAS-II 一个可写的强声明：剩余优化空间只在
数据通路面积/频率（T10 已做）与净服务计费，不在拍数。

### 4.2 Idea（方法论移植，零 RTL 成本）：GustavSNN 式解析模型收尾

把 4 traces 的 17.3–17.7% 升格为解析曲线：拍数/组 = f(margin 分布, ρ)，
拟合 margin（|m−thr|/L1）分布 → 给出任部署点的预期供数比 + 置信区间。
这是 GustavSNN "Nsilent ∝ 1−ρ^T" 的同款动作，直接强化 TCAS-II 评估节
（T6 的 thr 扰动实验已是其特例）。成本：纯模型侧脚本。

### 4.3 Idea（叙事移植，零成本）：Bishop ECP 式论证结构

Bishop 最强的一句论证是"二值性给出 ANN 不存在的 100% 置信界"。C1 拥有
同构且更完整的版本：**消费者是二值阈值比较（输出 {0,θ}）+ 激活二值 + 权界静态
→ MSB-first 供数下逐判决精确区间证书存在，且无需训练配合**（对照 Bishop 需要
BSA 训练才能长出 bundle 稀疏）。相关工作五档组织（近似→精确符号→精确值级→
精确 bit-parallel→启发式 bit 级）恰好把 C1 放在"精确 bit-serial"空档，
论文叙事可直接套用此结构。

### 4.4 未被七篇触碰的差分（维持既有判断）

F1（双完成 last-use：连续 θg 双消费者占用并集）仍是七篇均未涉及的调度侧
结构差分；但注意它是 C1 之前的生产调度线，若投稿只容一个机制（TCAS-II 5页），
C1 优先，F1 列未来工作。

### 4.5 行动清单（按用户工作流"先判适配"过滤后）

| 项 | 类型 | 适配判定 | 下一步 |
|---|---|---|---|
| C1 净服务实验（真实 consumer，同端口计费） | 实验 | 已排第一优先 | t10_rtl 接生产 trace |
| C1 综合/PPA | EDA | 已排第一优先 | t10_rtl 综合 |
| §4.2 解析模型 | 模型脚本 | 适配（零 RTL） | 与净服务并行 |
| COMPASS 全文索取 | 用户行动 | 推测+恢复唯一存活变体 | 加入 11 篇索取清单 → 12 篇 |
| 仿射算术/值预测/CSD/词序重排 | — | 解析杀死（§3 表） | 只写入论文论证，不立项 |

## 5. 一句话总结

七篇标杆的方法论（profiling 先行、解析模型先行、用 SNN 独有性质换精确界、
 operand 两侧分别攻击、静态数据离线化）**恰好就是 T1–T10 已经在执行的纪律**；
把它们当作镜子照当前工作，新增产出有二：**C1 协议拍数下界的解析证明（§4.1）**
与 **GustavSNN 式解析评估模型（§4.2）**；非 SNN 移植空间在文献级与方法论级
双通道确认穷尽，唯一存活外部输入仍是全文索取（现 12 篇，+COMPASS）。
