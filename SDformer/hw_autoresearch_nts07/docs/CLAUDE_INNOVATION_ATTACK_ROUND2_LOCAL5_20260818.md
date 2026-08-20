# Local5 二轮创新攻击：分数路径上的从未被攻击轴（跨 pair 统计平面 + pc(Q) 位门控残差）

日期：2026-08-18。只读分析，不写 RTL，不改任何既有文件；模型与复算脚本全部在
`/tmp/l5attack/`（CPU-only），输入为封存 trace
`results/local5_fullres_postg0_qfsa_profile100_20260730/ordered_term_items.npz`
（schema `et3_ordered_term_trace_v2`，post-G0 合格，4800 group / 2,160,000 descriptor）。
证据分层：[prof]=封存 profiler/RTL 报告；[模型]=本机对封存 trace 的只读复算（可证伪，
不构成 RTL/周期/PPA 主张）。

## 0. 总裁决

| ID | 候选 | 新存储 | 新执行 | 裁决 | 创新分 |
|---|---|---|---:|---|---|
| C1 | 跨 pair 统计平面 + pc(Q) 位门控残差（score 分解） | 6-bit 统计平面（descriptor 侧，跨 pair 边界保留） | stat-add + m-bit 门控 AND 残差 | **主候选**（条件晋级） | 3.5；4.0 需三闸（§2.6） |
| C2 | RCSD 差分残差数据通路（C1 的子件） | m-bit 残差寄存器 + 差分字母表 | delta 差分 + 定向 popcount | 支持对象，数据门首次闭合 | — |
| C3 | 目的地 gate 5 元组码本 | 241 项 ROM | 查表 | 数据否决（净存储损失 + 433 码本封杀） | — |
| C4 | K-delta 差分字母表闭包（delta 稀疏化） | delta 编码字 | delta 解码 | 数据否决（both-active 不稀疏） | — |
| C5 | 跨 tile gate 商保持 | tile 边界商缓存 | 复用 | 数据否决（mask 不相交 + W4 已覆盖） | — |
| C6 | 跨时相 plane gate 模式复用 | 跨 plane 模式表 | 查表 | 数据否决（0/152,516） | — |

## 1. 覆盖审计：之前各轮真正杀死的是什么，没有杀死的是什么

### 1.1 已封杀清单全部在 term/issue/accumulator 侧

| 来源 | 封杀项 | 对象侧 |
|---|---|---|
| docs/432 | Q==0 不打分、逆模板编译、相等 gate 合并、五色 1RW 无冲突写 | term/issue/accumulator |
| docs/433 | cache、码本、2-wide、第三 stencil、近似剪枝、分配律换序 | term/issue/accumulator |
| docs/439 §1 | 不再造一轮 FCSR/TCFM5 换对象 | term/issue |
| 前一轮 L1 | FCSR/TCFM5 换对象（时间 stencil 等），引用 432/439 | term/issue |

**没有一个封杀项落在 score 计算路径的算子合同上。** 分数路径上现有的两项机制
（QS、ident-K 广播）都是 score 叶的退化特判，封存列与 RTL 均在
`rtl_qfit/qfit_local5_qsilent_score_leaf.sv`；score 路径本身是硬件执行对象
（score_service 44,614 cycles / TCFM5 L1 283,664 = 15.7%，docs/425），QS/ident-K
作为硬件机制是已接受贡献。攻击 score 路径 = 攻击硬件侧，不违反 433 的"硬件侧
无候选"结论——那个结论只对 term/issue 侧成立。

### 1.2 分数路径三件套的精确覆盖（recount 与封存报告一致）

- QS（Q==0 → score=32−pc(K)，6-bit 统计量）：190,575 条 silent edge；popcount
  eval 190,575→45,000（−76.39%）；score 侧 K 读 bit −72.70% [prof]；
  python 与 RTL recount 一致：Q==0 40,257（100 group × 450 destination 口径）。
- ident-K（非静默且 5 个有效邻域 K 全同 → raw16 广播）：3,396/4,743 非静默
  destination = 71.6% [prof]（python/RTL 双口一致）。
- leftover（非静默、非 ident-K，唯一完全密集分数路径）：4,743−3,396 = 1,347/45,000
  = **3.0%**；其中 uniform-relation 证书在 QS/identK 之外可证行只有 711（1.58%）
  [prof]。这些 edge 仍走 `local5_axnor_score_q7.sv` 的原始逐边
  q_count/k_count/overlap_count（3×32-bit popcount + 1×32-bit AND）——**逐边对象
  产生后即弃，从不跨 pair 边界复用**。

### 1.3 RCSD：弃置但从未判死

docs/150 §4 主候选一 RCSD（Remainder-Carried Stencil Delta）：score 分解为
`S7(Kr)=RNE((A0+Δr)/16)`、`Δr=65(pc(Q&U)−pc(Q&D))−(pc(U)−pc(D))`、
`U=~K0&Kr、D=K0&~Kr`。§4.5 设了晋级门槛：delta engine 的 trace work ratio 相对
四个完整邻居 32-bit score 不高于 `0.50`、p95 不高于 `0.75`、`EDP_RCSD <= 0.85 *
EDP_direct5`、与冻结软件逐整数零失配。原文第 5 条写明"第 3、4 条是工程筛选门槛，
**不是现有结果**"——即门槛从未被数据跑过。docs/150 之后（G0 前）RCSD 从文档中
消失：**被弃置，未被数据否决，未被任何轮次判死**。前一轮攻击全文未提及 RCSD。

### 1.4 两 pair 统计边界：全文档库从未出现的对象

Local5 的 pair 语义：frame f+1 是 pair p 的 T1（作 K，需要 k1=pc(K)），同时是
pair p+1 的 T0（作 Q，需要 q1=pc(Q)）。**同一 descriptor 的 popcount 被需求两次：
一次作为 k1、一次作为 q1。** 该统计量现在每次逐边重算；若在 descriptor 侧保留 6-bit
统计量跨 pair 边界，则 q1[p+1] 与 k1[p] 是同一份存储。这与 Motion 389"quotient
保持到新边界"是同族但不同对象：389 封杀的是 slot 侧 score **值**共享（score-front
CSE），C1 保留的是 descriptor 侧 score **分量统计量**，不共享任何 score 值、不改
slot FIFO、不改目录、不改投影。

本机对封存 trace 的复算：`k1[p] == q1[p+1]` 在 2,159,999 个相邻 pair 边界上命中
**1,714,184（79.36%）**（[模型]，两个方向一致）。剩余 20.6% 在 trace 层无法解析
（npz 只存 source 侧、无 Q 标签），因此该身份是**可测的 workload 规律，不是代数
恒等**；exact 合同必须由带 Q 标签的 dump 裁决（§5.1）。这是 C1 诚实上限的直接来源。

## 2. C1 主候选：跨 pair 统计平面 + pc(Q) 位门控残差

### 2.1 合同一句话

五路分数从"五个独立 32-bit XNOR+popcount"改为精确分解：`q1/k1` 来自跨 pair 边界
保留的每 descriptor 6-bit 统计平面（一次写入、两次消费），`n11` 用 `pc(Q)` 位门控
AND 残差计算（m=pc(Q) 位，而非 32 位），逐候选 RNE((65·n11+32−q1−k1)/16) 不变。

### 2.2 位级存储构造

- **统计平面**：每 descriptor slot 增加 6-bit `stat`（popcount 的 Q7 精确值，范围
  0..32）+ 1-bit `stat_valid`。FCSR 有界生命周期 ring（3 行 × W × 5 × 9 bit gate）
  的 descriptor 槽位扩展 7 bit；或独立 2 行 × W × 7 bit 平面。写入一次（source
  retire 时），消费两次（本 pair 作 k1、邻 pair 作 q1）。Q==0 时 stat==0，退化为
  QS 现有路径（score=32−k1），**QS 是 C1 的 stat==0 特例**。
- **残差通路**：m-bit 门控 AND + m-bit popcount（m=pc(Q)）；不需要 32-bit XNOR。
- 规模：W=450 行宽的 plane 为 2×450×7 = 6,300 bit（一行 3,150 bit），相对 gate
  ring（3×450×5×9 = 60,750 bit）为 +10%；相对 descriptor 原始 K 字无新增（stat
  是 K 字 popcount 的压缩，QS 路径已论证同样压缩）。

### 2.3 新执行对象

- stat-add：score 合成 `(65·n11+32−q1−k1)/16` 中 q1、k1 来自平面读数（两个 6-bit
  add），不再是逐边 2×32-bit popcount。
- m-bit 门控 AND 残差：`n11 = pc(Q_m & K_m)`，m=pc(Q)。本机复算 pc(Q)|Q≠0：
  mean 3.88、p50 3、p95 9、max 21；≤2 占 40.8%、≤4 占 65.1%、≤8 占 92.5%
  （[模型]）→ m=8 位残差覆盖 92.5% 非静默边，m=4 覆盖 65.1%。
- 与 ident-K 正交：ident-K 是 K 侧广播（5 个角色共享同一 K 字），C1 是 Q 侧统计
  复用（q1/k1 跨 pair 共享）；二者可叠加，leftover 3% 是两者都不成立的完全密集
  路径，也是 C1 残差通路收益最确定的部分。

### 2.4 与既有机制的关系

| 边类型 | 占比 | 现机制 | C1 后 |
|---|---:|---|---|
| Q==0（silent） | 89.5% destination | QS：6-bit stat | 不变（stat==0 特例） |
| 非静默 + ident-K | 71.6% 非静默 | raw16 广播 | 广播保留；n11 改 m-bit 门控 |
| leftover | 3.0% | 原始 32-bit 叶 | 全分解：stat + m-bit 残差 |

### 2.5 前人工作占位

PADE（systolic XNOR score）、SpAtten（token 级稀疏，非 per-position 统计）、
Transitive Array（GEMM 闭包，不涉及 attention score 分量）、FuseMax（block 级
two-pass max）、TeAAL（非精确 attention）；SDFormer 项目自身文献（docs/13 扩展
清单）无"相邻 pair 共享 Q/K popcount 统计量"的存储对象。与 389 的边界：
389/445 封杀"score-front CSE（score 值共享/quotient）"，C1 不共享 score 值、不
共享 quotient，共享的是 score 的**输入统计量**（descriptor 侧 6-bit），存储与
执行对象均在 descriptor/score 边界，不在 slot 边界。RCSD（docs/150）是唯一同族
先例，但它是残差**数据通路**设计（C2），从未定义跨 pair 统计平面这一存储对象，
且从未过数据门。

### 2.6 创新分论证（为什么 3.5、4.0 需什么）

- 3.5 依据（对照 407：4.0 需"新机制改变现有 descriptor/term/accumulator 的物化
  对象或跨算子执行边界"）：C1 改变了 descriptor 侧物化（新增 6-bit stat 平面，
  跨 pair 生命周期）与 score→gate 执行边界（m-bit 残差替换 32-bit XNOR）——满足
  "新存储 + 新执行"两腿，且该轴（分数计算分解）在 432/433/439/前一轮全部文献中
  从未被攻击。比 H81 预评 2.8、当前 Local5 3.1 高，低于 4.0 因为：覆盖对象是
  score 路径（总周期 15.7% 的上限，docs/425），且 pair 边界身份目前只是 79.36%
  的 workload 规律而非 exact 恒等。
- 4.0 的三个必要条件（任一不过则维持 3.5 或降级）：
  1. rank-1（ep44）口径带 Q 标签 dump，`q1[p+1]==k1[p]` 在目标 stage 达到可主张
     的 exact/统计一致率，且统计平面实现与 32-bit 叶同端口 miter 零失配；
  2. 与 QS+ident-K 组合后的同端口活动对照（score K-read bit 在 72.70% 基础上再
     降，m-bit 残差 vs 32-bit 叶的同资源对照）在任一 stage 不出现退化；
  3. SAIF/PTPX 锚定：score datapath 的 m-bit 门控 + stat-add 活动相对
     direct5 叶的 EDP 改善可测（复用 RCSD 150 的 `EDP <= 0.85x` 门槛作为对照
     基准）。
- 4.0 叙事（若能过三闸）："五路 stencil score 的每位置 popcount 统计量跨相邻
  pair 边界保持（一次生成、两次消费），残差数据通路只处理 pc(Q) 位门控差分"——
  是 descriptor/score 边界上的新算子合同，已有先例只有 Motion 的 quotient 保持
  （389，且对象在 slot 侧）；此时工程上它是唯一为"统计保持 + 差分残差"存在的
  engine。

### 2.7 三大死穴 + 锚定证据

1. **389/445 score-front CSE 类封禁的邻接风险**：若 C1 被读成"帧间分数复用"
   （Motion 已封的类别），整条线死亡。化解：合同必须钉死"共享的是 popcount 输入
   统计量，不是 score 值、不是 quotient、不发生在 slot 侧"；锚定证据是
   docs/389 §5、docs/445 的类别定义原文对照。
2. **433 码本封杀线的吞噬**：若 C1 被写成"统计量查表"，即落入"码本"封杀。化解：
   C1 没有表——6-bit 平面是精确算术值（与 QS 的 6-bit stat 同一压缩，已有接受
   先例），残差是门控 AND 而非查表；锚定证据是 QS 的 [prof] 报告（同一 6-bit
   统计量对象已被接受）与 docs/433 §4 原文。
3. **边际工作量**：score_service 44,614 / TCFM5 L1 283,664 = 15.7%（docs/425），
   leftover 只占 3.0%；ident-K 边虽可叠加但广播已吃掉 K 侧大部分工作。若 m-bit
   残差在同端口对照下收益被 QS 吸收，C1 只剩工程敏感度。锚定证据：docs/425 的
   cycle 明细与所需同端口 miter/SAIF（§5.3、5.4）。

## 3. C2：RCSD 差分残差数据通路（C1 的子件）

合同：C1 的 m-bit 门控残差继续分解为 anchor 差分——`n11 = pc(Q&K0) + pc(Q&U) −
pc(Q&D)`，其中 `U=~K0&Kr、D=K0&~Kr`、`pc(U)+pc(D)=pc(K0 xor Kr)`；anchor 项
`pc(Q&K0)` 每 destination 一次，邻居项在差分字母表上计算。

数据门第一次闭合（docs/150 §4.5 第 1、2 条从未跑过）：本机复算 [模型]，
valid 邻域 K 对（both-active，n=189,056）：
`pc(K0 xor Kr)` mean **6.77**、p50 6、p95 13、p99 16、max 22；xor≤1 仅 1.02%、
xor≤8 为 70.4%。结论：**差分 popcount 均值为 6.77（32 位的 21%），但不存在小
差分字母表**（p50 即 6，分布宽），因此"delta 稀疏/差分查表"（C4）死，而"门控
残差 + 差分定向 popcount"（C1+C2）活：残差工作量 = 每边约 m(=3.88) + 2×|U|/|D|
位 popcount，相对逐边 2×32-bit 有明确数量级差距，但必须由同端口 RTL 对照裁决
（150 门槛第 3 条：work ratio ≤0.50、p95 ≤0.75）。

C2 不单独成贡献（它是 150 已声明的对象），它作为 C1 的残差子件存在；其独立价值
是把 C1 从"Q 侧统计复用"变成"Q 侧统计复用 + K 侧差分残差"的双侧合同，让"新
engine 只为此合同存在"的 4.0 叙事更完整。

## 4. 被数据否决的候选（本机复算明细）

- **C3 目的地 gate 5 元组码本**：241 个 distinct 模式，(16,16,16,16,16) 占
  65.0%（197,802/304,327）[模型]。否决理由：净存储损失（模式表 ROM ~10.8 kbit >
  节省的 ring 位 ~1.7 kbit）+ docs/433 码本封杀。注意该 65% 均一结构 ⊂ ident-K
  （ident-K 时 5 角色同 K、同 Q 只产生单一 gate），已被覆盖，不能另立对象。
- **C4 K-delta 差分字母表闭包**：both-active `pc(K0 xor Kr)` mean 6.77、p95 13、
  xor≤1 仅 1.02% [模型]——不稀疏，无小字母表可闭包。
- **C5 跨 tile gate 商保持**：tile mask 是每 tile 不相交子集，商无可跨 tile
  复用；产物已由 W4 cache 语义覆盖（docs/404）；与 docs/403、410 的否决一致。
- **C6 跨时相 plane gate 模式复用**：跨 plane 相同 5 元组模式的 destination
  数为 0/152,516 [模型]——数据上不存在。

## 5. 晋级所需锚定证据（按序）

1. **带 Q 标签的 rank-1 dump**（ep44 口径）：解析 pair 边界 79.36% 的剩余语义，
   并把 `q1[p+1]==k1[p]` 的统计一致率在目标 stage/head 落表；这是 C1 exact 身份
   的裁决器。
2. **差分字母表分 stage 统计**：|U|、|D|、|Q&U|、|Q&D| per stage/head/window
   （150 §4.5 门槛 1、2 的正规补跑）。
3. **同端口 miter**：stat 平面 + m-bit 残差 vs 32-bit XNOR 叶，逐 (Q,K) 整数零
   失配（对齐 150 门槛 5 与 425 的 360,000 Acc32 零失配口径）。
4. **SAIF/PTPX**：score datapath 活动对照（复用 150 的 `EDP<=0.85x` 门槛；工程
   服务器执行，本机只准备活动合同）。
5. **433 封杀线措辞对照**：合同文案与 docs/433 §4、docs/389 §5 的类别定义逐条
   对照，明确"统计量 ≠ score 值"。

## 6. 底线

- 前一轮"Local5 硬件侧无候选、封顶 3.1"的结论只对 term/issue/accumulator 侧
  成立；score 计算路径的算子合同（逐边 3×32-bit popcount + 32-bit AND）从未被
  任何轮次攻击，RCSD 被弃置未判死。
- 主候选 C1（跨 pair 统计平面 + pc(Q) 位门控残差）是本轮唯一满足"新存储 + 新
  执行"的 exact 候选，诚实估 3.5；4.0 需要三闸（Q 标签 dump 的 exact 身份、
  同端口零失配 + 无 stage 退化、SAIF 锚定的 EDP 对照），其中第一闸当前只有
  79.36% 的 workload 规律支撑，是最大风险。
- 被否决候选均给出本机复算数字（C3 净存储损失、C4 mean 6.77 不稀疏、
  C6 0/152,516），不把"无新对象"当作先验。
