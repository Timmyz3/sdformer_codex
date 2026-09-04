# DATE 模拟评审 r5（独立评审人 grok）

日期：2026-08-29 13:20 CST。接替 grok r4（2026-08-26，总分 3.2 / Borderline Reject）。
评审对象：Motion / H67 ep35，Codex 当前硬件包（约 M700–M932）。
docs/359 SHA 复核为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改。
不注入 Codex。本评不把未 hammer 的中间 DC 日志写成论文数字。

截点现场：C1 原结构宏 DC（M892）已 quarantine；metadata-pipeline 后继（M912）功能 VCS 已 PASS（M929）；其后继宏 DC（M931）仍在跑，中间负 slack ~5 ns，**不得引用**。C2 等带宽三轴 DC 已 hammer（M903）。C3 Fixed-T10 logic-only DC 已 hammer（M928/M917）。无 `.tex`。Git 仍停在 `c1531749`（2026-08-25）。

---

## 0. 相对 r4 的增量裁定

r4 的致命空洞是「可引用硬件主线真空」：C1 只有 CPU 账本，C2 只有不对等的 4.76×，C3 只有 directed 协议。过去三天把其中两扇物理门真正打穿了。

| 增量 | 证据 | 本评 |
|---|---|---|
| C1 功能 VCS r21 UNIT_DELAY | M879 100/100；13 cover、六类攻击、唯一 PASS | **准入为 directed 功能/覆盖**；1.746753× 仍是 CPU |
| C1 原结构 9-macro DC | M913 forensic：WNS **−7.05 ns** / TNS −73959 ns / 12553 paths；面积 156395 µm²（逻辑 77570 + 九宏 78825） | **3 ns 失败是科学结果，不是 runner 误杀**（runner 另有 HOME/`Error:` 假阳性，但不改变 timing） |
| C1 metadata-pipeline VCS | M929 100/100；14 cover、六攻击、SVA cover | 功能后继成立；**周期未重测**，不得继承 1.75× |
| C2 等带宽三轴 DC | M903 100/100；K1 124620 / K8 131086 / K1×8 585479 µm²；setup 全 MET | **C2 现在有诚实物理点** |
| C2 等带宽 directed 周期 | 五 workload 求和 1913 vs 1945 = **1.0167×**；吞吐/mm² **4.541×**；面积 −77.61% | 摘要只能卖面积效率，不能卖稀疏加速 |
| C3 Fixed-T10 DC | M928 99/100；**62434 µm²**，setup +0.0003 ns MET，0 macro | 可写 component setup/area；hold −0.02 ns 诊断；**无性能分母** |
| M700 官方 decoder Prosperity | D0/D2/D3 product-vs-bit **3.088×**（geo 2.945） | **外部机会**；D1 阻断 complete decoder |
| N2 / LB-FUSE / 双-bank / 全 PIDP | 1.352× 更差 / 0.924× / 1.010× / 0.497× | 负结果地图加强 Soundness |
| 全系统 Table-A | production rows = 0；M910 只开 component annex 1 行 | 系统表仍缺 |
| 正文 | 仍无 LaTeX | Manuscript 未动 |

一句话：硬件从「全是合同」变成了「有一块可引用的等带宽面积表 + 一块功能过的 1RW island + 一块 3 ns setup 过的 T10 引擎」。这够写 **Table B**，不够写 **DATE 摘要里的系统倍速**。

---

## 1. 审稿人会看到的三条贡献（以及他们会怎么拆）

论文若按 docs/524 的三条写，审稿人的拆法如下。

### C1 — Constrained 1RW product capture

**可写：** 在 240 KiB、单口 parent scratch、dead-write-only 约束下，把 Prosperity 式 product 机会变成 H67 四层 bottleneck Conv 的 exact CPU 同账本 **435,293,339 cycle = 1.746753× vs M468 strong-zero / 1.741232× vs same-bit**；directed VCS 证明 island 功能与攻击覆盖；九块 TSMC 128×128b 1RW 宏已实例化。

**审稿人攻击（高杀伤）：**

1. **1.75× 不是硅上周期。** M879/M929 自己把 `rtl_cycle_speedup_verified=false`。DATE 加速器论文的主表几乎总是 cycle-accurate 或 post-synth 时钟下的吞吐。CPU ledger 只能进 Table B「model」列。
2. **3 ns 点已经测过，失败了。** M913：最差路径 `exec_bank_q → psum_write_valid`，到达 9.60 ns / 要求 2.55 ns，**623 级逻辑**。这不是差 50 ps，是差一个数量级的组合锥。后继 pipeline 中间仍约 −5 ns（未封，仅诊断）。在 28 nm 上宣称 333 MHz 目前不成立。
3. **Capture gap 会被用来打 novelty。** 官方 Prosperity 同 Conv 机会 2.46×、decoder 子集 3.09×；你们可执行捕获 1.75×（CPU）且尚未过时序。诚实故事是 gap，不是「我们做出了 Prosperity」。
4. **九宏只绑定 18,432 B 物理容量**，同账本义务 213,376 B。M932 已写明其余 194,944 B 未进这个 DC top。不能说「240 KiB 已闭环」。

**本评对 C1：** Keep 为主贡献候选，但 **Significance 被物理频率卡住**。若投稿日仍是 −5 ns 级，C1 必须降为「功能 island + CPU 同账本」，不能当摘要第一数字。

### C2 — Typed signed K8 fabric vs equal-bandwidth K1×8

**这是当前包里最像 DATE 主表的物理结果。**

| 轴 | 面积 µm² | directed 周期（五条求和） |
|---|---:|---:|
| K8 | 131,086 | 1913 |
| K1×8 | 585,479 | 1945 |
| K8 / K1×8 | **0.224× 面积** | **1.0167× 周期** |
| 吞吐/mm² | | **4.541×** |

**可写：** 共享 Acc24 + 八 bank 的 typed signed source 协议，在等服务下几乎不增加周期，用大约四分之一逻辑面积换同等 directed 吞吐。

**审稿人攻击：**

1. 「八路共享状态机比八个副本小」是预期结果，不是稀疏创新。Novelty 必须落在 **H67 signed/binary ATLIF descriptor、fault-closed completion、与 Conv/FC/decoder 共用同一 fabric**，不能落在 4.54× 本身。
2. 五条 directed shape 不是 120-record frozen FC2 全量，更不是 FFN。M911 已承认 full-trace recurrence 不存在。
3. ZeroWireload、0 macro、hold 未闭。K8 的 18,432-bit context 全是 FF。一旦补权重 SRAM，面积比会变。
4. 若摘要出现 **4.76× vs 单 K1**，这是拒稿级口径错误。M903 已禁止。

**本评对 C2：** Keep 为物理主点。写法必须是 **throughput/mm² 与面积**，周期列写 1.02×。这比 r4 时「只有不对等 4.76×」强一个档。

### C3 — Fixed-T10 neuron island

M518 r11 VCS：17 cycle/tile、0 mismatch。M917/M928 DC：62,434 µm²，3 ns setup MET，0 macro。

**可写：** exact dense 10×10 PSN 服务 island，28 nm 3 ns setup 闭合。

**不可写：** 任何加速比。ep35 满秩，rank-3/PAFT 未准入。hold −0.02 ns 未闭。TDA/MCM-96 在本评同意 M904：DATE 窗内 **NO-GO RTL**。

**本评对 C3：** 降为 **完整性 / 协议支撑**，不要与 C1/C2 并列成第三条性能贡献。否则审稿人会问「第三条的数字在哪」。

### 附录级

- **RQTB：** 局部 1.1865×，旧 envelope ~1.0009×。保留为 attention 完整性。
- **Decoder：** 密度 23.32% 跨三序列稳定；A1-OSG 已是 product 下界；LB-FUSE/PIDP/N2 均杀。只写 C2 覆盖 + polyphase 完整性。官方 3.088× 进 Table C。
- **RQTB 四顶 + Local5：** 仍是预宏、MEMORY_IMPL=0 的旧闭环，与 C1/C2 新岛不是同一张系统表。

---

## 2. DATE 审稿意见（按杀伤力）

### 2.1 值得肯定

1. **等带宽纪律是真的。** 多数稀疏加速器投稿会死在「K8 vs K1」。你们主动做了 K1×8，并且结果是 1.02×——这会让审稿人相信其余负结果不是选择性报告。
2. **失败实验已经是论文资产。** N2 1.35× 更差、LB-FUSE 慢 8.3%、双-bank 1.01×、全 PIDP 0.50×、BN 1.5× 是弱基线。DATE 六页用半栏画这些死亡，比再开 M94x 更像完整研究。
3. **功能与时序身份分离是对的。** UNIT_DELAY VCS ≠ STA；不 `+notimingcheck`；不把 debug 设 false-path。M913 把 −7.05 ns 当诊断而不是 PASS，这是 Soundness 加分。
4. **外部 artifact 对照方法学仍成立。** M472 2.46×、M618 2.37×、M700 3.09× 构成 Prosperity 机会上界。Capture-gap 叙事比「我们 2.46×」可发表。
5. **对象边界比 r4 清楚。** 非 Q/K ATLIF 在 ep35 是 binary（θ=1），G2 analog-inject 被正确降级。少一条假 novelty。

### 2.2 弱点（审稿人会写在 review 里的那些）

**W1 — 没有同资源、decoder-complete 的全网周期（仍致命）**  
旧 620M 分母作废；修正 envelope ~791–804M 仍是分析。Decoder ~22% 没有可执行行。DATE Topic-E 可以接受模块级，但摘要若出现「accelerator for optical-flow SNN」却只给四层 Conv 的 CPU 1.75× 和 FC2 directed 1.02×，Significance 会被打到 3 以下。

**W2 — C1 过不了宣称频率**  
623 级组合锥、WNS −7 ns（原结构）/ ~−5 ns（pipeline 中间值）意味着要么降频到 ~100–150 MHz 再报吞吐，要么再切 datapath。降频会直接吃掉 1.75× 的墙钟意义。流水若加拍，必须重跑 CPU same-ledger；否则审稿人会说你们用未定价的延迟换时序。

**W3 — C2 的 4.54× 吞吐/mm² 需要对抗「显然性」**  
没有 signed-source 协议、completion 语义、跨算子复用，这就是「共享 vs 复制」。Related work 必须主动对齐 SNE/FireFly-T/OpenEye 的多源 issue，并写出 **H67 对象差**。否则 Novelty 停在 3.3。

**W4 — 能量仍几乎为零**  
C1 有 parent-scratch 组件 2.04 mJ（生成宏 datasheet，不是 island PTPX）。C2/C3 无 PTPX。没有 mJ/frame、没有 DRAM J。CICC/Phi 式能效表目前写不出。摘要禁止谈 energy-efficient。

**W5 — 正文仍是零（截止日期级）**  
Abstract 2026-09-13，全文 09-20。今天 08-29。没有 `.tex`、claim registry 未进 production、git 四天不推。Presentation 1.5 会把加权分按住。DATE 双盲六页，**现在每一小时写纸的期望收益已经高于再开一条 decoder scheduler。**

**W6 — 多身份未合成一张表**  
RQTB 旧四顶、C2 M803 岛、C1 M528/M912 岛、C3 M518 岛、decoder A1 模型，五套资源坐标。审稿人无法 dedupe。M698/M706 骨架正确，但 production=0 等于没有表。

**W7 — 流水 C1 可能制造新的口径债**  
M912 若增加 commit 拍数，1.746753× 的分母变了。任何「pipeline 后仍 1.75×」都必须是 **同一 51.84M 行账本重放**，不能口头「功能等价所以周期不变」。

### 2.3 口径红线（写进投稿检查单）

禁止出现在摘要/主表：

- K8 vs 单 K1 的 4.76× 当作稀疏收益；
- 1.746753× 当作 RTL/硅上/系统倍速；
- M472 2.46× / M700 3.09× / M618 2.37× 当作 ours；
- M473 fused 1.94× 当作已实现；
- C3 17 cycle/tile 当作加速比；
- M892 −7.05 ns 的面积 156k 当作 paper PPA（quarantine）；
- 99.4% 非注意力份额当作 skip rate；
- 0.0118 W、1.770× encoder、ep44 AEE 1.2819。

允许：

- C2：等带宽 directed，1.017× 周期，4.54× 吞吐/mm²，−77.6% logic 面积（pre-macro, 3 ns, ZeroWireload）；
- C1：CPU same-ledger 1.747× @ 240 KiB；UNIT_DELAY VCS 功能 PASS；宏 DC 3 ns **未闭合**（可报失败或降频分析，不可报 MET）；
- C3：62.4 kµm² setup MET，17 cycle/tile directed；
- Table C：官方 Prosperity 机会。

---

## 3. 打分

DATE 常用五维；加权沿用 r3/r4：Novelty/Soundness/Validation 各 0.25，Fit 0.15，Manuscript 0.10。

| 维度 | grok r4 | Codex 自审 M707/M709 | **本评 r5** | 说明 |
|---|---:|---:|---:|---|
| Novelty | 3.3 | 3.4–3.6 | **3.4** | C1 约束映射 + C2 对象差清楚；4.54× 本身不增加 novelty |
| Soundness | 3.9 | 4.1–4.2 | **4.2** | 等带宽、负结果、功能/时序分身是目前最强维 |
| Validation | 2.9 | 3.0–3.1 | **3.3** | C2 DC + C3 setup MET + C1 VCS；缺系统行、能量、C1 频率 |
| Fit | 3.8 | — | **3.8** | 事件光流稀疏加速器仍是 DATE Topic-E |
| Manuscript | 1.5 | 2.4（ readiness 自报偏高） | **1.5** | 零正文，不给 2.x |
| **加权** | 3.2 | 3.4–3.45 | **3.4** | Soundness/Validation 上涨被 Manuscript 按住 |

换算：`0.25*(3.4+4.2+3.3)+0.15*3.8+0.10*1.5 = 3.445` → 记 **3.4 / 5**。

录用判断（假设双盲六页按上面红线写，不按幻想写）：

| 投稿形态 | 预期 | 概率（粗） |
|---|---|---|
| **as-is，无正文** | 无法审 | 0 |
| 现证据 + 诚实 Table B + 六页完稿，C1 仍 −5 ns | **Borderline / Weak Accept** | 40–50% |
| 同上，但 C1 3 ns MET 且 same-ledger 重放 ≥1.50× | Weak Accept 偏 Accept | 55–65% |
| 再加 decoder-complete 同资源全网 ≥1.10× + 能量列 | **Accept 竞争** | 65%+ |
| 摘要出现 4.76× 或把 1.75×/3.09× 当硅上 | Weak Reject / Reject | 高 |

**Strong Accept：现在不够。** 不是 idea 数量不够，是（1）C1 频率、（2）全网表、（3）纸，三件套。

与 Codex 自审 3.45 的差：他们把 Presentation 写成 2.4，本评坚持 1.5；Significance/Validation 他们略乐观，因为 1.75× 仍非 RTL。

---

## 4. 如果我是 DATE Reviewer 2 会写的短评

> The work studies a real event-based optical-flow SNN and is unusually careful about equal-bandwidth controls and negative results. The signed-source K8 fabric versus replicated K1×8 is a legitimate physical comparison (≈1.02× cycles, ≈4.5× throughput/mm² at 28 nm pre-macro). The 1.75× product-capture number, however, remains a CPU ledger, and the 1RW SRAM mapping misses 333 MHz by several nanoseconds. There is no decoder-complete system table and no energy/frame. I would accept a tightly scoped component paper; I would reject a system-accelerator claim built on mixed denominators.

---

## 5. 推进（按期望收益，不是按 M 编号）

### P0 — 决定录用形态的三件事

1. **今天下午开始六页 LaTeX。** 骨架锁死：Intro=capture gap（官方 2.46× vs 可执行 1.75× CPU）；C2=等带宽面积效率；C1=1RW 约束映射 + 功能 VCS + 3 ns 失败/降频诚实报；C3=exact T10 island；半栏负结果；Table C 外部机会。不要等 M931 结束才动笔。
2. **C1 频率二选一，禁止第三条结构。**  
   - A：pipeline 后继 DC 若 WNS 仍 < −1 ns，**降频报告**（例如 150–200 MHz）+ 墙钟/mm²，停止再切 RTL。  
   - B：只切 `psum_write_valid` 锥一次，VCS 后 **必须重放 51.84M 行账本**；局部 <1.50× 则 C1 退出摘要。  
   M931 跑完封诊断即可，不要对同一 RTL 再 `compile_ultra`。
3. **一张 decoder-complete 直接周期表（模型行也行，标 [model]）。** 用已有 A1-OSG + C1 CPU + C2 directed 映射到修正 envelope，三序列。没有这一行，摘要不能写 optical-flow accelerator。

### P1 — 服务主表，不开新贡献

- C2 补权重 SRAM 延迟敏感性（命中率 × tRC），否则 4.54× 会被问 memory。
- C1 parent-scratch PTPX 或 datasheet 全岛能量，标 [component]。
- Table-A 继续 0 行是正确的；只允许 component annex（M910 已做）。
- git 推送证据包；claims 冻结 SHA。

### Stop list r5

- 新 decoder matcher / PIDP RTL / 线缓冲 RTL；
- TDA/MCM-96 RTL；
- analog NS-INJECT 当第四贡献；
- 并行第二条 DC；
- 把 M700/M472 乘进 C1；
- 在 C1 周期重放前把 pipeline 写成「仍 1.75×」。

---

## 6. 结论

当前 Codex 硬件是一篇 **诚实的 DATE 组件论文胚**，不是强 Accept 系统加速器。C2 的等带宽 4.54× 吞吐/mm² 是目前唯一站得住的物理 headline；C1 是正确的科学问题（capture gap）但还停在 CPU+功能 VCS；C3 是完整性。Soundness 已经够用。卡录用的是 **频率、全网表、和还没写的六页纸**。

加权 **3.4 / 5，Borderline Weak Accept（有纸）/ 无法审（无纸）**。相对 r4 的 3.2，涨分全部来自 C2 DC 与 C1 VCS，不是来自新 idea。
