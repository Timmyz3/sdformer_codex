# DATE 模拟评审 r6（独立评审人 grok）

日期：2026-08-29 22:00 CST。接替同日 grok r5（13:20，总分 3.4）。
评审对象：Motion / H67 ep35，Codex 硬件包至约 M1008。
docs/359 SHA `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
不注入 Codex。

r5 的核心否定句是：C1 在 28 nm 上差数纳秒才到 333 MHz。本评截点该句被后继结构改写。

---

## 0. 相对 r5 必须改口的增量

| 项 | r5 截点 | 本评截点 | 裁定 |
|---|---|---|---|
| C1 宏 DC setup | 原结构 −7.05 ns；pipeline 中间 ~−5 ns 不得引用 | M993/M1006：**WNS +0.001795 ns**，TNS 0，0 违例，100/100 MET | **3 ns setup 准入为 component** |
| C1 面积 | quarantine 156k 不得引用 | **147,246.392 µm²**，九宏 `TS1N28HPCPHVTB128X128M4S` 全在 | 可进 Table B component 行 |
| C1 发布形态 | M892/M931 失败或 raw | runner rc=9 误杀 GUI 噪声，`dc_shell rc=0`；copy-only 升格，原 `FAILED_OR_INCOMPLETE` 标记保留 | Soundness 加分；主表须披露 provenance |
| C1 存储 | 九宏 18,432 B vs 账本 213,376 B | M1000 仍阻断：不可与 1.746753× 配吞吐/mm² | **未改** |
| C1 RTL 周期 | 无 | 仍无。1.746753× 禁止升级 | **未改** |
| C1 hold | 原结构 −0.08 ns 诊断 | **−0.09 ns，9992 条**，未签核 | **未改** |
| C2 | 等带宽 DC 1.017× / 4.54× 吞吐/mm² | 同；SAIF 源包 GO（M1002），PTPX 未跑 | 能量仍 false |
| C3 | 62,434 µm² setup MET | 同 | 无性能分母 |
| Decoder | 无执行行 | M998/M1008：D2/D3 各 10K 前缀 exact+cycle PASS；commit=0；非 full-row | 诊断前缀，不是系统行 |
| 正文 / git | 无 tex；`c1531749` | 同 | Manuscript 未动 |

C1 时序轨迹（均 TSMC 28 nm、3 ns、ZeroWireload、九 1RW 宏）：

| 身份 | setup WNS | 面积 µm² |
|---|---:|---:|
| M892 原结构（quarantine） | −7.05 ns | 156,395 |
| M931 metadata-pipeline | −4.91 ns | 158,975 |
| **M962/M993 三段 matcher** | **+0.0018 ns** | **147,246** |

---

## 1. 三条贡献的可写边界（审稿人拆法）

### C1 — 有限容量 1RW product-capture

**现在可以同时写两列，但必须分列：**

1. **Model：** CPU same-ledger 435,293,339 cycle = 1.746753× vs M468 zero / 1.741232× vs bit，scratch 213,376 B < 240 KiB。
2. **Component PPA：** 28 nm，3.000 ns ideal clock，setup MET，九 parent 宏，147,246 µm²。功能 VCS（UNIT_DELAY）PASS。

**仍会被打的点：**

- 1.75× 不是这块 147k 岛上测到的 RTL 周期。流水加拍后分母可能变了，**没有同 51.84M 行重放**。
- 147k 只含 parent 九宏 + 扁平逻辑；psum/weight 共约 171 kB 宏代理仍在芯片外端口上。用 147k 去除以 1.75× 得到吞吐/mm²，是审稿人一票否决的拼表。
- hold −0.09 ns / 9992 paths。摘要写 333 MHz 只能叫 setup 点，不能叫 signoff。
- copy-only 从 quarantine 升格：数字真，但必须在 artifact 段写 runner rc9 / `dc_shell` 0。

**本评：** C1 从 r5 的「功能 island + CPU」升为 **「3 ns setup 过的宏感知组件 + CPU 同账本」**。可以进摘要第二句，**不能当唯一系统倍速**。

### C2 — 等带宽 typed K8

不变：K8 131,086 µm² / 1913 cyc vs K1×8 585,479 / 1945 = **1.017× 周期、4.541× 吞吐/mm²、−77.61% 面积**。这仍是最干净的物理 headline。能量链源包已备、PTPX 未执行。禁止 4.76× vs 单 K1。

### C3 — Fixed-T10

62,434 µm²，setup +0.0003 ns MET，17 cycle/tile。完整性，不是第三条性能贡献。

### Decoder

D2 前缀 7,261 cycle / D3 8,976 cycle，10K request，**0 commit**。只能证明 scheduler 前缀 miter，不能外推全层、更不能填 22% 份额。官方 Prosperity 子集 3.088× 仍是 Table C。

---

## 2. 打分

加权沿用 r3–r5：Novelty / Soundness / Validation 各 0.25，Fit 0.15，Manuscript 0.10。

| 维度 | r4 | r5 | **r6** | 相对 r5 |
|---|---:|---:|---:|---|
| Novelty | 3.3 | 3.4 | **3.5** | C1 现在是「约束映射在宣称频率下可综合」，不再只是 CPU 故事 |
| Soundness | 3.9 | 4.2 | **4.2** | 升格 provenance 诚实；hold/存储缺口未假装闭合，不给 4.4 |
| Validation | 2.9 | 3.3 | **3.6** | 三条贡献都有 3 ns setup 组件点；缺 RTL 周期、全网、能量 |
| Fit | 3.8 | 3.8 | **3.8** | DATE Topic-E 不变 |
| Manuscript | 1.5 | 1.5 | **1.5** | 仍无六页 |
| **加权** | 3.2 | 3.4 | **3.6** | `0.25*(3.5+4.2+3.6)+0.15*3.8+0.10*1.5 = 3.545` → **3.6** |

录用（假设按红线写完六页，不把 1.75× 当硅上周期）：

| 形态 | 预期 | 粗概率 |
|---|---|---|
| 无正文 | 无法审 | 0 |
| 现证据 + 诚实 Table B（C1 3 ns 面积 / C2 4.54× 吞吐/mm² / 1.75× 标 model） | **Weak Accept** | **50–60%** |
| 再加 C1 同岛 RTL 周期 ≥1.50×，或降频墙钟仍赢 | Weak Accept 偏 Accept | 60–70% |
| 再加 decoder-complete 同资源 ≥1.10× + PTPX 能量列 | Accept 竞争 | 70%+ |
| 用 147k 去除以 1.75×，或摘要写 4.76× / 3.09× ours | Weak Reject | 高 |
| Strong Accept | 仍不够 | — |

**与 r5 的差：** Validation +0.3、Novelty +0.1，加权 3.4→3.6。涨分几乎全部来自 C1 **3 ns setup MET**。Manuscript 继续把分按在 Weak Accept，到不了 Accept。

DATE 五维口语对照：Originality 3.5 / Technical quality 4.2 / Significance 3.4（全网仍空）/ Presentation 1.5 / Overall **Weak Accept (borderline to accept if camera-ready exists)**.

---

## 3. 若我是 Reviewer 2

> The equal-bandwidth K8 vs K1×8 comparison is convincing (≈1.02× cycles, ≈4.5× throughput/mm²). The 1RW product-capture island now meets a 3 ns setup point with nine foundry SRAMs and 0.15 mm², which removes my previous frequency objection at the component level. I still cannot accept 1.75× as a measured RTL or system speedup: it is a CPU ledger on a larger storage budget than the synthesized island. Hold is open, energy is missing, and there is no decoder-complete application table. Recommendation: **weak accept** as a tightly scoped 28 nm component paper; **reject** as an optical-flow system accelerator.

---

## 4. 口径红线（r6）

禁止：

- 1.746753× 当作 RTL/硅上/系统；
- 147,246 µm² 与 1.75× 组成吞吐/mm²；
- K8 vs 单 K1 4.76×；
- M472/M700/M618 官方数字当 ours；
- 10K 前缀 7k/9k cycle 当 decoder 层延迟；
- 摘要写 333 MHz signoff（setup ≠ hold/CTS）；
- 隐瞒 M962 曾被 runner rc9 隔离。

允许：

- C1 component：3 ns setup MET，147.2 kµm²，9×1RW parent 宏；
- C1 model：CPU 1.747× @ 213 KiB 账本；
- C2：等带宽 1.017× / 4.54× 吞吐/mm² / −77.6% 面积；
- C3：62.4 kµm² setup MET；
- 负结果半栏；Table C 外部机会。

---

## 5. 还差什么才到 Accept

1. **C1 周期桥：** 在 M935/M962 这份 RTL 上重放同一 51.84M 行。≥1.50× 才能把 1.75× 从 model 挪到「同结构」。加拍后跌破 1.50×，摘要只留面积+功能。  
2. **不要用 147k 假装 240 KiB 全闭合。** 要么补 psum/weight 宏再 DC，要么主表写「parent-scratch island only」。  
3. **C2 PTPX**（源包已 GO）：等带宽 pJ，门仍是 ≥2× 能量效率才值得上主表。  
4. **六页 LaTeX，今天。** 3.6 分的硬件已经够写 Weak Accept；没有纸还是 0。

Stop：第四版 C1 为再抠 50 ps；decoder matcher；TDA RTL；把 10K 前缀写成系统。
