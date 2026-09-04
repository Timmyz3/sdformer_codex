# M1563 — DATE 模拟评分增量与机制集成审阅（author view）

日期：2026-09-01。该审阅只对已经封存的证据与明确的准入门做评分推演，
不把 source-only、capture-only、metadata 或 opportunity 数字当作性能结果。

## 1. 当前基准分

采用六维 5 分制模拟尺：Novelty 18%、Soundness 18%、Significance 22%、
Implementation 16%、Evaluation 18%、Presentation 8%。当前证据的 author-view
估分为：

| 维度 | 分数 | 主要依据/缺口 |
|---|---:|---|
| Novelty | 3.4 | C1 对象差成立；C2/C3 更像约束映射与专用服务 |
| Soundness | 4.1 | fail-closed、VCS/DC 与独立打铁纪律较强 |
| Significance | 3.2 | C1 `1.75917254x` 仍是 CPU same-ledger；无 decoder-complete 系统行 |
| Implementation | 3.4 | C1 功能 VCS、宏 DC，C2 等带宽 DC，C3 setup；仍缺完整 memory/power closure |
| Evaluation | 3.1 | ep34 capture 已到位，但 decoder address-timed、C2 production power 与统一表仍缺 |
| Presentation | 2.4 | 可写证据已成形，但当前不是完整六页投稿包 |
| **加权** | **3.35** | **Borderline / Weak-Accept 边缘；不是 Strong Accept** |

当前可引用而不能混写的锚点：C1 `1.75917254x` 是四层 Conv 的 CPU
same-ledger component opportunity；C2 是等带宽 `1.01672765x` directed cycle、
`4.541078x` throughput/mm2、logic area `-77.6104%`；C3 是 setup-MET exact
Fixed-T10 island。局部倍率不得相乘。

## 2. 先完成现有收口的评分作用

若不增加任何新机制，只完成以下三项：

1. decoder D0/call0 pilot 后得到 decoder-complete、同资源、address-timed Table-A；
2. C2 production SAIF/PTPX 或可审计的 memory-inclusive energy；
3. C1 把 same-ledger 操作重放接回可执行端口/存储账并保持诚实标签；

则估分约 `3.65–3.80`。这一步主要增加 Significance、Implementation 和
Evaluation，录用作用大于任何尚未过门的新稀疏 idea。若仍没有 unified
full-network `>=1.10x` 行，则 Strong-Accept 评分上限仍约 `3.8`。

## 3. 各 idea 的边际评分

下表“通过后”均指在上述现有收口基线上，而不是从当前 source-only 状态直接加分。

| idea | 现状加分 | 过门条件 | 过门后的总分推演 | DATE 位置 |
|---|---:|---|---:|---|
| **TSBG**（无损 FC row bundle） | `0`；当前只有 producer/source | ordinary row-buffer 同资源基线；局部周期 `>=1.15x`，或周期退化 `<=5%` 且 weight bytes `>=-30%`、memory energy `>=-20%`；每序列守门 | **3.78–3.90** | C2 memory specialization；最优先 |
| **S2 CCBS 16x16**（有损 contribution-bound block skip） | `0`；当前只有 retained metadata/active-bound 候选 | paired AEE：均值 `<=0.02`、每序列 `<=0.03`；相对 C1-enabled/C2-enabled baseline 同资源 `>=1.15x`；O16 必须真实关闭 bank/burst | **3.78–3.95** | 最多一个 C2 optional lossy mode；高收益、高 novelty 风险 |
| **S1 ABCG**（analog boundary gate） | `0`；只搭车采 histogram/debt | 同 checkpoint AEE 过门，且可在 fetch 前使 byte/energy `>=-30%/-20%`；不能仅报激活率 | **3.68–3.78** | C2 frontend 消融或 S2 fallback |
| **ACES/格式自适应** | `0` | 相对固定格式在相同 decoder/FC trace 上能量或流量显著下降，并完整计入选择器/metadata | **3.68–3.78** | C2 能量附录，不是新贡献 |
| **LBWC/ARPE**（bitwidth/compression） | `0` | 先有 authoritative INT8/Acc 定点桥，再证明无损或单列 AEE；同资源计 codec | **不建议本轮加入** | 会触碰量化身份；当前风险大于增益 |
| **T10 phase/rank 改造** | `0` 或负 | 新 checkpoint、重新绑定全部 C3/系统证据 | **3.50–3.70**（排期风险） | 本轮不做；可能改变 C3 故事 |
| **N:M 静态剪枝** | `0` 或负 | ep34 当前无 exact-zero block；须重训并重绑完整硬件身份 | **3.45–3.65**（排期/质量风险） | 本轮不做 |
| **27.5% adjacent overlap / M501** | `0` | 已知是 event/traffic reduction，不是 time reduction；需独立带宽瓶颈才可写能量/流量 | **不升总分** | 最多消融，不复活为周期 headline |

如果 TSBG 与 S2 都完整过门、所有 overlap 收费，并补齐 unified full-network
`>=1.10x` 与 memory-inclusive energy，总分才可能到 `3.90–4.10` 的 Strong-Accept
边缘。若没有统一系统行，即使两个局部机制都过门，也应封顶在 `3.7–3.8`。

S2 的 `29.7%` 仅是 active-bound-mass 下的本地候选窗口，不是 AEE、周期或系统
收益；旧的 `99.2%` 全容量分母不得进入论文。

## 4. 是否替换或伤害现有 idea

结论不是“完全无影响”，而是“现有三条贡献不替换，但前端模式与共享资源必须收费”。

| 新机制 | 功能上是否替换 C1/C2/C3 | 实际微结构影响 | 性能合并规则 |
|---|---|---|---|
| TSBG | 不替换 C1/C2/C3 | **替换** C2 前端逐 token weight-row fetch/scan，保留 typed K8 与独立 Acc24 context | 与 C2 联合重放；不能把局部倍率相乘 |
| S2 | epsilon=0/bypass 不替换 exact 路径 | 在 fetch 前增加 bound/metadata gate；打 Conv 会与 C1 争同一 weight/product work，故正文限 FC/patch | baseline 必须已启用 C1/C2，只报 residual gain |
| S1 | 不替换 exact 路径 | 与 S2 竞争同一个 lossy frontend 位置 | 正文 S1/S2 默认二选一；不能双重累计 drop |
| ACES | 不替换执行器 | 改 descriptor 编码/选择器及 SRAM/NoC 流量 | 必须计 selector、metadata、format conversion |
| RQTB | 与上述功能正交 | 仍共享 SRAM 端口和 phase control | 保持 attention 局部/能量行，不能作系统 headline |
| LBWC/ARPE、N:M、phase/rank | 可能改变数值/训练身份 | 需要新量化或新 checkpoint，现有绑定需重跑 | 本轮不集成 |

因此论文结构仍保持三条：

1. C1：有限容量、single-port 1RW exact product capture；
2. C2：typed-signed K8/共享 Acc24，并把 TSBG（若过门）作为 memory specialization，
   S2 或 S1 最多一个作为 optional lossy mode；
3. C3：Fixed-T10 exact temporal/neuron service 与全网组成。

RQTB、decoder mapping、TSBG/S1/S2 都嵌入上述段落或消融，不新开第四条贡献。

## 5. 执行顺序

1. 先完成 decoder D0/call0 单次 pilot 的独立 release review 与实际校准；
2. 修复 reduced-binary producer 的 permit provenance，重锤后再做一次 capture；
3. 只先跑 TSBG B2/B4/B8，同资源 ordinary row-buffer 作基线；
4. TSBG 过门后才投入 S2 paired AEE；
5. S1 只搭车，N:M/phase/LBWC/ARPE 不占本轮 RTL/EDA 队列。

该排序同时最大化 DATE 分数增量并保护现有 C1/C2/C3 的证据身份。
