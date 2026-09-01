# M1567｜M1563 DATE 评分增量与机制集成独立审阅

日期：2026-09-01（Asia/Shanghai）  
对象：`m1563_date_score_marginal_and_idea_integration_author_review_r1_20260901`  
性质：只读、fail-closed、证据质量与算术审阅  
裁决：**当前加权算术 PASS；三贡献集成方向 PASS；情景分数只可作为 author planning range，TSBG/S2 上界与“无 unified 系统行则封顶 3.8”存在条件冲突，必须修复后才可作为项目决策尺。S2、TSBG、S1、ACES、N:M、phase 的若干证据门需恢复。**

本审阅没有修改 M1563 或任何作者文件，没有运行 EDA、VCS、GPU、capture、SSH、训练、性能模拟器，也没有 commit/push。`docs/359_DATE终局冻结_20260813.md` 未修改，审阅时 SHA256 为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 1. 数据用途与审阅粒度

M1563 的“数据集”不是实测 reviewer 分布，而是：

- 一组主观六维 DATE author-view score；
- 一组对未来证据闭合的 scenario range；
- 一张 idea 准入/集成决策表。

因此可机械检查的是权重、加权和、范围条件和证据门传递；不能把 `3.78--3.95` 当作有统计置信度的录用预测。任何未来总分只有在给出相应六维 score vector 时才能复算。

## 2. 加权算术

权重和：

`0.18 + 0.18 + 0.22 + 0.16 + 0.18 + 0.08 = 1.00`。

当前贡献：

| 维度 | 分数 | 权重 | 加权贡献 |
|---|---:|---:|---:|
| Novelty | 3.4 | 0.18 | 0.612 |
| Soundness | 4.1 | 0.18 | 0.738 |
| Significance | 3.2 | 0.22 | 0.704 |
| Implementation | 3.4 | 0.16 | 0.544 |
| Evaluation | 3.1 | 0.18 | 0.558 |
| Presentation | 2.4 | 0.08 | 0.192 |
| **合计** |  | **1.00** | **3.348** |

所以 M1563 的 `3.348` 与 Markdown 的四舍五入 `3.35` 均正确。当前六维本身也与 M1529/M1266 的证据强弱大体一致：C1 novelty/significance 最强，C2 area efficiency 强，C3 validation 强而性能弱，统一系统评估仍不足。

独立审阅不会把这组主观分数伪装成客观量表。若考虑 M1565 仍阻断 reduced-binary production wrapper，当前 author center `3.35` 可以保留；独立合理带约为 `3.25--3.45`，不是置信区间。

## 3. P1｜情景范围与 hard cap 条件冲突

M1563 明确写：没有 unified full-network `>=1.10x` 与 memory-inclusive energy 时封顶约 `3.7--3.8`。但 JSON/表中：

- closure + TSBG pass：上界 `3.90`；
- closure + S2 pass：上界 `3.95`；
- 这两行的过门条件只有局部 same-resource/bytes/energy/AEE，没有要求 unified `>=1.10x`。

所以两个上界分别超过自身 hard cap `0.10` 与 `0.15`。这不是浮点误差，而是条件集合不一致。

建议把 scenario 明确拆成：

1. `local_candidate_pass_without_unified_system`: 总分不得超过 planning ceiling `3.8`；
2. `candidate_pass_plus_unified_ge_1p10_plus_memory_energy`: 才能使用 `3.9+` 情景；
3. 每个 scenario 保存六维 low/high vector，而不只保存总分。

另外，`existing_closure_only [3.65,3.80]`、`TSBG [3.78,3.90]`、`S1 [3.68,3.78]` 大量重叠，不能解释为独立“边际增量”。例如 TSBG range 下界甚至低于 closure-only 上界。它们只能叫 conditional total-score range，不是 idea marginal score。

## 4. “无 unified >=1.10x 则封顶 3.7--3.8”是否合理

### 作为本项目内部 Strong-Accept 守门：合理

当前论文定位仍是事件光流/SNN accelerator，而不是单一 RTL component note。没有 decoder-complete、memory-inclusive、same-resource unified row 时，Significance 与 Evaluation 同时受限；局部 C1/C2/TSBG/S2 比率不能替代全网证据。因此用约 `3.8` 作为**内部 planning ceiling**是保守且有用的。

### 作为 DATE 普适硬门：不成立

DATE 并没有“全网必须 `>=1.10x`”的统一评分公式。一个明确收窄 scope、宏/功耗/代表工作负载极强的 component paper 也可能录用；反之，仅有 `1.10x` 而无 accuracy、energy、memory、公平 baseline 和多序列，也不够 Strong Accept。

所以正确表述是：

> 对当前 full-accelerator framing，若缺 decoder-complete、memory-inclusive unified row，则内部 Strong-Accept 预测上限约为 3.8；这是项目管理启发式，不是 DATE 规则或统计上界。

`>=1.10x` 应与 memory-inclusive energy、paired accuracy、multi-sequence 和 strongest same-resource baseline 组成联合门，而不是单独充分条件。

## 5. idea 边界逐项核对

### 5.1 TSBG

**M1563 正确：**当前加分为 0；TSBG 是 C2 exact memory specialization；同容量 ordinary row buffer 是强 baseline；不能把 bundle/source 准备当性能。

**需恢复的门：**

- contributor multiset、Acc24、output `0 mismatch`；
- signed/non-unit source 占 admitted bundle `>=5%`，否则只能称 ELSA-style binary mapping；
- cycle 分支须 FC1+FC2 ratio-of-sums `>=1.15x` 且每 sequence `>=1.05x`；
- energy 分支须 cycle regression `<=5%`、weight-byte reduction `>=30%`、memory-energy reduction `>=20%`；
- builder/search、row buffer、destination contexts、bank conflict、tail、commit全收费。

M1563 的 `weight bytes >=-30%` / `energy >=-20%` 记号不严谨，建议统一写 `Delta bytes <= -30%` 或 `reduction >=30%`。

**当前时效更新：**M1565 已证明 M1564 production permit 仍可被 caller-controlled free-space 与 synthetic provenance 绕过。因此 TSBG 仍是 source/capture blocked，当前 score increment 继续为 0；M1563 执行顺序第 2 项还需一轮 successor fix + independent rehammer。

### 5.2 S2 CCBS 16x16

**M1563 正确：**`29.7%` 只是 active-bound local candidate，不是 AEE/cycle/system；`99.2%` global-capacity denominator 禁止；O16 必须真实关 bank/burst；相对 C1/C2-enabled baseline 只报 residual gain。

**P1 缺口：**M1555 的当前裁决是 reference gate 必须在任何 S2 AEE、性能、RTL 前修复。M1563 的执行顺序写“TSBG 过门后投入 S2 paired AEE”，中间少了一步：

1. activity-relative safe reference repair；
2. 仅重筛 16x16；
3. metadata `>=8x` 小于 G11 且 total metadata `<=2%` weight bytes；
4. 同一 `(G,O)` 至少一次 keep、一次 drop 的 dynamic witness；
5. O16 physical fetch suppression；
6. 之后才允许 paired AEE。

如果 TSBG 与 S2 都保留，S2 的 exact baseline还必须启用 TSBG，分别统计“TSBG减少重复row read”与“S2减少整块fetch”，否则会双计memory收益。

M1563 把 S2 正文 scope 收窄到 FC/patch 是合法的保守设计选择，但其 score upside 不能再借用包含 C1 的 `86.52% targetable proxy share`。

### 5.3 S1 ABCG

**M1563 正确：**只作 piggyback/fallback；不能只报激活率；S1/S2 默认二选一。

**需恢复的 veto：**

- S1目标边界在79-module proxy中合计仅`6.022%`，不是系统share；
- metadata + beta read达到saved weight bytes的`25%`即NO-GO；
- beta port造成cycle regression `>5%`即NO-GO；
- 必须观察downstream activity/weight bytes的真实传播；
- paired AEE overall `<=0.02`、per-sequence `<=0.03`。

因此 `3.68--3.78` 只能是“完成既有closure后仍保留S1消融”的总分区间，不是S1本身贡献了`0.33--0.43`分。

### 5.4 ACES

M1563 正确把 ACES 放在 C2 energy/traffic appendix，不当新贡献。但“能量或流量显著下降”过于模糊，弱化了 M1534 的门：

- source tuple exact、escape coverage非零；
- 相对bit-packed/raw4 strong baseline，transport bytes reduction `>=30%`；
- 计header/padding/sidecar后SRAM/NoC energy reduction `>=20%`；
- exposed cycle regression `<=5%`；若ingress是max()且要写latency，局部cycle `>=1.10x`。

在这些结果出现前，ACES的current increment为0；`3.68--3.78`没有可复算dimension vector，只能删除或标为heuristic total range。

### 5.5 N:M

M1563 的方向正确：本轮不做，需要row-local r2、新checkpoint、paired AEE和全硬件重绑。但“ep34当前无exact-zero block”不是M1538证明的事实。M1538证明的是：

- 50% count的oracle N:M仍删除约`21.2%--25.0%` L1 mass；
- 因此**当前checkpoint没有lossless N:M route**；
- 原静态审计还有ATLIF权重混入与少量cross-row grouping问题。

应把“无exact-zero block”改成“无lossless N:M route”。未来门还须包含row-local executable layout、local cycle `>=1.15x`，或bytes/energy分支，以及metadata/selector/decompressor/tail/psum/dense commit收费。`3.45--3.65`同样只是排期风险直觉，不是证据推导分数。

### 5.6 T10 phase/rank

M1563 的“不做、需要新checkpoint并重绑C3/系统”正确，但它把phase pruning与rank改造合为一行，漏了两条不同反证：

- 45张10x10 temporal matrix没有exact zero，phase mass近均匀；phase pruning只有aligned phase mask、C3 local cycle `>=1.25x`并通过paired AEE才可保留；
- 当前exact T10 matrix不是rank-3子集，且没有admitted rank-3 accuracy。

两者都不能写成“已有有损候选”，更不能给正向评分增量。若未来训练，应作为新checkpoint研究线，不进入当前DATE收口关键路径。

### 5.7 LBWC 与 ARPE

M1563 将两者合成一行会模糊identity：LBWC是authoritative INT8后的lossless width/zero compression；ARPE是省低bit refinement的lossy mode，需要AEE。两者都被INT8 authority阻塞是共同点，但应分行，不用“无损或AEE”含混处理。

## 6. 评分范围的安全改写

| Scenario | M1563 原范围 | 独立裁决 |
|---|---:|---|
| current | 3.348 | **算术PASS；author-view center，可保留** |
| existing closure only | 3.65--3.80 | **可作heuristic，但必须给六维low/high vector** |
| closure + TSBG local pass | 3.78--3.90 | **无unified时截到<=3.8；3.9只允许附加unified+energy条件** |
| closure + S2 local/AEE pass | 3.78--3.95 | **无unified时截到<=3.8；且先补reference/metadata/dynamic witness** |
| closure + S1 | 3.68--3.78 | **不越cap但不是可识别的marginal；保留为消融总分直觉即可** |
| closure + ACES | 3.68--3.78 | **缺量化gate和dimension vector；不应给精确总分范围** |
| phase/rank or N:M | 3.45--3.70 | **仅表示schedule/rebind风险；不作为evidence-based score** |
| TSBG + S2 + unified + energy | 3.90--4.10 | **作为aspirational Strong-Accept edge合理，但仍非录用概率** |

## 7. 最终裁决与最小修复

M1563 的核心策略不需要推翻：先收统一系统证据，TSBG优先，S2随后，S1搭车，三贡献保持C1/C2/C3。需要的只是把评分与证据门收紧：

1. 给每个scenario补六维low/high vector和明确predicate；
2. 所有无unified情景显式应用`<=3.8` planning ceiling；
3. TSBG恢复exactness与signed/nonunit门；
4. S2在AEE前恢复reference、metadata、dynamic witness、O16 bank和TSBG-enabled baseline；
5. S1/ACES恢复量化veto；
6. N:M改成“无lossless route”，phase/rank分开；
7. M1565未闭前，不把reduced-binary capture记作implementation/evaluation进展。

修复后，M1563可作为内部DATE收口优先级文档；修复前，只有当前`3.348`算术和三贡献集成结论可直接采用，idea情景总分不可用于论文或对外宣称。

