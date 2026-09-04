# M1545｜ep34 稀疏候选优先级第一性原理独立审阅

日期：2026-08-31（Asia/Shanghai）
性质：只读、prior-aware、fail-closed 独立审阅
对象：S1 ABCG、S2 CCBS、TSBG；Motion H67 ep34 / M1458 / M1529 / M1534 / M1535 / M1540 / M1541
裁决：**双主推 S2 + TSBG；S1 只 piggyback 同一次 capture，不单独占用 RTL、GPU AEE 或性能收口队列。**

本审阅没有运行 GPU、训练、SSH、VCS、EDA 或生产 fast-kill，没有修改任何既有 source/result、`ucli.key` 或受保护的 `docs/359_DATE终局冻结_20260813.md`。`docs/359` SHA256 复核为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 1. 结论先行

M1540 把 S1、S2、TSBG 同列 T0 是合理的“准许测量”结论，但不是科学 ROI 排名。加入 Amdahl、数据就绪度、prior collision、实现风险与 48 小时判定力后，本审阅给出分阶段优先级：

1. **S2 CCBS 先做现有 retained C1/decoder 上的 CPU local fast-kill。**它覆盖的潜在算子份额最大，而且不等增量 capture 就能先判断 max-weight bound 是否过粗、block metadata 是否真能在 fetch 前工作、是否存在动态 keep/drop witness。若这一步失败，立即封死，不让它重演 G11。
2. **TSBG 是第一条应做完整同资源 fast-kill 的候选。**它无损、不换 checkpoint、不需要 paired AEE；增量 capture 到位后先跑 B2/B4/B8。它的 novelty 上限不高，但最可能在 48 小时内形成一条可审计的 C2 memory specialization，或被干净否决。
3. **S1 ABCG 只在共享增量 capture 中顺手采 magnitude/debt 统计。**它不应成为独立 production 线。M1458 的两个目标边界在 79-module activity-weighted-MAC proxy 中合计只有 `6.022%`；即使局部无限快，该 proxy 的 Amdahl 上限也只有约 `1.064x`。它只有在 capture 先证明真实 weight-byte / downstream-activity 收益时，才允许消费一次 paired AEE；否则直接降为负消融。

因此不是“S1 先、S2 次、TSBG 再等”，而是：

- **立刻：S2 retained-data 快杀；**
- **增量 capture 后：TSBG 完整快杀优先；**
- **S1：piggyback profiling，过本地物理收益门才进入 AEE。**

在任何 fast-kill 通过前，三条均不授权 RTL。最多一条有损模式进入正文；TSBG 即使通过也并入 C2，不增加第四条并列贡献。

## 2. 统一评分

评分只表示“本轮值得投入的测量优先级”，不是已有论文结果。权重：Amdahl/覆盖 `30%`，数据就绪 `20%`，prior 后的可辨识度 `15%`，实现/验证风险 `15%`，48 小时判定力 `20%`。

| 候选 | Amdahl/覆盖 /5 | 数据就绪 /5 | prior 后辨识度 /5 | 实现风险 /5（高分=低风险） | 48h 判定力 /5 | 加权 /5 | 裁决 |
|---|---:|---:|---:|---:|---:|---:|---|
| **S2 CCBS** | **4.5** | 3.0 | 2.4 | 2.5 | 4.0 | **3.49** | **最高科学 upside；先做 retained-data local fast-kill** |
| **TSBG** | 3.4 | 2.6 | 2.7 | **4.0** | **4.2** | **3.39** | **第一条完整 fast-kill；无损、claim 风险最低** |
| **S1 ABCG** | 1.5 | **3.3** | 2.5 | 3.0 | **4.3** | **2.80** | **只 piggyback capture；不得独立占资源** |

S2 与 TSBG 的总分差不足以支持串行等待，正确做法是两阶段并行：S2 用已有 retained data 先筛，TSBG 等同一次增量 capture 后优先闭合。S1 不进入同等级实现竞赛。

## 3. Amdahl 审阅

M1458 / M1529 的 denominator 是 79 个 Conv2d/Linear module 的 `activity-weighted-MAC proxy`，不是 decoder-complete system cycles。以下计算只用于相对排序，禁止写成系统速度。

### 3.1 S1

- `patch_embed.head.conv.0` proxy share：`1.713%`；
- `patch_embed.proj.conv_res` proxy share：`4.309%`；
- 合计：`6.022%`。

在这个 proxy 内：

- 局部无限快上限：`1 / (1 - 0.06022) = 1.0641x`；
- 若 S1 局部仅过 `1.15x` 门：proxy 敏感性约 `1.0079x`；
- 若局部 `1.20x`：约 `1.0101x`。

所以 S1 可以贡献一个“有界 analog ingress gate”消融，但不能承担 Strong Accept 的系统性能缺口。若它只报 source drop，而不能同步减少 weight fetch、compute/psum 或 downstream activity，学术意义不足。

### 3.2 TSBG

- FC1 proxy share：`23.39%`；
- FC2 proxy share：`6.91%`；
- 合计：`30.30%`。

在该 proxy 内：

- 局部无限快上限约 `1.4347x`；
- 局部 `1.15x` 对应 proxy 敏感性约 `1.0411x`；
- 局部 `1.20x` 对应约 `1.0532x`。

TSBG 只减少 weight-row fetch，destination update 数并不减少。因此 cycle 分支是否成立取决于 baseline row-buffer miss / SRAM service 是否是 `max()`；否则它最多是 memory-energy 支撑。这正是必须对同容量 ordinary row buffer 的原因。

### 3.3 S2

若按候选目标合并 patch、FC1、FC2、C1，79-module proxy share 为：

`40.58% + 23.39% + 6.91% + 15.64% = 86.52%`。

在这个非系统 proxy 内，局部 `1.15x` 的敏感性约 `1.1272x`，局部 `1.20x` 约 `1.1685x`。这是三者中唯一从覆盖面上有机会明显补系统行的候选。

但该数字只是“targetable proxy share”，不是可跳 block 比例。S2 的 `M(G,O) A(G)` 是保守上界；16x16 block 中一个大权重就会抬高整个 block 的 `M`，多 active source 又会抬高 `A`。因此 S2 同时拥有最高 Amdahl 与最高“bound 过粗、机会归零”风险。先在 retained data 上快杀比先写 RTL重要得多。

## 4. 数据就绪度

### S2：可立刻开始，但只能 local

M1458 已保留 C1 / decoder 的 FP32/support-sign payload，可立即做：

- block `{8x16,16x16,32x16}` 的 `M(G,O) A(G)` bound；
- `epsilon=0` exact 检查；
- skip / keep 分布与 dynamic witness；
- metadata capacity/read、weight-block fetch 与 issue/psum proxy。

这足以做粗界与碰撞快杀，不足以给高份额 FC1/FC2/patch 或 end-to-end AEE。若 retained C1/decoder 上已经没有低预算 block skip，S2 大概率不值得再抢 GPU capture；若局部存活，再扩到共享增量 capture。

### TSBG：缺一次最小增量 capture，但不缺训练

现有 aggregate activity 无法重建真实 cross-token overlap、weight-row reuse distance、bundle occupancy 或 bank conflict。需要：

- FC1/FC2 per-token/channel support bitset；
- nonzero fixed-point code、sign、non-unit marker；
- token/window/spatial order和 consumer tile；
- weight-row address/bank key；
- ordinary row-buffer baseline address key。

它不需要新 checkpoint、不需要 modified forward、不需要 paired AEE。capture 到位后，B2/B4/B8 的 exact contributor / Acc24 / output 与同资源 schedule 在 48 小时内有很强判定力。

### S1：capture 便宜，但完整准入并不便宜

M1458 只给 sampled range / mean-abs：raw ingress 的 non-binary ratio与很低 mean-abs提示可能存在小量，但没有预提交 grid 的 drop histogram、`beta |x|` debt、后续 sparsity propagation 或 AEE。

S1 的 magnitude histogram / debt 可以 piggyback TSBG/S2 capture，增量成本低；但真正准入还要 modified forward、per-sequence AEE 与 downstream counters。由于 Amdahl 很小，只有 local 物理收益先过门时才值得跑 AEE。

## 5. Prior collision 与可写边界

### TSBG

- ELSA 已明确使用 bundled AER 与 mini-batch spiking Gustavson 来减少通信与 memory access；
- SpikeX 已明确利用跨 time/space 的 weight sharing与数据复用；
- Bishop 也以 token-time bundle 和 inter/intra-bundle weight reuse 为核心。

因此 TSBG 不能宣称发明 bundle、Gustavson 或跨 token weight reuse。唯一可写对象差是：H67 的 typed signed/non-unit source、多个 Acc24 destination context、K8 / 96 lane / 240-KiB / completion 约束，以及“value 不同只复用 weight row、不错误复用 product”。若 signed/non-unit bundle 占 admitted bundles `<5%`，降级为 ELSA-style binary mapping。

### S1

runtime activation pruning/threshold gate prior 很强；AccelTran/DynaTran 已明确在运行时剪 activation，ProSparse 类工作又覆盖训练推动 threshold sparsity。S1 只能主张 event-optical-flow 中两个 non-binary boundary 的对象差、local certified debt 与 C2 fetch-before-compute protocol。单独 novelty ceiling 不足以抵消低 Amdahl。

### S2

Bishop 已有 error-constrained pruning；内部 G11 已做 static beta、top-m 与 token-dynamic cumulative budget。S2 只有同时满足以下四项才不算 G11 换名：

1. 一个 metadata read 控制整个 `(G,O)` block 的 fetch；
2. metadata 相对旧 per-source beta/order 至少小 `8x`；
3. total metadata `<=2%` of weight bytes，pointer/bank/read/debt 全收费；
4. 同一 block 在不同 token 存在至少一次 keep / 一次 drop 的 dynamic witness。

缺任何一项，S2 必须退回 G11 negative result，不计新机制。

## 6. 实现风险与 48 小时门

### P0-A：S2 retained-data 快杀

先不碰 RTL、GPU 或完整 capture。枚举 block `{8x16,16x16,32x16}` 和预提交 epsilon grid，使用强 baseline：existing zero-source skip + 同容量 ordinary row buffer + 同 K8/K1x8 port/BW。

一票否决：

- `epsilon=0` 不 exact；
- local bound violation 非零；
- 无 dynamic witness；
- metadata 不满足 `>=8x` compaction 或 `<=2%` weight bytes；
- 只有 MAC 数下降，fetch / issue / psum 没有同步跳；
- scan + metadata 后不快于 exact K8，且 bytes/energy 分支也不过门。

只有 retained-data 通过，才进入 FC/patch capture 与 paired AEE。完整晋级仍要求 `Delta-AEE<=0.02` overall、每 sequence `<=0.03`，以及 same-resource cycles `>=1.15x`，或 cycle regression `<=5%`、weight bytes `>=30%`、memory energy `>=20%`。

### P0-B：TSBG 完整 fast-kill

共享增量 capture 到位后，先于 S1 modified forward 执行。B2/B4/B8 两臂使用相同 K8/K1x8、96 lane、bank/port/BW、240 KiB 与 ordinary row buffer。

必须计 bundle builder/search、metadata、多个 destination context、bank conflict、tail、update、commit、queue/backpressure。门：

- contributor multiset、Acc24、output 0 mismatch；
- FC1+FC2 ratio-of-sums cycles `>=1.15x` 且每 sequence `>=1.05x`；或 cycle regression `<=5%`、weight bytes `>=30%`、memory energy `>=20%`；
- signed/non-unit bundle `>=5%`；
- baseline ordinary row buffer 与候选同容量。

若只省 weight bytes/energy，保留为 C2 supporting memory specialization；若连该分支不过，封 NO-GO。

### Piggyback：S1

只在相同 capture schema 加 magnitude histogram、预提交 theta/epsilon counters 与 `beta|x|` debt。先设 metadata veto：metadata + beta reads 达被省 weight bytes的 `25%` 或 beta port slowdown `>5%`，立即 NO-GO。

只有静态/local audit 已满足 `weight bytes>=30%`，并观察到 downstream activity变化，才运行 paired forward / AEE。否则不启动 S1 RTL 或独立生产任务。

## 7. 48 小时执行顺序

### 0--8 小时

1. 冻结 S2 retained-data fast-kill schema、block size、epsilon 与四个 collision gate；立即跑 C1/decoder local audit。
2. 冻结 TSBG/S2/S1 共用增量 capture schema；恢复 M1541 的三个 P1：S1 `25%` metadata veto、S2 `<=2%` metadata、TSBG 每序列 `>=1.05x`。
3. S1 只增加 hist/debt 字段，不单独发起 capture。

### 8--24 小时

1. S2 retained-data 未过门：封 NO-GO，增量 capture 只服务 TSBG；过门：保留 FC/patch group payload。
2. 一次增量 capture 到位后先跑 TSBG B2/B4/B8 exact + address-timed schedule。
3. S1 只做 local source/byte/debt screening，不跑 AEE。

### 24--48 小时

1. TSBG 过 cycle 或 bytes/energy 分支，才允许最小 C2 bundle frontend RTL；否则降级/封死。
2. S2 只有 local collision/metadata/physical-saving 全过，才跑 paired AEE；AEE 过门后才能讨论最小 prefetch-kill RTL。
3. S1 只有物理收益先过门才跑 AEE；无论结果如何不抢占 S2/TSBG 与既有 decoder/Table-A 收口。

## 8. 最终论文定位

- **TSBG**：并入 C2 typed signed source service 的 memory specialization；引用 ELSA/SpikeX/Bishop，不单列“新 bundle 算法”。
- **S2**：若所有门通过，作为 C1/C2 的 optional certified fetch-before-compute mode；exact 与 lossy 表严格分开。
- **S1**：最多是 C2 frontend 的 analog-boundary ablation。鉴于 proxy Amdahl 上限，不进入摘要性能 headline。

最终推荐：**S2 与 TSBG 双主推，但角色不同——S2 是最高 upside 的先验快杀，TSBG 是最应优先闭合的无损完整候选；S1 只 piggyback。**这比 M1540 的 `[S1,S2,TSBG]` 执行次序更符合 Amdahl、验证成本和 DATE 收口风险。

## 9. 证据身份与公开来源

冻结内部证据：

- M1458 manifest SHA256：`3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d`；
- M1512 capture hammer review SHA256：`2d4e977dbe23e2c95f647025585c874a45f5d2e285a271b96e11c6ec8b6c22e1`；
- M1529 review SHA256：`8e90e886a5533f168fc497efce16f6995a43988c83dd0c107e0ccde41c22618e`；
- M1534 review SHA256：`88bdd756110b3055127f61b563564d35ce90b6d373c9d73a1f7efbcedf9a4ff1`；
- M1535 review SHA256：`f7d497be4eef3fdeca58cecd8e92c91f36517a3e4b8d527dca01b9a52c2b4f40`；
- M1540 review SHA256：`f1d5754d5e5b5fbb5cad8724d41041e8feb3be2236a343b351aa1d4fe89c3d5d`；
- M1541 review SHA256：`b06ac09f44e78c5eca2e2baf52187e559f87453c5d6414768f2039762f6d7e32`；
- M324 G11 collision README SHA256：`f247106c90592d25c53bb41284f4799196a5eb8121dccc873754aa55d14e8cbb`；
- M370 G7 fast-kill README SHA256：`02a8c69ca062682c234021537a773e3d3bfd9586cac29a17c1baa3e34a16afa9`。

公开原始来源：

- ELSA：https://arxiv.org/abs/2605.20802
- SpikeX：https://arxiv.org/abs/2505.12292
- Bishop：https://arxiv.org/abs/2505.12281
- AccelTran / DynaTran：https://arxiv.org/abs/2302.14705
- SNE：https://github.com/pulp-platform/sne

本审阅没有产生新的 cycle、speedup、traffic、energy、AEE、RTL 或系统 headline 数字。
