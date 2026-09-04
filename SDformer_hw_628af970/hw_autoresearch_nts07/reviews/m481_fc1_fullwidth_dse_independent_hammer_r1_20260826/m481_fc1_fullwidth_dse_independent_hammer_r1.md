# M481 FC1 全宽同资源 DSE 独立打铁与 DATE 增量评审

日期：2026-08-26  
裁定：`PASS_RESULT_GO_TO_VCS_SRAM_GATE__DATE_ADMISSION_UNCHANGED`  
打铁评分：**91/100**

## 1. 身份与重标

FC1 正式里程碑为 **M481**。先前 M480-FC1 与动态 BN 的 M480 身份冲突，两个旧结果目录均已增加 `SUPERSEDED_BY_M481.txt`，原 payload 与 seal 保留但禁止引用；旧合同状态已改为 superseded。`docs/359` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

正式 M481 证据：

| 对象 | SHA256 |
|---|---|
| contract | `53be88fea5beef747e09e1dff2f10a67508d9f89d26e640b93f51a5276a8d556` |
| analyzer | `360de324f38094ce2523870dd0b6551131e586ead1a1ee32ec17b2026995d3ae` |
| exact runner | `b0564e98b1b695975cc0c7aa2ea72c45c122132efc442947a1cecef7b1503700` |
| result JSON | `2a7a1c917cb2f9aa1adb61092c7619de8d9b495aab5550f1fa41291188006578` |
| result CSV | `e4acb794cf3849a2c9fe4eaa4f5b4d4b39cf9e67caad7750ad9392b74fa54973` |
| run `SHA256SUMS` | `fe323dc43a90b2fa33d23fb15c2eb55289b6685819da17c7b581ea340a846713` |
| seal | `3d46c38616375f16ce0b9178cf5ac77b74fc86b25c63a1848589b7d70649d021` |

正式 seal 与内部 manifest 已重新执行并全部通过；wrong-contract-SHA 路径 fail-closed，未生成错误输出。

## 2. 独立复算

独立审计器没有调用 producer 计算函数，而是从冻结 JSON/CSV 重算：

- 4 lane × 3 fanout × 3 chunk × 3 bank = **108 个唯一坐标**；
- 每点 baseline/candidate 九项周期分量求和；
- 同资源倍率、weight-request reduction、stage3 fallback 投影和 envelope 算术；
- 2-cycle factor/weight response、单 factor/weight port、bank issue round 下界；
- 48 个数值过门点；
- CSV 108 行与 JSON 逐点一致；
- 全部 `system_speedup/headline=false`。

结果为 **2,376 checks、0 mismatch**。

## 3. 候选裁定

### 首选：L96_F2_C16_B2

| 项目 | 数值 |
|---|---:|
| clean-latency 同资源 module-lifecycle ratio | **2.018643×** |
| weight request reduction | **2.580060×** |
| scope-corrected ideal envelope sensitivity | **1.089418×** |
| unchanged stage3 fallback | 17,474,490 cycles |
| lane adders | 192 |
| Acc banks/ports | 2 × 1R1W |
| equal-capacity descriptor buffer | 2,816 bit |

该点是全宽过门集合中按 `lane adders → bank count → descriptor bits` 排序最紧凑的点。它同时超过合同的 module `>=1.50×` 和 ideal envelope `>=1.08×` 门，应进入 **full-width VCS + physical SRAM/bank timing** 的下一阶段设计合同。

### 高吞吐边界：L96_F4_C64_B4

该点为 `2.299217×` module ratio、`1.101214×` ideal envelope sensitivity，weight request reduction 同为 `2.580060×`；但需要 384 lane adders、4 个 Acc bank 和 12,288-bit descriptor buffer，并产生 51,785,832 个额外 bank-conflict issue rounds。

它适合保留为 DSE 上边界，不宜作为第一版 RTL：相对紧凑点，硬件并行度约翻倍，而 ideal envelope 只从 `1.0894×` 增至 `1.1012×`。C64 相对 C16 的主要变化也是以 4.36× descriptor buffer 换取很小的周期改进。

### GO/NO-GO

- **GO：L96_F2_C16_B2 进入全宽 VCS、真实 bank/端口 stall 和物理 SRAM 门。**
- **KEEP-DSE：L96_F4_C64_B4 作为吞吐上边界，不优先 RTL。**
- **NO-GO：F1 作为性能主点。** 虽有约 1.666–1.670× module ratio，但 ideal envelope 仅约 1.0695–1.0698×，未过 1.08 门。
- **NO-GO：把 2.0186× 或 2.2992× 写成完整 FC1/FFN/系统倍速。**

## 4. 为什么该结果仍不是 Performance admission

M481 解决了“小宽 M262 是否能与 M230 held-weight fanout 放进同一资源模型”的机会问题，但没有解决以下硬件问题：

1. 96-lane context-factorized datapath 尚无 RTL/VCS；现有 VCS 只绑定 8-lane M262 和另一边界的 M229。
2. analytical model 假设 clean latency/no backpressure；bank conflict 被计数，但 physical SRAM 的 stall、仲裁、read-during-write 和时序未测。
3. descriptor/factor/weight/Acc 容量与端口已结构化收费，但没有宏面积、布线、DC/STA、SAIF/PTPX。
4. `1.0894×/1.1012×` 是把 trace ratio 线性映射到 100,895,624 eligible FC1 ledger 后的 ideal envelope sensitivity，不是执行级系统周期。

因此 Performance admission 仍为 **false**。下一阶段若紧凑点在真实 stall/SRAM 下保住 module `>=1.50×` 且 scope-corrected sensitivity `>=1.08×`，才能讨论升级；若降到 `1.05×` 以下，应退回 traffic/energy 贡献。

## 5. DATE 模拟评审增量

### Novelty

M481 的新意不是“又一种零跳过”，而是：一个 exact source descriptor 携带 8-context/sign mask，以单次 weight response 驱动 bank-aware held-weight replay，并把 stage3 nonbinary fallback 留在同一投影中。这比此前分散的 M230/M262 证据更像一个可以画成数据流图的 FC1 机制。

但该机制与 Prosperity/Phi 的 product/pattern reuse、以及一般 multicast/weight reuse 有明显邻域。没有全宽 RTL、same-resource physical comparison 和消融前，审稿人仍可能把它判为“两个已知优化的合理组合”。所以：

- Novelty 模拟分可从约 **3.2 提到 3.3（+0.1）**；
- **Novelty admission 不改变**，不能单独列成已经闭环的新贡献。

### Performance / Validation

正向变化是，非 Conv 路线不再只有抽象的小宽比值：现在存在一个 scope-corrected、stage3-charged、同端口/同容量的 `2.0186× module / 1.0894× ideal envelope` 紧凑候选。这能把“性能主线真空”改写为“有一个可信、待物理化的 FC1 候选”。

但它没有改变 Performance admission。评审分的合理增量是：

- Validation 约 **2.7 → 2.9（+0.2）**，来自 exact 100-record mask histogram、108 点、fail-closed 和独立复算；
- Performance headline/admission：**不变，仍为 false**；
- Overall 连续分约 **3.1 → 3.2**，仍是 **Borderline Reject/Borderline**，而不是 Weak Accept。

若后续全宽 VCS、真实 SRAM stall、matched DC/PTPX 后紧凑点仍保住 `>=1.8× module` 和 `>=1.08× scope-corrected sensitivity`，M481 可作为 FC1/FFN 贡献段，并与 C2/C3 组成多算子执行故事；仅凭当前 CPU DSE 不足以支撑 DATE headline，也不足以对标 Prosperity/Phi 的已实现系统倍率。

## 6. 打铁评分

| 维度 | 分数 | 说明 |
|---|---:|---|
| identity / seal / fail-closed | 20/20 | 正式 M481 身份闭合，旧 M480 明确 superseded |
| arithmetic / CSV independent replay | 20/20 | 2,376 checks，0 mismatch |
| same-resource fairness | 17/20 | 点内资源相同；跨点无虚构 weighted-area 排名 |
| cycle/resource completeness | 14/20 | 已计 bank/port/latency/overhead/fallback，仍缺 stalls 与 macros |
| claim discipline | 20/20 | 无倍率相乘，无 performance/system/headline admission |
| **总分** | **91/100** | **PASS_RESULT__NOT_PERFORMANCE_ADMISSION** |

## 7. 与现有 M229 FANOUT=2 RTL 的差距

M229 F2 可以复用为 **核心服务岛**，但其 VCS/DC 不能直接当成 M481 L96_F2_C16_B2 的验证或 PPA。

已经相同、可复用的部分：

- 96 lanes、INT8 768-bit weight response、FANOUT=2、两组 96-lane Acc19 add/sub；
- source/context/sign descriptor、4-credit current/next weight prefetch、tag/epoch/source 回执身份；
- request/update stall、错误 descriptor/response、overflow quarantine 的 VCS/SVA；
- F2 3 ns logic-only DC `24,013.21 µm²` 可作为核心面积锚点。

仍不相同的关键边界：

1. **bank 语义不同**：M229 假设 8 个 context bank 都独立，最低 context 优先每拍选两个；M481 只有两个物理 1R1W bank（`context % 2`）。当前 M229 可能同拍选择 context 0 和 2，在 M481 中发生冲突。必须改为 even/odd bank-aware pick，或在 wrapper 中序列化并收费。
2. **chunk/descriptor 前端缺失**：M229 接收已经生成的 context/sign descriptor，mask builder 明确排除。M481 需要 C16 directory/transpose、24-entry maximum chunk directory 和同容量 128×22-bit descriptor buffer。M228 有固定 C32 scanner，可借逻辑结构，但不能原样代表 C16。
3. **完整 lifecycle 缺失**：M229 没有 empty-tile bypass、8-context accumulator zero-init、两周期 factor/descriptor fetch、C16 directory scan、单 commit port 的 24-cycle commit 和完整 done wrapper。
4. **存储/PPA 边界不同**：M229 DC 把 14,592-bit Acc capacity 置于 port cut 外、macro_count=0，也不含上述 descriptor SRAM、directory、bank mux/arbiter 和 commit。因此 `24,013.21 µm²` 是 core anchor，不是 M481 面积。
5. **周期模型不同**：M229 会把下一 weight request 与当前 replay 重叠；M481 当前是保守的 clean-latency lifecycle 求和。接上 M229 后必须以 wrapper VCS 的实际周期替换解析式，不能复用 `2.0186×`。

最短实现路径：

1. 冻结一个只含 `L96_F2_C16_B2` 的 M481 execution contract；
2. 保留 M229 F2 的 4-credit/weight/192-adders 核心，首先把 replay selector 改成每拍最多一个 even、一个 odd context；
3. 在前面增加 C16 directory + factor/descriptor fetch + 128-entry equal-capacity FIFO，在后面增加 2-bank Acc wrapper、init、empty bypass 和单口 commit；
4. 先跑 directed VCS/SVA：bank collision、directory empty/nonempty、FIFO full/backpressure、2-cycle factor/weight response、commit stall、协议攻击；再导出冻结 trace 的 exact descriptor replay；
5. 只有 wrapper VCS 周期仍过 `module >=1.50× / ideal envelope >=1.08×`，才对完整 wrapper 做 matched DC；随后以真实 2×1R1W SRAM macro/DB 跑 STA/SAIF/PTPX。

因此最短答案是：**复用 M229 F2 数据通路与大部分 SVA，不能复用它的现有结果作为 M481 admission；还缺 C16 前端、B2 bank-aware replay、完整 lifecycle wrapper 和宏化存储。**
