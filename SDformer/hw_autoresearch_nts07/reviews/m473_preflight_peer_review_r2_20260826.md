# M473 preflight contract / analyzer 独立复审 r2

日期：2026-08-26  
复审对象：M473 H67 online-subset + row-indexed live-PWP CPU DSE preflight  
裁定：**REVISE_BEFORE_EXECUTION**  
评分：**82/100**  

## 1. 冻结身份

| 对象 | SHA256 / identity | 复核 |
|---|---|---|
| `contracts/m473_h67_online_subset_live_pwp_preflight_contract_r1_20260826.json` | `fae63e5b019c9e60bf2d76d2145a1c6682e87e9c2832bff455f6b7c43d613fce` | 本轮所审最新磁盘版本 |
| `system_simulator/scripts/analyze_m473_h67_online_subset_live_pwp.py` | `4cadc9c98a47d8c0d6bbdfff43874f8cde0b5bad6da73befc8483158ee634733` | 本轮所审 analyzer |
| `docs/359_DATE终局冻结_20260813.md` | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` | 未改 |
| official Prosperity | commit `6ee1c6f1cb419fcf942f2eda63db84ca28248f4b`，dirty lines = 0 | PASS |

合同列出的九个文件型 frozen inputs 全部逐文件 SHA 匹配，包括 M468r6 independent hammer、M467r4 VCS、M472 admission 和 M41 accumulator independent audit。该结果只说明身份闭合，不等于 analyzer 已检查这些 JSON 的语义状态。

## 2. r1 十项 P0 收口复核

| # | 项目 | 合同 | analyzer | r2 裁定 |
|---:|---|---|---|---|
| 1 | full capture / match 完成后才 issue | 明确冻结 | product frontend 在 capture、全 search、17-pass scan 后才进入 work | **GO** |
| 2 | 17-pass stable popcount scan | `17*ceil(valid_rows/8)`，同 popcount 按原 row index | product frontend 精确计入 17 次 descriptor scan | **GO** |
| 3 | CAM lane 物理口径 | L 个并行 16-bit comparator，不是 `L*row_tile` | search 为 `search_rows*ceil(valid_rows/L)` | **GO** |
| 4 | fused / unfused completion 与 sync-delay 成对 | 两种模式均明确，fused 零 bubble 仍标 RTL-open | 两种点在相同 coordinate 生成；upper 加 parent-edge + active-row completion | **GO（模型级）** |
| 5 | parent-edge traffic | 每 child-parent edge、每 output block 一次 144 B read | `parent_rows` 使用 edge count，不用 unique parent；读写均乘 block/pass | **GO** |
| 6 | row-index scratch | original row ID 索引、每 active row 写、one-block 1R1W | logical/macro 都按 `row_tile*144 B`/64-depth rounding，读写账本存在 | **GO** |
| 7 | 不用 compact map/refcount 过容量 | peak-live 仅 diagnostic | 容量完全按 row tile；liveness 不参与 gate | **GO** |
| 8 | DMA / pipeline 方程 | 4-bank 两个独立 432-partition pass；8-bank 一个 pass；每 sample/pass reset | half DMA、command、tail、sample commit 和 pass 数均一致 | **GO** |
| 9 | M468 hammer 证据 | frozen SHA 已列入合同 | 只核 SHA；未核 hammer status/admission 及 128 B/cyc anchor 数值 | **PARTIAL** |
| 10 | M41 arithmetic / PPA 前提 | frozen M41 audit，且已新增 96-lane signed12 row accumulator + dual-update signed19 psum PPA obligation | 只核 SHA；未核 M41 `PASS_INDEPENDENT...` status 和 checkpoint-tight 19-bit 字段 | **PARTIAL** |

十项基础设施中 8 项已实质收口，2 项完成身份冻结但缺语义 fail-closed。合同本身比 r1 显著完整；当前阻止执行的首要问题不是这两项 partial，而是下一节的官方 subset 语义偏差。

## 3. P0：clean-room subset 不等价于 official Prosperity

### 3.1 最小反例

官方 `third_party/Prosperity/simulator/simulator.py` 在选出候选后还执行：若最佳 subset popcount `<1`，则不建立 parent。M473 `cleanroom_subset()` 漏掉该条件，会把全零 mask 当成非零 row 的 parent。

同一冻结 official API 的直接对照结果：

| masks | M473 clean-room `(residual, parent)` | official `(residual, parent)` |
|---|---|---|
| `[0,3]` | `([0,3],[-1,0])` | `([0,3],[-1,-1])` |
| `[0,5]` | `([0,5],[-1,0])` | `([0,5],[-1,-1])` |
| `[0,0,7]` | `([0,0,7],[-1,-1,0])` | `([0,0,7],[-1,-1,-1])` |
| `[0,1,3]` | `([0,1,2],[-1,-1,1])` | `([0,1,2],[-1,-1,1])` |

这不是无害的 tie-break 差异：

1. `parent_rows`、scratch read bytes、unfused parent-read cycles 和 peak-liveness sidecar 会被改变；
2. 合同规定只有 active row 写 row-indexed scratch。零 mask row 不写结果，但错误 child 会读取该 parent，硬件语义不自洽；
3. 正式运行若 128 个随机 official checks 恰好采到该形态会 fail-closed；若没有采到，则可能封存错误账本，结果依赖抽样运气。

### 3.2 当前 self-test 无法捕获

冻结 analyzer 的 `--self-test` 输出仍为：

```text
M473 synthetic self-test PASS cases=4
```

四个 case 没有“唯一候选是 zero mask”的 official corner；self-test 也没有把 synthetic cases 逐例送入 official 函数比较。因此这个 PASS 不能证明其 docstring 所声称的“exactly reproduce official”。

### 3.3 必须修改

在确定 candidate popcounts 后、写 parent 前，若最大 candidate popcount 为 0 必须 `continue`；并至少加入 `[0,3]` 的 deterministic test，要求 row 1 `parent=-1`。synthetic corner suite 应逐例与 frozen official API 对照，而不只验证 XOR reconstruction。

该问题修复、analyzer SHA 重冻并通过 r3 复审前，**不得启动 full 51,840,000-row sealed run，也不得产生 CPU nomination**。

## 4. 其余 analyzer 忠实度

### 已正确实现

- `product_frontend = capture + search_rows*ceil(valid_rows/L) + 17*capture + 2`；empty task 仍支付 scan，符合合同。
- same-coordinate bit/product、4/8 bank、bandwidth、row tile、CAM lanes 的坐标绑定正确。
- fused 与 matching unfused upper 以同一坐标成对晋级；门限同时要求 fused `1.75x/1.50x` 与 upper `1.25x/1.10x`。
- logical 和 macro-rounded capacity 均把 full row-index scratch 计入 240 KiB gate，没有用 measured peak-live rows 偷容量。
- 4-bank 完整 preprocess/work 序列重复两次；8-bank 两个 half-DMA command 后只执行一次完整序列。
- claim boundary 保持 CPU DSE only；CAM、scheduler、1R1W scratch、RTL、Synopsys、energy、system/headline 均未冒充 admitted。

### P1/P2 完整性缺口

1. 128 次 official mapping validation 是 3 个 mandatory trace case 加固定 RNG；没有分层保证四个 operator、十个 sample、每个 tile，也没有 deterministic zero-only subset / tie / later strict subset / exact duplicate corner suite。应把“trace stratification”和“synthetic official parity”分开报告。
2. analyzer 对 M468r6 和 M41 仅作 SHA 检查。建议继续 fail-closed 核对：
   - M468 hammer status/admission，以及 128 B/cyc strong-zero anchors：4-bank `752,580,192`（row tile 192），8-bank `760,350,133`（row tile 64）；
   - M41 status `PASS_INDEPENDENT_CHECKPOINT_REEXPORT_ALL_WEIGHT_LAYOUT_MULTICAST_AND_S10_RAW_CONV_RECOMPUTE`，以及 `accumulator_width.checkpoint_tight_signed_bits=19`。
3. traffic CSV 记录 parent scratch、weight/source 等主要外部载荷，但没有显式记录内部 candidate-store search reads 与 17-pass descriptor reads。当前 energy/physical SRAM 未 admitted，不影响本轮 cycle gate；若字段继续称“per-point traffic”，应增加这两列或明确 traffic scope。
4. point 字段 `bit_frontend_cycles` 实际保存的是含 work/commit 的 bit total cycles，命名容易误解。建议改为 `bit_cycles`，另设真正 frontend 字段。
5. 合同 fail-closed 文本仍写“Fill every TO_BE_FILLED identity”，但当前已无该占位；capacity PPA 列表还重复写了一次 max-popcount/lowest-index reduction，均为非阻塞清理项。

## 5. 评分

| 维度 | 得分 | 说明 |
|---|---:|---|
| exact subset semantics | 12/20 | reconstruction 可成立，但 zero-only subset 与 official 不一致 |
| schedule / cycle equations | 19/20 | full capture、17-pass、DMA、pipeline、4/8-bank 已闭合 |
| capacity honesty | 15/15 | row-index full-depth logical/macro 双门正确 |
| latency / traffic | 13/15 | 双模式和 edge traffic 正确；内部 matcher/descriptor traffic scope 不全 |
| baseline / paired nomination | 14/15 | 成对门正确；字段命名和 bit bypass 口径需更清楚 |
| provenance / validation | 9/15 | frozen inputs 全匹配；official corner 与语义状态检查不足 |
| **总分** | **82/100** | **REVISE_BEFORE_EXECUTION** |

## 6. 最终裁定

M473 的架构合同已从概念性 live scratch 收紧成一个可审计的 full-tile、row-indexed、双延迟 CPU DSE，容量和周期没有明显“免费硬件”漏洞，值得继续。但冻结 analyzer 当前不忠实实现 official Prosperity 的 zero-popcount parent 禁止规则；而该偏差同时触及功能、traffic 和 unfused cycle 账本。

因此本轮为 **REVISE**，不是机制性 NO-GO，也不是 GO。修复 exact mapping、补 deterministic official parity、增加 M468/M41 语义 fail-closed 后再做 r3；通过后才允许正式全量执行。
