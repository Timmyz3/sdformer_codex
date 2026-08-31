# M473 r3 sealed full-result independent hammer

日期：2026-08-26  
独立裁定：**PASS CPU-DSE integrity；GO M474 fused-pipeline micro-RTL only**  
评分：**93/100**  

## 1. 一句话结论

M473 r3 的 17,280-phase CPU 账本可信，fused `389,974,420` cycles 的机会足够大，不应直接 KILL；但 `PASS_M473_CPU_DSE_NO_GO` 也完全正确，因为 matching unfused upper 只有 `1.01468x/1.01790x`，使 paired admission gate 得到零 nomination。下一步只能做一个最小 M474 fused 1R1W+RAW pipeline micro-RTL 来证明或杀死 zero-bubble 假设，不能直接展开 full matcher/controller，更不能把 `1.94358x` 写成 admitted 性能。

## 2. 身份与封存

| 项目 | 独立结果 |
|---|---|
| producer result SHA256 | `a415f8474f3a351d123670c2d3691a6414f620e3d60848a9c51242802a6956e5` |
| execution contract SHA256 | `7d169651338ce1b5b02950bc0574b11078ee84da67e486f79916ef1bc962d515` |
| preflight SHA256 | `3c9a66edc2e9bf5dcde4be3a335b0993d66e35accc999021466044b010a2053b` |
| analyzer SHA256 | `e3dab3fbf528e9e3df5365b268af5676804385df9028e81ace3e4cadaf183557` |
| docs359 SHA256 | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未改 |
| inner SHA256SUMS | 7/7 PASS |
| outer seal | PASS |
| frozen-input identity mismatches | 0 |

r1 与 r2 目录都只含 `RUN_ABORTED_WORKER_FORK_OOM_NO_RESULT.txt`，非 marker artifact 数为 0；两次 fork OOM 没有留下 JSON/CSV/NPZ/seal，不能被解释成结果。r3 是唯一 sealed result。

## 3. 全量 sidecar 独立重建

本轮使用独立脚本重新读取 M410 original mask 行，不导入 M473 analyzer 的 mapping、liveness、cycle 或 capacity 函数。对每个 bounded tile 独立重建：

- official-style maximum-popcount/lowest-row subset parent；
- zero-mask parent exclusion；
- residual 与 exact-parent one-cycle issue；
- `(original popcount, row index)` topological order；
- parent refcount、peak-live、future parent 与最大 span；
- exact reconstruction 与 topology mismatch。

| 检查 | 数量 | mismatch |
|---|---:|---:|
| phases | 17,280 | 0 |
| source rows per tile axis | 51,840,000 | 0 |
| row-tile axes | 6 | 0 |
| NPZ arrays | 102 | 0 |
| sidecar cells | 66,096,000 | 0 |
| reconstruction | full population | 0 |
| topology | full population | 0 |

最佳坐标所用 tile64 全量聚合也逐项一致：`input_nnz=92,640,472`、`active_rows=27,305,568`、`parent_edges=18,205,389`、`residual_nnz=42,806,256`、`product_issue_per_block=45,439,249`、`peak_live_max=24`、`max_refcount=56`、`max_parent_span=63`。

## 4. official parity

封存的 128 个坐标全部重新读取原始 masks，并直接调用冻结 Prosperity official `find_product_sparsity`：

| 检查 | 结果 |
|---|---:|
| residual + parent mismatches | 0 |
| sealed check-receipt mismatches | 0 |
| sample coverage | 10/10 |
| operator coverage | 4/4 |
| tile coverage | 6/6 |
| short-final-tile cases | 4 |

因此 r2 发现的 zero-parent P0 已经在真实 sealed run 中关闭，不只是 synthetic self-test 修复。

## 5. points / cycles / capacity / traffic

从独立验证的 sidecar 重新生成全部 `6 tile × 2 bank × 3 bandwidth × 5 CAM × 2 latency = 360` 点：

| 检查 | 数量 | mismatch |
|---|---:|---:|
| scalar + nested capacity | 19,800 | 0 |
| points CSV fields | 9,720 | 0 |
| materiality comparison CSV fields | 2,880 | 0 |
| operator/sample summary CSV fields | 1,920 | 0 |
| receipt/result crosscheck | 1 | 0 |
| independently recomputed nominations | 0 | 与 producer 一致 |

独立复算覆盖以下易出错位置：4-bank 两个完整 pass、8-bank 一个 pass；sample-boundary reset 和 commit；17-pass descriptor scan；`search_rows*ceil(rows/L)`；每 parent edge 的 scratch read；每 active row 的 scratch write；logical/macro-rounded 240 KiB 双门；M468 同 bank/bandwidth strong-zero anchor；fused/unfused 同坐标 paired gate。

## 6. `PASS_M473_CPU_DSE_NO_GO` 到底意味着什么

最佳 128 B/cycle 可行坐标为 tile64、8 bank、64 CAM lanes：

| 模式 | cycles | vs same-coordinate bit | vs M468 same-budget zero |
|---|---:|---:|---:|
| bit | 757,946,784 | 1.00000x | — |
| fused 1R1W+RAW | 389,974,420 | **1.94358x** | **1.94974x** |
| unfused sync upper | 746,979,771 | **1.01468x** | **1.01790x** |

`NO_GO` 不是说 exact product reuse 没有 cycle opportunity，而是说现有 CPU evidence 无法证明这个 opportunity 可物理实现：

- fused 与 unfused 相差 `357,005,351` cycles；
- 按 `8*(parent_edges+active_rows)` 计算的原始额外读/完成代价为 `364,087,656` cycles；
- 因而 **98.05%** 的两模式差距正是尚未证明的 parent-read/row-completion bubble，只有约 7.08M cycles 被 pipeline overlap 隐藏；
- parent scratch 逻辑访问量为 read 20.97 GB、write 31.46 GB（分别约 19.53/29.30 GiB）；这不是可以用一句“1R1W”免费消除的细节。

所以，producer 给零 nomination 是 sound 的。fused 数字目前只能称为“zero-bubble RTL-open opportunity”，不能称为 performance admission。

## 7. 为什么不直接 KILL

彻底 KILL 也不合理，原因有三点：

1. exact mapping、population 和账本已经全量独立通过，机制不是建立在抽样或错误 trace 上；
2. fused opportunity 同时以同坐标 bit 和同预算 M468 为公平 anchor，约 1.94–1.95x，足以支付一次很小的可证伪 RTL 实验；
3. 主要不确定性集中在一个窄问题：dependent stream 能否把 sync parent read、row completion 和 dual psum update 融合在 residual issue 拍内。它可以由 directed VCS/SVA 直接回答，不需要先实现复杂全局调度。

因此裁定是：**不 KILL exact mechanism；不 admission；允许 M474 micro-RTL only。**

## 8. M474 允许的最小范围与硬门

### 允许实现

- 96-lane signed12 current-row accumulator；
- 同拍 dual-update 到 signed19 resident psum；
- synchronous 144-byte 1R1W row-index parent scratch wrapper；
- one-issue-ahead parent read；
- same-address RAW forwarding；
- exact-parent（zero residual 仍一拍）、partial-parent、无 parent、back-to-back parent/child、stall/backpressure directed VCS；
- SVA 证明无丢 issue、无重复更新、parent 来源唯一、scratch write 在 final issue、无 backpressure 时每拍可 accept 一个 residual issue；
- 通过 VCS 后才做该 slice 的 DC/STA，检查 fused critical path，不先开 full controller。

### 禁止扩张

- 不在 M474 中实现 64-CAM full matcher、17-pass scheduler 或全系统调度；
- 不把 behavioral 1R1W memory 当 physical macro/PPA；
- 不把 producer fused cycles 或 VCS directed throughput写成 full-module/system speedup；
- 不因 fused directed PASS 删除 unfused result；两列继续并报，直到 physical evidence 取代假设。

### Kill rule

满足任一条件立即终止 M473 RTL 轴：

1. 无 backpressure 的依赖流不能维持一个 residual issue/cycle；
2. exact/partial parent 需要独立 scratch-read bubble 或 row-completion bubble；
3. same-address RAW 只能依赖未定义的 SRAM read-during-write 行为；
4. signed12 row + signed19 psum dual-update 不能闭合冻结时序；
5. 为闭合时序而增加的固定 bubble 使实测 cycle proxy跌破 admission 门。

只有 M474 VCS/SVA 和最小 Synopsys slice 都通过，才值得另开 full matcher/controller；这不是 M473 r3 当前已经获得的权限。

## 9. 评分

| 维度 | 得分 |
|---|---:|
| frozen identity / double seal | 10/10 |
| full-population exactness | 20/20 |
| official parity | 15/15 |
| cycle/capacity/traffic reproducibility | 20/20 |
| baseline/materiality honesty | 15/15 |
| physical realizability evidence | 5/10 |
| claim-boundary discipline | 8/10 |

正式总分：**93/100**。扣分全部来自 fused physical path、1R1W macro、CAM/scheduler 和 PPA 尚未证明，不来自 CPU 数据质量。

## 10. 最终 claim boundary

- admitted：M473 r3 sealed CPU-DSE integrity、exact mapping、全量 sidecar、周期/容量/traffic 算术、`nomination_count=0`；
- not admitted：fused zero-bubble performance、RTL、Synopsys、macro、energy、PPA、full network、system speedup、DATE headline。

最终状态：**PASS_INDEPENDENT_FULL_POPULATION_RECOMPUTE_GO_M474_FUSED_MICRORTL_ONLY**。
