# M355：M349 FC2 equal-bandwidth K8 vs K1×8 独立打铁评审

结论：**89/100，P0/P1/P2 = 0/1/4。M349 exact-SHA directed 集成与等峰值带宽 `1.000x` GO；正的 coalescing/control 加速、等面积、完整 FC2/FFN、physical/system/headline 全部 NO-GO。**

我在隔离临时树中用未修改的 exact-SHA runner fresh 编译并运行 Synopsys VCS。原 M349 `SHA256SUMS` 的 100 项和二层 seal 全部回放通过；fresh receipt 和 `assert.report` 与封存版本字节一致、无 assertion failure。

| B | 独立重建 events | active bank reads | K8 cycles | K1×8 cycles | ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 120 | 51 | 51 | 1.000x |
| 2 | 41 | 492 | 131 | 131 | 1.000x |
| 4 | 90 | 2,160 | 486 | 486 | 1.000x |
| 8 | 110 | 5,280 | 1,231 | 1,231 | 1.000x |

非零 suite 两边均执行 8,052 个 active bank-word read；零事件均为 14 cycles。周期从 `header_accept` 到 `token_done_accept`，由同一个 posedge monitor 记录，使用 `end-start+1`，已消除 M342 的 active-region 计数竞争。

## 公平性结论

M349 在它声称的层面是公平的：双方都是 8 个 128-bit logical bank、峰值 8 word/cycle 或 1,024 bit/cycle、固定 L4，并从相同 reset-relative edge ordinal 使用相同 request、response、result、done allow 公式。raw payload、source/destination mapping、signed INT8 weight 函数和 Acc24 边界相同。

逐请求/响应 multiset 以 `(block,slice,channel)` 计数，bank 可由 `channel[2:0]` 唯一推出；请求、响应、权重、软件参考和两架构结果均为 0 mismatch。candidate 观察到 1,080 次、baseline 观察到 7,024 次 younger-before-still-live-older response。M218 service SVA 已绑定，八个 M219 service SVA 也全部绑定且 required cover 非零。

但这不是等面积/等 service resource：K8 是一个 M218 O8/FIFO4，K1×8 是八个 M219、聚合 O64/FIFO32，再加 atomic join。它是故意偏强的 cycle baseline。因此 `1.000x` 是可信的等峰值带宽 directed 诊断，但不能推 throughput/mm² 或能耗结论。

receipt 中 `same_raw_weight_request_response_result_and_done_trajectories=true` 也应收窄理解：相同的是 payload、transaction multiset 和周期 allow gate；由于 bundling、per-bank outstanding 与 ready 状态不同，实际内部 request/response 时序并不逐拍相同。

## M342 5.281× 的处置

M342 的 `5.281374845x` 没有被删除：它仍是“单个 bundled request port 下 K8 对 serialized K1”的 exact-SHA directed 观察。但它作为以下含义已正式撤销：

- equal-bandwidth K8 speedup；
- grouping/control-only speedup；
- 完整 FC2、FFN、physical、system 或 headline speedup。

等带宽主表必须写 M349 的 `1.000x`。若保留 `5.281x`，只能放在明确标为 serialized-port sensitivity 的独立行，不能作为加速器优势。

剩余主要缺口是冻结 120-record FC2 trace、等面积/DC/Fmax、真实 SRAM/interconnect 与 SAIF/PTPX。完整证据、评分和门控见 `m355_independent_hammer_review_r1.json`。M349、M342 与 `docs/359` 均未修改。
