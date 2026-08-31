# M739｜M700 官方 Prosperity decoder 重放结果独立打铁

## 结论

**99/100，PASS。** M700 可以进入论文 Table C 的“外部官方 artifact、同 workload 支撑机会”一行，但不能进入 ours/headline/system 表。结果目录与上游 payload 双封存均通过，`docs/359` 仍为 `dedde7ce...`。

唯一 P2 是措辞约束：官方 `run_fc` 在这一路径中消费二值 support 与 K/N 维度，不消费数值权重。因此这里的 exact 是 D0/D2/D3 冻结二值 support 映射 exact，不是 ConvTranspose 数值输出 bit-exact。

## 独立复算

| 范围 | records / calls per mode | bit cycles | product cycles | ratio-of-sums | per-call geo | min–max |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 10 / 40 | 216,215,930 | 78,608,261 | 2.750550× | 2.749678× | 2.704414–2.780745× |
| D2 | 10 / 40 | 283,049,128 | 98,824,495 | 2.864160× | 2.862653× | 2.822719–2.887822× |
| D3 | 10 / 40 | 939,297,789 | 288,485,511 | 3.255962× | 3.244557× | 3.173460–3.315798× |
| **D0/D2/D3** | **30 / 120** | **1,438,562,847** | **465,918,267** | **3.087586×** | **2.944887×** | **2.704414–3.315798×** |

Exact subset 的 product support 从 1,294,417,347 降至 296,884,475，减少 77.0642%；总模型周期减少 67.6122%。这些均与 sealed result 逐项一致。

D1 的 10 records / 40 calls 被完全隔离。其诊断值为 301,911,164 vs 115,417,494 cycles，即 2.615818×，但 folded-weight deployment 未准入，不能混入 exact subset。

## 公平性和结构零

- runner 对 bit/product 两种模式传入同一个 materialized activation，配置只切换官方 `product_sparsity` 模式；所有固定 DRAM、activation、psum、weight-write 与 memory-stall 字段逐 phase 相同。
- product preprocess 已计入；本批 workload 中 preprocess 未超过 compute，因此 `preprocess_stall_cycles=0`。
- 48,424,400 个边界补零占 5,434,560,000 个 materialized entries 的 0.8910%。它们被单列，且 bit support 与 mapper 的 active event 完全一致，不能将其包装为数据/event sparsity。
- D0 的 direct full-N 与 N128×3 全计数 miter 共 80 项，0 mismatch。D2/D3 使用 direct full N，不做虚构 N 扩展。

## 唯一合法论文措辞

> On the frozen H67 ep35 decoder binary-support subset D0/D2/D3 (30 sample-module records; 120 polyphase support calls per mode), the unmodified official Prosperity CPU simulator reports 465.918M product-sparsity cycles versus 1,438.563M bit-sparsity cycles, a 3.088× ratio (per-call geometric mean 2.945×; range 2.704–3.316×). This external-artifact result is a phase-summed support-work opportunity, not our accelerator or monolithic decoder/system speedup; D1 is excluded.

禁止写成“our decoder 3.09×”“完整 decoder/full-network/system 3.09×”“同资源 local RTL 3.09×”，也禁止与 C1/C2 倍率相乘。D1 2.616× 只允许在单独 diagnostic 脚注或附录中出现。
