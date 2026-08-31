# M152 独立打铁评审 r1

结论：**69/100，P0=1 / P1=5 / P2=2**。M152 的 destination grouping / `d mod 4` 零冲突算术通过；M150 的单 768-bit 向量广播假设仍然失效，因此 75,032,786 cycle 和相对 M143r2 B4 的 1.805357581× 仍不是硬件 admission。

## 全量重算结果

独立审计器没有导入 M152 production analyzer，而是从冻结 packed source plane 和 INT8 权重重建 20 条 heldout record，重算 25,920,000 个 partition-row、23,522,595 个 active source-key 和 188,148,490 个 destination event。

- ideal K4 descriptor：47,040,777。
- mod4 bank-safe descriptor：47,040,777。
- conflict key：0；extra descriptor：0。
- 候选 recurrence：75,032,786 cycle，与 M152 逐字段一致。
- destination update 没有丢失或重复。

零冲突是真实的冻结数据结果，但不是架构恒真。所有 active key 只出现三种非零 destination mask：

- `0xff`：23,516,803，占 active key 的 99.975377%；
- `0x20`：4,413；
- `0xdf`：1,379。

完整枚举 256 种 mask，其中 82 种在 mod4 下会比 ideal K4 多一轮。所以 paper-safe 口径必须保留“20 条冻结 H67/Motion heldout record”限定。

## 与 M150 P0 的分界

mod4 结果只回答“哪些 destination 可以同拍写不同 accumulator bank”。它不回答“四个 destination 的权重是否相同”。

冻结 INT8 权重重算共比较 774,144 对跨 destination 的 96-lane 向量，相同数为 **0**。因此：

- 可复用的是 source activation/key；
- 四个 destination 仍需四个不同的 768-bit 权重向量，或一个 6,144-bit full source row 加抽取网络；
- M151 的 generic single-vector broadcast 不等价于冻结 Conv payload。

## correction overlay 判定

现有 `contracts/m150_m151_m152_cross_destination_vector_identity_correction_overlay_r1_20260824.json` 精确绑定当前 M152 analyzer/result/contract SHA，同时保留零冲突算术并把 candidate cycle 与 1.805× 硬件 admission 设为 false。因此**不需再叠加第二个 overlay**。

但原 M152 contract 单独读取仍不安全：它保留 `heldout_cycle_model_admitted=true`，result 也保留 `source_vector_read_once/reused=true`。所以当前必须执行 overlay-aware consumption；下一版 contract 应直接改为 `bank_grouping_arithmetic_admitted=true` 与 `hardware_cycle_admitted=false`。具体建议见 [correction_recommendation.json](correction_recommendation.json)。

## 硬件 admission 剩余门槛

- 四路独立 destination 权重供给；
- 四 bank accumulator RMW、same-address forwarding 和 SRAM macro；
- source-key switch/load-to-use 时延（仅每 key 加 1 cycle 的独立敏感性已从 75,032,786 增到 96,762,797 cycle，倍率降至约 1.39993×）；
- 17,557,357 个负事件的 ordered signed INT8-to-Acc19 replay；
- macro 带宽、时序、面积和能耗。

本评审只新增 review 证据，未修改 production、contracts 或 docs/359。
