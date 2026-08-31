# M150 source-stationary destination-K4 独立打铁评审

结论：**58/100，P0=1、P1=4、P2=2。** 数值 opportunity ledger 通过；`1.805357580618×` 硬件性能 admission 明确拒绝。

## 全量独立重算

本评审没有导入或调用 M150/M147 analyzer 或其 schedule class，而是从 M40 packed support、M72 centers、M41 INT8 weights 独立重建全部 20 个 heldout records，再独立实现四 bank recurrence。

| 指标 | 独立值 | 与 M150 |
|---|---:|---|
| `Σ_source ceil(active_dest/4)` | 47,040,777 | 精确一致 |
| Full-4 descriptors | 47,034,985 | 精确一致 |
| Active row/source keys | 23,522,595 | 精确一致 |
| Destination events | 188,148,490 | 精确一致 |
| Destinations/active key | 7.998628127551 | 精确一致 |
| PWP1024 tokens | 60,478,417 | 精确一致 |
| Zero-floor producer cycles | 61,118,882 | 精确一致 |
| Optimistic cycles | 75,032,786 | 精确一致 |
| Optimistic ratio vs M143 B4 | 1.805357580618× | 仅 opportunity |

全零 floor 和排序攻击通过：14,078,105 个全零 raw rows 均计 1 producer cycle；人为 nearest-center tie 在中心排列变化后仍选择数值较小的 `0x0003`。20 个 record key 按 sample 5..9、operator 0..3 完整且无重复。

Unrestricted mosaic 为 47,037,211 descriptors，差额确为 3,566。25,919,638 行差额为 0，只有 362 行为正，最大单行差额 12，没有任何一行 source-K4 小于 unrestricted mosaic。因此算术正确，但 source-K4 不比 M147 mosaic 更快：它多 3,566 descriptors、总周期多 3,196，ratio vs M147 为 0.9999574。

## P0：一个 source 不等于一个可广播 weight vector

M150 把一个 destination contribution vector 定义为 768 bits，并声称相同 source 的一个 vector 可供最多四个 destination block。冻结卷积权重实际为每个 source 对 768 个输出通道各有独立权重；每个 destination block 对应不同的 96-lane vector。

独立审计比较了四个算子的 27,648 个 weight rows 中全部 774,144 对跨 block 向量，**相同数量为 0**。因此：

- 一个 destination vector：768 bits；
- 四个 distinct destination vectors：3,072 bits；
- 完整 source weight row：6,144 bits；
- 当前 16-source partition 的完整 row store：至少 98,304 bits；
- 整个 operator：42,467,328 bits。

所以 `source_vectors_required_per_descriptor=1` 与 `source_vector_payload_bits_int8=768` 不能同时成立。除非把 resident object 改成 6,144-bit full row 并提供四路 block extraction，或实现四个独立 768-bit bank/read path，否则核心硬件机制不成立。

## 其余 admission blocker

### Key-switch / load bubble

乐观 recurrence 没有显式 source-key switch 或 store load-to-use latency。仅作敏感性攻击，每个 active key 加 1 cycle 后得到 96,762,797 cycles，ratio 降至 1.399929×。这不是预测值，但证明 75.03M 对未定接口高度敏感。

### 四写口

47,034,985/47,040,777，即 99.987687% descriptors 是 full4。M149 虽有四输入 signed combiner，但合同明确要求四个 contribution vectors 已经可用，并排除 storage、SRAM ports 和 accumulator commit。

同一 recurrence 的乐观端口敏感性为：

| Destination update ports | Cycles | Ratio vs M143 |
|---:|---:|---:|
| 4（M150 假设） | 75,032,786 | 1.805× |
| 2 | 119,968,634 | 1.129× |
| 1 | 213,772,648 | 0.634× |

所以四路 vector supply、冲突安全 accumulator banking 和四写口缺一不可。

### Signed negate

独立重算发现 17,557,357 个负 residual events 和 4,389,684 个负 descriptors，均约占 9.33165%。M149 的 9-bit-safe negate primitive 存在，但 M150 尚未把 source-stationary descriptor 的 sign/destination materialize 后接到 `tuple_negate` 做 ordered integer replay。因此 support-only 周期不能证明数值正确。

### PWP recurrence 公平性

四 bank recurrence、PWP/correction token、zero release、W384 barrier 和 fair-fixed8 baseline 1,114,863,448 均独立精确复现，算术层是公平的。但硬件资源并不公平：candidate 同时引入 PWP1024 与理想四 destination capability，对比的是 M143 B4/PWP512，且没有匹配 store、ports、area、energy 或 frequency。故 1.805× 不能称作 source-stationary 或硬件 speedup。

## 处置

保留 M150 作为可信的 heldout ordering/opportunity probe；禁止接纳或宣传 1.805×。下一里程碑应先冻结并实现：

1. 6,144-bit full-row 或四个 768-bit block-vector 的 resident/banked supply；
2. key switch、地址和 load-to-use 时序；
3. signed descriptor materialization 与 M149 integer replay；
4. 四 accumulator update ports、bank conflict proof 和 macro-aware PPA。

机器结果见 `m150_independent_hammer_review_r1.json`；全量逐 record 数据、敏感性 schedules 和攻击见 `independent_full_recompute.json`。本评审未修改 production 或 `docs/359`，未 commit/push。
