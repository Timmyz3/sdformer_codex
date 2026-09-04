# M480 dynamic-BN materialization elision 独立打铁评审

## 裁定

**72/100。算术与封存 PASS；baseline hygiene GO；standalone novelty / RTL NO_GO。**

M480 fused schedule 应成为之后所有 BN 机制必须击败的公平基线，但它本身不能作为论文贡献或新加速点。`performance_admitted=false`、`rtl_nominated=false`。

## 独立复算

- 严格 receipt-blind：先从冻结 M232 geometry、M159 语义、M281 协议和 M161 fairness overlay 重建预期，再打开 M480 CSV/JSON。
- 16,198 checks，0 mismatch；18 个 DSE summary 点、432 个 phase-DSE 行完整且主键唯一。
- 每个实际配置为 24 个 current-batch/no-running BN phase，不是 432 个物理 BN；BN1/BN2 为 350,208,000 / 87,552,000 elements，总计 437,760,000，系数对 22,080。
- 全部 16/24/32-bit × 32/64/128 B/cycle × overlap on/off 点逐字段吻合。
- 24 个 barrier、raw retention、raw replay 均未被省略。materialized 使用 1R1W 同地址 read-before-write；fused 只移除 normalized tensor 的 write/read。
- M281 latency=8、II=9。最短 replay/channel 仍大于 II；开启 overlap 后每个 phase 只暴露首结果 8 cycles。

Q24、64 B/cycle、overlap 参考点：materialized 61,568,856 cycles，fused 41,048,856 cycles，表面 local ratio 1.499892x；traffic 5,253,120,000 B 降至 2,626,560,000 B。加上 M159 固定 205,384,111 cycles 后表面 serial ratio 1.083268x。

## P0

1. **公平基线归零 novelty。** M159/M161 已规定 strong dense baseline：流入时累计 moments 并写 raw 一次，barrier 后读 raw 一次，边 normalize 边直接喂 consumer。M480 fused 与它完全同语义。因此 2x traffic、1.4926–1.5000x local、1.0312–1.1738x serial 都只是“弱物化基线应被淘汰”的诊断；相对公平 strong baseline 的机制倍率是 1.0x。
2. **raw store 未落地。** 单 phase 峰值 raw retention 为 140.625 / 210.9375 / 281.25 MiB（16/24/32-bit），elision 后不下降。缺少地址化 SRAM/DRAM 事务、宏容量、延迟和能量，所以不能转成可实现 BN/PPA 性能。

## P1

1. `exact` 只能限定为相同 payload width 下的 schedule exact；尚无 runtime-affine fixed-point、round/saturation、ATLIF downstream miter。
2. materialized comparator 依赖 SRAM 的同地址 read-before-write；未用选定宏验证，24-bit packed bus 也未实现。
3. overlap 复算成立，但假设未计价的 bus-wide affine/consumer datapath；16-bit、128 B/cycle 点最高需要 64 elements/cycle。

## 后续规则

- 保留 fused replay，作为所有 BN 新机制的 mandatory fair baseline。
- 不为 M480 单独写 RTL，也不把它列为贡献或 headline speedup。
- 若继续 BN 创新，必须在同一 24-barrier/raw-retention/raw-replay 合同下击败 fused，并补地址化存储、固定点 miter 与 Synopsys PPA。

复跑：`/opt/anaconda3/envs/pytorch310/bin/python hw_autoresearch_nts07/reviews/m480_independent_hammer_r1_20260826/audit_m480_independent.py`
