# M369：M367 自然 source-stream gate 独立打铁

结论：**92/100，P0=0、P1=1、P2=5；NO-GO A800、RTL、新思和论文贡献。**

M367 双层 seal 通过。96-thread exact-SHA fresh replay 的两个 CSV 与封存
产物逐字节一致；JSON 去除唯一非确定字段 `elapsed_seconds` 后完全一致。
88 条 module-policy 和 8 条 aggregate row 的速度比、drop、B0、误差链均
独立复核通过，violation=0。

## 是否是 M341 没覆盖的新路由

是，但需准确描述：它是同一个 cumulative-budget 家族中的新 selection/
routing 点，不是自动成立的新论文贡献。

- M341：按 4-bit code 排序、16 桶 capture/drain、stable prefix；
- M367：自然 source-ID 顺序，逐 source 执行 skip-and-continue greedy，无
  sort/bucket；
- 但没有 repack 时，drop 只是原 K8 issue 的 bubble，周期不快。M367 想
  获得性能仍需新增 8-active-word→8 bank FIFO 的 compactor。

所以它是 M352 留下的一个真实缺口，但仍属于“新 repack 路由”，而不是
免费与原 K8 同拍共存。

## 1.15x gate

- B256：免费 compaction 也只有 1.114518x，直接死亡；
- B512：免费为 1.236897x，但最乐观 8R、同拍 greedy/route、无限 queue
  只有 1.048615x；注册 8R+D8 为 0.938805x；
- B1024：免费为 1.525942x，但最乐观同拍自然流也仅 1.106350x，注册
  D8 为 1.006120x。

1.15x 对应周期上限 5,810,153,280。B512 最乐观自然流仍多 561,755,680
cycles；注册 D8 多 1,307,059,624 cycles。D16 几乎等于无限队列，说明
调大 FIFO 不能救活。

FC1-only B512 的最乐观/注册 D8 为 1.063395x/0.937607x；selected Conv
为 1.012632x/0.941883x，也没有可偷出的 A800 子范围。

## 成本与误差

Persistent beta table 为 498,816 B、one-bit reference 的 4.0x。八个任意
读若以八份单读 scratch 复制，需要 3,456 B；D8 source-ID payload 为
640 bits。仅加这两项已是 502,352 B、即 4.028355x，尚未含 8R mux、
八级 11-bit greedy chain、8x8 route、valid/tag/head/tail 和 bypass。

廉价的八 bank 单读 metadata 产生 5,667,875,712 个额外 conflict cycles，
因此 B512 仅 0.561015x。昂贵 8R 模型即使不计 timing/area 仍不过线。

严格证明为
`|sum dropped Wq| <= sum exact beta <= sum conservative U <= B`；但它只
是 raw INT8 accumulator 局部界。B512 最大 raw error 为 508，没有精度
结论。

## 问题与最终门槛

唯一 P1 是 active-list frontend、8R、greedy chain、atomic multi-enqueue 和
tagged K8 issue 仍是 CPU recurrence，不是 RTL cycle。由于更乐观的组合/
无限模型都低于 1.15x，这个缺口不会把 NO-GO 翻转。

五个 P2：非零行误填 `b0_exact=true`；B0 行保留未加到总周期的 scan/
conflict diagnostics；4.0x 未含 runtime port/queue；finite FIFO 使用 whole-
word atomic accept；raw bound 不等于 accuracy。

若以后另开跨 task、多 accumulator 或 per-bank 多 lookahead 路由，必须先在
相同 baseline 资源下做到 B<=512、总周期不超过 5,810,153,280，并完整计
metadata/queue；过线后才值得 A800 valid。M367/M369 不对整个机制家族作
普遍死亡证明。
