# M471R2 独立 hammer 结论

结论：数值 DSE 决策通过独立重算，评分 **93/100**；确认 **0 nominations**，继续 `KILL_G15_AS_HEADLINE_RETAIN_ONLY_NONHEADLINE_SIDECAR`。本 review 不准入性能、RTL、PPA 或 headline。

## 独立重算结果

- producer 三个 payload 的 inner seal 与 `SHA256SUMS` outer seal 均通过。
- 重新汇总 17,280 条 phase、36 条 row-tile/K 记录；phase/row-tile mismatch 为 0。
- K1 的 PARENT 数在每条 phase 上均等于 M430 use-PWP 数，mismatch 为 0；总数都是 15,909,646。
- 从 4,250,880 条 task sidecar 记录重建全部 648 个 DIRECT/stored/lazy 点，cycle、traffic、capacity 字段 mismatch 为 0；point JSON/CSV mismatch 为 0。
- 重建全部 432 个 same-K 比较，comparison JSON/CSV mismatch 为 0。

| K | EMPTY rows | DIRECT rows | PARENT rows | direct cycle sum | selected cycle sum |
|---:|---:|---:|---:|---:|---:|
| 1 | 24,534,432 | 11,395,922 | 15,909,646 | 92,640,472 | 53,965,135 |
| 2 | 24,534,432 | 20,346,837 | 6,958,731 | 54,393,136 | 42,835,311 |
| 4 | 24,534,432 | 25,434,808 | 1,870,760 | 37,063,882 | 34,805,548 |
| 8 | 24,534,432 | 27,035,741 | 269,827 | 32,424,687 | 32,154,860 |

## 240 KiB gate 与 same-K baseline

两个 gate 均按 `logical <= 245,760` 且 `macro-rounded <= 245,760` 重新判定，0 mismatch。

| block banks | direct 可行 tile | stored 可行 tile | lazy 可行 tile | best-budget direct tile |
|---:|---|---|---|---:|
| 4 | 32, 64, 96, 128, 192 | 32, 64, 96, 128 | 32, 64, 96, 128 | 192 |
| 8 | 32, 64 | 无 | 32, 64 | 64 |

120 个 gate-feasible candidate 中，最好一点是 banks=4、BW=infinite、K=1、stored、tile=128：778,503,248 cycles；同 K/banks/BW 的 best-budget direct 是 tile=192、866,912,128 cycles，speedup=1.1135626347，仍低于 1.15。因此独立 nomination 数为 0。

stored/lazy 的 cycle 与 traffic ledger 都逐字段复算。上述 stored 最好点的保守 DRAM traffic 是 11,899,553,280 B，其中 PWP 上界 6,986,127,360 B；同坐标 lazy 是 793,896,724 cycles、4,913,425,920 B DRAM 上界，并另计 14,681,397,504 B generator weight reads 与 6,287,514,624 B cache writes。

## 必须保留的边界

1. K 不可横比。K=1/2/4/8 分别需要 1/2/4/8 个 source banks/ports、96/192/384/768 个 product slots，以及 0/96/288/672 的 signed-preadder proxy。producer 的 nomination 只做 same-K direct/candidate 比较，这一点正确。
2. G15-specific task center mask 不存在。M430 mask/runs 是 G15 PARENT 集合的 superset，因此 stored/lazy 的 payload、command、generator traffic 只能称 conservative upper bound，不能称 exact G15 traffic。
3. route 总数虽然重算闭合，但输入只含 M469 已算好的 per-phase K columns；M471 没有逐行 original/center masks，独立 review 无法从原始逐行掩码重新生成每个 route choice。
4. M410/M40 无直接 contract input，静态 read-path 审计未发现直接读取；没有原运行 syscall trace，因此这是静态保证。
5. `docs359` 是 execution contract input，而 analyzer 会 SHA 每个 input，所以 producer 实际存在一次 docs359 只读路径；未发现写路径。若规则只是“不得修改 docs359”，producer 符合；若规则是“不得读取 docs359”，则这里是明确 scope exception。

最终处置：保留为 nonheadline sealed CPU-DSE sidecar，不修改 producer、不准入性能、不推进任何候选。若重启该方向，应先导出 G15-specific task center masks，并明确 docs359 是 zero-read 还是 no-write 规则。
