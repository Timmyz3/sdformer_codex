# M158 r2 独立评审修正 overlay

M158 r2 的 signed-event 重构和冻结权重 `sum(abs)` 数字保留，但独立评审发现其“每个 feature 至多出现一次”的文字前提不符合 PWP 实际事务语义。

对于 `center=1,target=0`，同一 feature 合法地交付一次 `+1` anchor 和一次 `-1` correction。Acc19 的真实安全条件是：每个 raw-Conv output/feature 的 accepted prefix coefficient 始终位于 `[-1,1]`，并且每个必要 anchor/correction exact-once。当前集成 RTL 尚未证明 stall/reset/stale/duplicate/cache-alias 下的这个条件。

因此立即撤销以下接纳：

- 集成 source-major reorder exactness；
- 冻结域不需要 runtime overflow detector；
- 从 RTL 删除 overflow tree。

在集成 VCS/SVA exact-once miter 完成前保留动态 guard，或采用 signed20 fail-closed fallback。M158 r2 只保留为条件式 Acc19 数学候选，不是 RTL/PPA/speedup 证据。
