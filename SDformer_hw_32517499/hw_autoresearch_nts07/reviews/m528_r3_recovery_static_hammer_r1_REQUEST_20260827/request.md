# M528 r3 recovery source-only static hammer request

请对 author handoff `reviews/m528_single_port_same_ledger_recompute_author_handoff_r3_20260827` 做独立、只读静态锤审。禁止运行 smoke 或 production analyzer/runner；本评审唯一可能放行的下一步，是由 root 新签一次 preflight-only schema-smoke admission，不能放行 production。

重点逐项核对：真实 JSON pointer/corner/9 宏面积；所有 live key path；smoke 在加载旧 analyzer 和 process pool 前返回；wrong-pointer/wrong-corner 负测；旧 analyzer 的 worker/cycle/traffic/capacity/aggregation/gate 语义保持 byte-frozen；production attempt 必须晚于动态资源门和现场 smoke；r2 证据永久保留且 r2 admission 不可复用。

通过门为 P0=0、P1=0。请输出双封 `reviews/m528_r3_recovery_static_hammer_r1_20260827`，明确列出 analyzer、两个 runner、execution contract、author outer-seal file 和 `docs/359` 的 SHA，并给 root 下一份 smoke-only admission 所需字段模板。
