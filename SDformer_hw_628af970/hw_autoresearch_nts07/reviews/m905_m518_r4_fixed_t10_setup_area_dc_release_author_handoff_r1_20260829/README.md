# M905｜M518 r4 Fixed-T10 单点 DC release author handoff

本目录只封装一次 inert release 的 author-side 静态证据。它将既有 r4 per-point
runner 严格收窄到 `M518_R4_POINT=fixed` 和冻结的 Fixed admission；不授权第二个
C3 点、paired comparison、VCS、PT、Formality、PTPX、远端或任何论文指标。

作者仅运行双 Python 静态检查和内存内 mutation attack，没有调用 runner
production path、Synopsys 工具或 license，也没有创建 result、attempt、work 或
quarantine。即使 release 的 `launch_now=true`，也必须先由 fresh independent
M906 final hammer 达到 100/100，且当前 C2 one-shot 完整终止、共享主机 collision
与资源门重新通过，root 才能考虑一次 exact 命令。

成功的 DC 也只产生 Fixed logic-only、3 ns、ideal-clock、ZeroWireload 的 raw
setup/area 证据。Hold 只保留 `not_closed_at_dc` 诊断边界；power、energy、PPA、
系统倍速与 headline 均为 false，结果还必须另做 independent point-result hammer。
