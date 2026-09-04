# M946 author preflight（非独立 hammer）

结论：source candidate 的三层 selector、synthetic/real 1K exact miter、fail-closed CLI 和 static identity checker 均通过。D1 100K bounded prefix 也通过 2× memory/timeout projection，但 `full_row_authorized=false`，不能由此启动 full row。

Fresh hammer 预审发现默认 `/usr/bin/python3` 为 3.6.8，而作者测试使用 M925 同源 Python 3.10.18。候选已补 exact path/version/SHA pin；默认解释器现应在加载 M896 前明确拒绝。最终结论仍需更新身份后的独立重打。

D1 始终是 `COMMON_CHARGED_FULL_SHAPE_DIAGNOSTIC_NONHEADLINE`；D2/D3 是 `EXACT_BINARY_SUPPORT`。1K 前缀只覆盖首个 source-fetch transaction，因此空 commit hash 是预期现象，不代表 commit coverage。D2/D3 10K/100K 未运行，避免在 source-author 阶段物化大 contributor 图。

本目录只是 author preflight，不是 fresh independent source hammer，不给分，也不允许论文引用、Table-A、decoder-complete 或 system-speedup 声明。
