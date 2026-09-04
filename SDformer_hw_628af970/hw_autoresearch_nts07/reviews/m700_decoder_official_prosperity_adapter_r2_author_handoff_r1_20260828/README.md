# M700 decoder 官方 Prosperity adapter r2 作者交付

## 结论

M700 是 additive r2 identity；M693、M697 和 `docs/359` 保持只读。r2 只修复 M697 指出的报告与失败协议，不执行 production mapper 或官方模拟器。

作者静态测试 `32/32` 通过。当前仍为 `execution_authorized=false`；必须由新的独立代理给出精确状态 `GO_M700_FULL_OFFICIAL_CPU_REPLAY__P0_0_P1_0` 后才能执行。

## M697 两个 P1 的闭合

1. `aggregate_breakdowns` 现在为 exact D0/D2/D3 与 D1 diagnostic 两个 population 分别生成 `phase:3/2/1/0`。每个 phase 都是相同调用集上的整数 counter ratio-of-sums；runner 强制验证所有 bit/product、DRAM、global-buffer、support-NNZ、mapped support accounting 和调用数从四 phase 合计到 overall。
2. fresh run-state failure guard 覆盖 authorization、preflight、两批 `execute_records`、post-execution identity recheck、atomic publication 和 post-verify。任何异常产生 noncanonical、non-overwrite、双封 failure receipt。单元测试注入了 `execute_records` 异常并复验双封。

## P2 收口

- worker 数固定为 3；runner 拒绝其它值，report 与 receipt 均记录该值；
- publication 持有 `O_EXCL` single-writer lock，绑定设备号/inode并在 rename 前复查；
- canonical leaf 用 `lexists` 检查，因此 dangling symlink 不能绕过 non-overwrite；
- 输出仍为 fresh staging、双封、rename、post-publish verify，失败后 quarantine。

## 冻结 identity

- runner：`a5e7113b3c56354bbcbd8196837ab444ed1830ab66c55f0d3610dd78cf713098`
- contract：`c340a167cc3641a468327697b57d43197ddb6699d3ca744d0a9d9f7f26c1bb65`
- tests：`0f587d0ca691ff0d5f53eeaefd56f6f9e8069d16d05931b633e9561ee63093ba`

## Claim boundary

本交付没有运行完整 M672 workload，没有导入/调用官方 `Simulator.run_fc`，没有产生 canonical result、周期、倍率、GPU 或 EDA 证据。D0/D2/D3 仍是 exact external official subset；D1 仍是 diagnostic-only；exact decoder-complete 周期/倍率仍为 null；`ours/full_decoder/system/headline=false`。
