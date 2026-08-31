# M1110D independent hammer of M1105Dr2 decoder source/contract/receipt

结论：**GO，但仅授权另一作者编写 runner；不授权 launch 或 production。**

本次以固定 source SHA `b2d8ef...a5c4`、canonical contract 的 file/sidecar/outer 三重身份、sealed author receipt，以及 M1106D STOP outer 为信任根。独立复核了 contract 的 136 个 leaf、120 个冻结 decoder call、总计 261,090,000 B bitpack 与 30 个 D1 exact scaled-binary miter，全部一致，D1 mismatch 为 0。

旧 M1106D 的 P0 已在 r2 被真正关闭：source 只接受零参数，从自身路径导出 repo/HW/payload，caller 不能选择 repo、contract 或 output，环境变量也不能改写身份。22 个 resource/address/dependency/time/D1/checkpoint/rebind/M700/claim 变异和 8 个 source-contract-receipt bytes/symlink/path-traversal 攻击均被 fail-closed 拒绝。

授权边界保持严格：本收据只证明 source/identity/address/timing schema 的执行信任根。它没有生成 runner，没有枚举 production transaction，没有给出 cycle、traffic、speedup、energy 或 PPA。下一步必须由不同作者编写 runner，再由另一作者做 runner 与 launch hammer，之后才可申请一次 production replay。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
