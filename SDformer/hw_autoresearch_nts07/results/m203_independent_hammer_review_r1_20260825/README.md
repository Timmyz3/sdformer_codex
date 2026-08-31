# M203 独立打铁评审 r1

结论：**92/100，`PASS_EXACT_FROZEN_PAYLOAD_ANALYTIC_REPLAY__P0_COMPOSITION_GATE`**。

M203 的数字是对的，而且这次确实修正了 M202 与 M199 recurrence 不同的问题。我没有导入
生产 analyzer：独立用 `numpy.unpackbits(bitorder=little)` 解 120 份冻结 H67 FC2 payload，
用 list queue 逐拍执行 aligned raw4、queue8、empty-queue fresh bypass、queue 非空不与 fresh
co-emit 的 M202 状态机，再用独立双 buffer event ledger 重算 wall。

全量独立重算精确复现：

- 5,580,000 tokens、143,894,510 events、36,480,000 raw96 beats、
  18,869,376 nonzero descriptors、6,523,707 compact windows；
- M202 stage-aware `90,107,277` cycles；基线 `114,645,510`，比值
  `1.272322434x`；
- M199 stage-aware `90,112,890`，所以 M202 仅快 `0.006229242%`；
- W1 `94,761,587`、pair `90,216,831`，pair/W1 `1.050375922x`；
- equal/faster/slower token 分别为 `5,387,736 / 189,567 / 2,697`；
- always-ready recurrence 下 queue maximum 为 `7`，与八项 RTL queue 容量相容。

另对 depth 2/4/8、1--8 raw beats 的 1,530 个小 bitmap 做了独立穷举，保留了
`110`（M202 快）和 `01111`（M202 慢）的逐拍 witness。finite-wall 又以另一份逐 tick
simulator 对 W1/pair、1--4 windows、不同 interval/trailing/group/output-block 的 75,492
个组合交叉验证，0 mismatch。结论是 interval 与 trailing 精确分割 source service，buffer
wait 只插一次，drain overlap 没有被双算。

## P0 裁决

数值层面没有发现新的 P0 错账；P0 是组合准入门：当前 M184 每拍只收一个 descriptor，且
一次只 drain 一个 window。它既没有 M203 假设的 four-wide sink，也没有 paired-window
bank-union drain。因此 `90,107,277` 和 `1.272322434x` 只能是 exact-payload analytic
schedule，不得叫 RTL measured、physical、complete FC2、FFN、system 或 headline speedup。
生产 contract 已正确保留这些禁止项。

finite-wall 自身没有双算，但它不是 ready/valid 组合证明：它把 buffer wait 插入
always-ready M202 close intervals，而没有在 sink stall 时继续演算 M202 queue 的预取与
backpressure。always-ready 下的 queue max 7 也不能代替组合 stall 下的 occupancy/cycle
测量。这个差异必须由 M204 组合 VCS 收口。

## M204 实现建议

M204 应直接把 M184 ingress 改成原子 1--4 descriptor 输入，保留两个 window buffer，四路
写入 bitmap/beat-index 并并行更新八个 bank popcount。M202 已保证一个 packet 不跨 compact
window；M204 仍须检查 last marker、remaining capacity，并覆盖同拍 release/refill。

drain 端对 stage1--3 把相邻两个 closed window 当成一个 fixed-bank union：每 bank 每拍取两
窗联合中的最早 source，最多形成八 bank structural group；同一 group 重放所有 output
blocks，最后一个 block 才清源。stage0 必须关闭 pair，因冻结账本中 stage0 pair 比 W1 更慢。
不要重新引入 top-K、bank ID 或 lane crossbar。为守 3 ns，应采用“每窗各自 priority + 两路
选择”的分层结构，并在 request 边界前流水，而不是平铺 16-descriptor 搜索。

准入回归必须把 M202 与 M204 直接相连，在 VCS 中覆盖 queue=8、partial-final、full pair、
odd tail、stage0 no-pair、sink stall、同拍 release/refill 和 sticky attacks，并报告真实
header-to-header cycles。随后接 M186/M185，用相同 weight-response latency/backpressure
跑 flat DC；M186 已知的跨 reset stale-response alias 也必须修复或由外部 reset/flush 合同
关闭。SRAM macro、context store、BN2/residual 仍不在本里程碑内。

机器可读裁决与完整重算见 `m203_independent_hammer_review_r1.json` 和
`independent_replay.json`。`docs/359` 未修改，SHA-256 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
