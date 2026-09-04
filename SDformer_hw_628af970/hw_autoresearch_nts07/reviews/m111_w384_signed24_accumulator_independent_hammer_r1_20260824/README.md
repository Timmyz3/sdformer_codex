# M111 W384 signed24 accumulator：独立打铁复审

日期：2026-08-24

结论：**91/100，P0=0 / P1=5 / P2=5。M111 的 standalone 96-lane signed24 accumulator、lazy-valid、8×384 commit 和 1R1W command protocol 可准入为 commercial-VCS directed functional；M109-r2 的 2.535462× 仍只是 precompacted same-clock software projection。**

## 独立 VCS 结果

- 工具：Synopsys VCS `V-2023.12-SP1`，编译/仿真 RC 均为 0，无 warning/error/assertion failure signature。
- 两个完整窗口：`6,144` 个 commit vector、`589,824` 个 signed lane comparison 全部一致。
- 正向流：`169` 个 update 对应 `169` 个精确 macro write；`167` 个 non-same-address II1 pair，`167` 个同步读写重叠周期。
- commit：严格 block-major/row-major；`1,604` 个 positive stall cycle 下数据和 sideband 稳定；每个 valid row 恰好读一次，invalid row 不发读命令。
- 数值边界：96 lane 同时覆盖 signed24 max/min、正负调整和重复 RMW。
- fail-close：同址连续攻击 1 次，第二请求被拒绝且旧写保留；正溢出和负溢出各 1 次，均 suppress write 并 sticky 到 reset。
- lazy clear：第二窗没有清 SRAM data，旧 `(block=0,row=7)` 内容仍在，但 commit 为零且不发该 row 的 SRAM read。
- 独立 SVA cover：II1 `171`、read/write overlap `171`、stall-release `1,424`、full complete `2`、fault `3`。

## 存储审计

`8 × 384 × 96 × 24 / 8 = 884,736 B`，M111 的 accumulator data 数字正确；lazy valid 是 `8 × 384 = 3,072 bit = 384 B`。

需要修正未来合并表的口径：M109-r2 的 `909,352 B` 明确排除了 valid/epoch tags。把 M111 实际存在的 384 B valid bits 加进去后，descriptor + minimum metadata + accumulator + valid 的逻辑小计至少是 **909,736 B**，仍不含 controller/grace、ECC、macro rounding 和外围。

## 仍未关闭的 P1

1. M110 controller 与 M111 accumulator 没有接成一个 commercial-VCS actual-record cycle miter；2.535462× 没有因此变成 RTL-measured。
2. 八个 SRAM 都是 behavioral sync 1R1W；没有 foundry macro、编译配置、RDW mode、depth/width rounding、ECC 或能耗。
3. 96 个 25-bit adder 和 2304-bit 读写/commit bus 尚无本里程碑 exact-SHA DC/STA/PTPX；同频物理可行性未知。
4. 连续同址会 reset-only quarantine；安全性已证明，但全 workload 的 liveness 依赖外部 M108 spacing，尚无 heldout integrated replay。
5. 没有同库、同频、等面积且包含 macro/controller/delivery/commit 的物理 baseline。

## P2 硬化项

- 将独立补到的 negative overflow 纳入 production regression。
- 将 accepted-update 精确读命令、buffered write data、valid-row unique read、invalid-row no-read 提升为 production monitor/SVA。
- 下游所有 commit sideband 必须由 `commit_valid` 限定；idle 时不是零值合同。
- 增加 constrained-random 与 formal safety，覆盖更多 reset/backpressure/address/value history。

## Claim boundary

- standalone numeric/storage/protocol directed functional：**GO**。
- M110 W384 controller geometry VCS + M111 standalone accumulator VCS：**GO，两个独立模块**。
- M109-r2 `2.53546204172554×`：**GO，仅 software projection**。
- integrated actual-record RTL cycles、foundry macro、macro-inclusive PPA、equal-area/energy、physical/system/full-network/headline speedup：**NO-GO**。

机器可读结论见 `m111_w384_signed24_accumulator_independent_hammer_review.json`。本评审只写本目录，未修改 production、contracts/results 或 `docs/359`；其 SHA 仍为 `dedde7ce...`。
