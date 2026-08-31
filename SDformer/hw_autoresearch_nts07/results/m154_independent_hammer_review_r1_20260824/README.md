# M154 independent hammer review

Verdict: **89/100; P0/P1/P2 = 0/2/4.** M154 closes the M150/M152 single-vector identity P0 at the standalone supplier interface. It does not admit the 1.805357581x cycle ratio.

Independent VCS exercised 10,112 exhaustive legal descriptors across all 16 sources, all four banks and all 32 addresses per bank, with every tuple position visiting every bank. It also covered 1,873 result stalls, 9,006 II=1 pairs, all 12 invalid masks, all 16 same-bank destination pairs, isolated sequence/operator/partition mismatches, API collisions and reset cancellation. With both elastic slots occupied, a younger same-bank fault preserved and drained both the old held result and the old pending request. Production SVA attached to this corpus reported no failure.

The 98,304-bit number is only the required external capacity: `4 banks * 32 vectors * 768 bits`. RTL and DC contain **zero SRAM macros**. Fresh exact DC reproduced 13,282.668059 um2, 15,075 cells, 3,273 sequential cells, 10 logic levels, +1.6514 ns setup and +0.0002 ns hold at 3 ns. Those numbers include the 3,072-bit result register but exclude resident SRAM, loading and accumulator macros.

The two-entry pipeline also depends on external bank outputs holding their last response while read enable is low. The frozen behavioral model does so. A negative no-hold sensitivity corrupts the pending second response at 52.5 ns, so an exact physical macro/clock-enable contract must be bound.

The 3,072 payload FFs are 93.86% of all sequential cells and occupy 6,193.152 um2, or 46.63% of total standard-cell area. A macro-output token can remove them while retaining ready-case II=1, but reduces stall-time capacity from two transactions to one unless responses are reread or stored. The preferred next architecture is a fused four-bank accumulator: consume the four macro outputs directly, add signed negate/Acc19 RMW and forwarding, and eliminate the wide result buffer while closing the missing accumulator P1. No area recovery is admitted until matched VCS/DC and macro-aware recurrence exist.

## M155 interface impact

There is no new M154 P0 blocking M155: the four genuinely independent destination vectors repair the M150 single-vector identity P0, and the prefix-mask, modulo-four conflict, row, destination and negate shapes align. Two integration P1s remain. First, a macro-output-token revision must bind output-holding SRAM semantics before M155 backpressure can be allowed to retain a live response. Second, M154 signed INT8 vectors require explicit bit-exact sign extension to M155 signed11, and M154 sequence/operator/partition context open/close must be atomically bound to M155's identity-less window start/end and fault drain.

At this review snapshot, M155 has RTL/TB/SVA sources but no contract or sealed `RUN_COMPLETE.txt`, so it is not admitted here. External weight SRAM, checkpoint loading, selected macro timing/energy, accumulator SRAM, physical RMW/commit, source-key load-to-use recurrence and end-to-end signed checkpoint replay remain open. Consequently there is no interface P0, but the two P1 integration boundaries can invalidate M155 evidence if they are left implicit.

`docs/359` remains unchanged at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
