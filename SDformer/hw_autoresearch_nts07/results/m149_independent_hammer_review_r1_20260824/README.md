# M149 independent hammer review

Verdict: **84/100, conditional pass for the narrow arithmetic island; protocol repair required before integration.** P0/P1/P2 = **0/2/4**.

The signed destination fold itself is strong. Independent VCS checked 10,956 descriptors, every legal destination assignment, all stable-first-owner masks, all signed-8 values under all negate masks, and 4,222,080 lane/group values with zero arithmetic mismatch. Fresh production VCS and fresh 3 ns DC also reproduce the sealed results.

The blocking issue is compositional. If a legal result is held under backpressure and an illegal input shape arrives, the RTL latches fault and deletes the unconsumed result. The independent attack reproduces one dropped transaction; attaching the frozen production SVA triggers `ap_result_payload_stable_under_stall` at 33,253.5 ns. Repair input quarantine so it cannot erase an already accepted output.

M149 does **not** admit M147's 1.805434x ratio. It assumes 3,072 contribution input bits are already available and does not implement SRAM/PWP delivery, M148 integration, or the mask-aware accumulator commit needed to retire as many as 4,224 result-vector bits per cycle. Gapped valid masks (`0101`, `1001`, `1011`, `1101`) are correct and must be honored by the eventual commit consumer.

Fresh DC reproduced 30,958.703937 um2, 36,772 cells, 4,284 sequential cells, 24 logic levels, 1.45 ns critical-path length, +1.0787 ns setup slack and +0.0221 ns hold slack at 3.000 ns. This remains flattened, ideal-clock, ZeroWireload, zero-macro logic-only evidence.

`docs/359` remains unchanged at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
