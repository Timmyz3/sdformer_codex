# M157 r2 independent hammer review

Verdict: **84/100; P0/P1/P2 = 0/3/3.** The frozen 20-record cache/work census and zero adjacent one-cycle RMW-hazard result are valid. M157 remains a disciplined DSE, not admitted hardware speedup.

Fresh execution of the frozen analyzer completed successfully and produced a byte-identical JSON. A separate checker that does not import M157 rebuilt all event masks from the pre-M157 lineage. It verified sample IDs 5..9 crossed with operators 0..3, all 20 packed payloads, four weight payloads, 47,040,777 descriptors, 188,148,490 events and 23,522,595 source keys. Reconstructing the scheduled tuple masks produced the same SHA256, with zero per-key descriptor mismatch against M152.

The heldout zero-hazard result is exact: all 46,971,957 context-internal adjacent descriptor pairs were checked. Its real reason is stronger and narrower than the prose suggests: every one of the 1,033,912 active source-contexts contains both destination halves and all eight destinations, so every cross-source boundary changes from half1 to half0 and therefore changes accumulator address bit2. The order is not universally hazard-free. Two consecutive one-row half0 phases at the same row and bank require a bubble even under the optimizer.

The 22.747159575x value is `188,148,490 / 8,271,296`: an uncached-to-cached vector-group read-work ratio, not cycles. It is also not an incremental improvement over M152, whose recurrence already charges `24,813,888 = 3 * 8,271,296` weight-load tokens.

The ping-pong 1.803703226x and nonoverlap 1.756938227x sensitivities are two alternatives, so they are not added to each other. Their arithmetic is exact. Their architectural accounting is not closed: both start from a baseline that already contains the cached payload-load tokens, while no port-level recurrence defines whether the added context/phase cycles are extra launch work, replacement service or hidden prefetch. Load-to-use, short-phase overlap, macro contention and backpressure remain unpriced.

Integer addition is reorderable only while arithmetic is exact and fault behavior is order-independent. The heldout correction-only population has a conservative bound of `1208 * 128 = 154,624`, inside signed19. That does not include PWP/base accumulator state or prove every source-major prefix. The architectural `432 * 16 * 128 = 884,736` bound exceeds signed19, so the runtime overflow guard cannot be removed without signed checkpoint replay.

Required next gate: an exact signed baseline-versus-source-major prefix miter, followed by a minimal fused cache-to-accumulator RTL with explicit identity/context/fault handling and a cycle recurrence tied to selected cache and accumulator SRAM ports. Only then should the 1.80x candidate be reconsidered.

`docs/359` remains unchanged at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
