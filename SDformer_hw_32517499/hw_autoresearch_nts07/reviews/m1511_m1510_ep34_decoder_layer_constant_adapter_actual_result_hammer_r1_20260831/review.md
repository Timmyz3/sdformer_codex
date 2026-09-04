# M1511 ep34 decoder layer-constant actual-result hammer

Verdict: **PASS** for the exact M1510 interpretation of the sealed ep34 decoder payloads. This is data-integrity evidence, not cycle or performance evidence.

The audit ran sequentially over all 120 retained decoder calls and 2,088,720,000 FP32 elements. Each layer has exactly 30 calls and one stable nonzero word: D0 `0x3f7ffd6b`, D1 `0x3f7fffa0`, and D2/D3 `0x3f800000`. Across the full population, negative words and nonfinite words are both zero. All 120 raw, compressed, and support payload identities are unique; shape, extent, padding, positive/negative support planes, ordered graph, and sample/module order passed.

M1510 tests reran 9/9 PASS, the source self-check passed, 43/43 independent controls passed, and 17/17 semantic/identity/graph mutations were rejected with zero false negatives. The evidence chain binds M1321/M1322/M1323/M1324, M1458, M1501, M1512, and M1513 seals and exact source identities.

Only continuation of the M1516 materializer release chain is authorized. No materialization has occurred, and cycles, traffic, speedup, energy, PPA, Table-A, and paper-result claims remain false.
