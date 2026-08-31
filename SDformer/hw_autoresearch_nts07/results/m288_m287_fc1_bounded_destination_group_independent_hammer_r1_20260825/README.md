# M287 FC1 bounded destination-group independent hammer

Verdict: **raw DSE arithmetic passes; the full-FC1 Amdahl promotion gate fails.**

The independent auditor imports neither the M287 analyzer nor its pickle helper.
It loads the ep35 model, rehashes and independently little-endian-decodes all 100
FC1 payloads, repeats per-output-row INT8 quantization, and recomputes all 45
aggregate and 450 per-module DSE cells.

The checkpoint/payload work is reproducible: 12 FC1 weights exist, ten binary
modules have ten payloads each, the two stage3 FC1 modules are excluded, and the
100 payloads contain 112,213,979 active source events. Producer mismatches are
zero for raw group-task arithmetic.

The blocking issue is the Amdahl denominator. M287 accelerates only ten binary
modules but scales all 118,370,114 frozen FC1 cycles. The frozen operator ledger
splits those cycles into 100,895,624 eligible cycles and 17,474,490 excluded
stage3 cycles. After module-cycle weighting, group4/beta80 is 1.145605x rather
than 1.175173x and does not cross the 1.15 gate. The first corrected crossings
are group4/beta96 at 1.171062x and group8/beta96 at 1.151312x; both remain highly
aggressive, accuracy-free opportunities.

The 4.569x task ratio is not an executable cycle result. Compaction, routing,
bank conflicts, accumulator service, accuracy, RTL, physical SRAM, DC and energy
remain unmodeled. The beta bound is a conservative quantized INT8 accumulator
bound only.

Primary evidence is `independent_recompute_r4.json`; r1/r2/r3 are superseded
audit iterations and are not publication evidence. The complete 5-by-9
scope-corrected sensitivity table is under
`amdahl.module_cycle_weighted_scope_grid`.

For beta96, group4 removes 91.999% of static source/group pairs and 89.787% of
activity-weighted tasks, so it is NO-GO as the primary candidate. Group8 removes
84.654%/80.768% and reaches only 1.151312x; retain it for paired S10 only and
fail closed before valid825/RTL if absolute AEE rises by more than 0.02. Across
the ten modules, the beta96 integer-accumulator bound ranges from 4,704 to
19,872 (median 15,744); it is not a float-output or accuracy bound.
