# M25 resource-bounded tiling + cycle architecture

Status: **headline >2x NO-GO**. Exact row-aligned tiling is proven for frozen non-attention C4 cohorts, but current >2x compute exists only at L16/160 ATLIF multipliers and is not a same-96-MAC comparison.

| identity | SRAM | fixed resident | cohort state | tiles | legal |
|---|---:|---:|---:|---:|---|
| H67 | 96 KiB | 51968 B | 209088 B | 6 | LEGAL_FOR_FROZEN_C4_COHORTS_ABSTRACT_ATTENTION_PHYSICAL_CAPACITY_UNKNOWN |
| H67 | 128 KiB | 51968 B | 209088 B | 3 | LEGAL_FOR_FROZEN_C4_COHORTS_ABSTRACT_ATTENTION_PHYSICAL_CAPACITY_UNKNOWN |
| H67 | 240 KiB | 51968 B | 209088 B | 2 | LEGAL_FOR_FROZEN_C4_COHORTS_ABSTRACT_ATTENTION_PHYSICAL_CAPACITY_UNKNOWN |
| H67 | 408 KiB | 51968 B | 209088 B | 1 | LEGAL_FOR_FROZEN_C4_COHORTS_ABSTRACT_ATTENTION_PHYSICAL_CAPACITY_UNKNOWN |
| Local5 | 96 KiB | 51968 B | 418176 B | 11 | LEGAL_FOR_FROZEN_NON_ATTENTION_C4_COHORTS_FULL_SYSTEM_CAPACITY_UNKNOWN |
| Local5 | 128 KiB | 51968 B | 418176 B | 6 | LEGAL_FOR_FROZEN_NON_ATTENTION_C4_COHORTS_FULL_SYSTEM_CAPACITY_UNKNOWN |
| Local5 | 240 KiB | 51968 B | 418176 B | 3 | LEGAL_FOR_FROZEN_NON_ATTENTION_C4_COHORTS_FULL_SYSTEM_CAPACITY_UNKNOWN |
| Local5 | 408 KiB | 51968 B | 418176 B | 2 | LEGAL_FOR_FROZEN_NON_ATTENTION_C4_COHORTS_FULL_SYSTEM_CAPACITY_UNKNOWN |

| line | lanes | multipliers | compute cycles | vs Fixed compute | evidence |
|---|---:|---:|---:|---:|---|
| local | 8 | 80 | 379440224 | 1.636274x | DSE_ONLY_REQUIRES_RTL_VCS_DC |
| local | 10 | 100 | 348715304 | 1.780444x | DSE_ONLY_REQUIRES_RTL_VCS_DC |
| local | 16 | 160 | 302627924 | 2.051589x | L16_SERVICE_MODEL_PLUS_EXISTING_L16_CHECKPOINT_VCS_FUNCTIONAL_PROOF |
| local | flat96 | 96 | 353836124 | 1.754677x | ARITHMETIC LOWER BOUND; NOT EXECUTABLE |
| hybrid | 8 | 80 | 377769950 | 1.643509x | DSE_ONLY_REQUIRES_RTL_VCS_DC |
| hybrid | 10 | 100 | 347045030 | 1.789013x | DSE_ONLY_REQUIRES_RTL_VCS_DC |
| hybrid | 16 | 160 | 300957650 | 2.062975x | L16_SERVICE_MODEL_PLUS_EXISTING_L16_CHECKPOINT_VCS_FUNCTIONAL_PROOF |
| hybrid | flat96 | 96 | 352165850 | 1.763000x | ARITHMETIC LOWER BOUND; NOT EXECUTABLE |

M21 is charged as the implemented one-slice/FIFO4 architecture with Local/Hybrid phase-1 increments 6098531/6260784 cycles plus 738 registered-result bubbles (123 lane tiles x 6 cycles, bound per operator). Three-slice/FIFO40 remains DSE-only. M23 ticks are not cycles. Local5 full-system capacity and speedup are UNKNOWN because attention is missing nonzero.

The next RTL point is an exactly-96 resource-shared ATLIF/M4 lane cluster with a bit-exact cohort-skip certificate, plus a barrier-indexed tile replay controller and one-entry M21 result snapshot queue. Lane sharing alone is insufficient because even the non-executable flat96 arithmetic lower bound misses 2x. It must be admitted by VCS, area-constrained DC A/B, and address-timed memory simulation.
