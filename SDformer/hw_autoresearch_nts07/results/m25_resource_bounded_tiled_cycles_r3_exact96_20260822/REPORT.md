# M25 resource-bounded tiling + cycle architecture

Status: **headline >2x NO-GO**. Legal tiling is proven, but current >2x compute exists only at L16/160 ATLIF multipliers and is not a same-96-MAC comparison.

| identity | SRAM | fixed resident | cohort state | tiles | legal |
|---|---:|---:|---:|---:|---|
| H67 | 96 KiB | 29568 B | 209088 B | 4 | LEGAL_BARRIER_PRESERVING_SPILL_REPLAY |
| H67 | 128 KiB | 29568 B | 209088 B | 3 | LEGAL_BARRIER_PRESERVING_SPILL_REPLAY |
| H67 | 240 KiB | 29568 B | 209088 B | 1 | LEGAL_BARRIER_PRESERVING_SPILL_REPLAY |
| H67 | 408 KiB | 29568 B | 209088 B | 1 | LEGAL_BARRIER_PRESERVING_SPILL_REPLAY |
| Local5 | 96 KiB | 29568 B | 418176 B | 7 | LEGAL_BARRIER_PRESERVING_SPILL_REPLAY |
| Local5 | 128 KiB | 29568 B | 418176 B | 5 | LEGAL_BARRIER_PRESERVING_SPILL_REPLAY |
| Local5 | 240 KiB | 29568 B | 418176 B | 2 | LEGAL_BARRIER_PRESERVING_SPILL_REPLAY |
| Local5 | 408 KiB | 29568 B | 418176 B | 2 | LEGAL_BARRIER_PRESERVING_SPILL_REPLAY |

| line | lanes | multipliers | compute cycles | vs Fixed compute | evidence |
|---|---:|---:|---:|---:|---|
| local | 8 | 80 | 373341693 | 1.663003x | DSE_ONLY_REQUIRES_RTL_VCS_DC |
| local | 10 | 100 | 342616773 | 1.812136x | DSE_ONLY_REQUIRES_RTL_VCS_DC |
| local | 16 | 160 | 296529393 | 2.093783x | L16_SERVICE_MODEL_PLUS_EXISTING_L16_CHECKPOINT_VCS_FUNCTIONAL_PROOF |
| local | flat96 | 96 | 347737593 | 1.785450x | ARITHMETIC LOWER BOUND; NOT EXECUTABLE |
| hybrid | 8 | 80 | 371509166 | 1.671206x | DSE_ONLY_REQUIRES_RTL_VCS_DC |
| hybrid | 10 | 100 | 340784246 | 1.821881x | DSE_ONLY_REQUIRES_RTL_VCS_DC |
| hybrid | 16 | 160 | 294696866 | 2.106803x | L16_SERVICE_MODEL_PLUS_EXISTING_L16_CHECKPOINT_VCS_FUNCTIONAL_PROOF |
| hybrid | flat96 | 96 | 345905066 | 1.794909x | ARITHMETIC LOWER BOUND; NOT EXECUTABLE |

M21 is charged as the implemented one-slice/FIFO4 architecture with 738 registered-result bubbles (123 lane tiles x 6 cycles). Three-slice/FIFO40 remains DSE-only. M23 ticks are not cycles. Local5 full-system speedup is UNKNOWN because attention is missing nonzero.

The next RTL point is an exactly-96 resource-shared ATLIF/M4 lane cluster with a bit-exact cohort-skip certificate, plus a barrier-indexed tile replay controller and one-entry M21 result snapshot queue. Lane sharing alone is insufficient because even the non-executable flat96 arithmetic lower bound misses 2x. It must be admitted by VCS, area-constrained DC A/B, and address-timed memory simulation.
