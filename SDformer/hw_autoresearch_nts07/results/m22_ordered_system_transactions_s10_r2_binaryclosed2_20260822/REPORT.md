# M22 ordered compressed SRAM/DRAM transaction ledger

This milestone assigns deterministic logical addresses, trace order, read/write direction, byte extents, and a separate serialized byte-service estimate to manifest-frozen ten-sample traces.

| identity | attention coverage | variant | records | DRAM read B | DRAM write B | SRAM read B | SRAM write B | serialized service ticks |
|---|---|---|---:|---:|---:|---:|---:|---:|
| h67_ep35 | ABSTRACT_PACKED1_COUNTER_SUMMARY_NOT_PHYSICAL_TRAFFIC | local_line | 24988 | 66482650680 | 54283830000 | 2106891710722 | 183088690680 | 24482955983 |
| h67_ep35 | ABSTRACT_PACKED1_COUNTER_SUMMARY_NOT_PHYSICAL_TRAFFIC | motion_selector_shared_state | 35052 | 66482650680 | 54283830000 | 2101723787884 | 183095407020 | 24429195362 |
| h67_ep35 | ABSTRACT_PACKED1_COUNTER_SUMMARY_NOT_PHYSICAL_TRAFFIC | motion_selector_explicit_copy | 40632 | 66482650680 | 54283830000 | 2101723787884 | 213860719020 | 24749667362 |
| local_ep44 | MISSING_FROM_EXECUTION_TRACE_NOT_ZERO_COST | local_line | 26077 | 66079450680 | 54283830000 | 1969713950182 | 182683600680 | 23047701307 |
| local_ep44 | MISSING_FROM_EXECUTION_TRACE_NOT_ZERO_COST | motion_selector_shared_state | 37598 | 66079450680 | 54283830000 | 1963624761120 | 182690556780 | 22984347115 |
| local_ep44 | MISSING_FROM_EXECUTION_TRACE_NOT_ZERO_COST | motion_selector_explicit_copy | 44078 | 66079450680 | 54283830000 | 1963624761120 | 214253340780 | 23313126115 |

h67_ep35 motion_selector_shared_state transport delta: SRAM read -5167922838 B, SRAM write +6716340 B; serialized byte-service -53760621 (-0.2196%). This is not a speedup.

h67_ep35 motion_selector_explicit_copy transport delta: SRAM read -5167922838 B, SRAM write +30772028340 B; serialized byte-service +266711379 (+1.0894%). This is not a speedup.

local_ep44 motion_selector_shared_state transport delta: SRAM read -6089189062 B, SRAM write +6956100 B; serialized byte-service -63354192 (-0.2749%). This is not a speedup.

local_ep44 motion_selector_explicit_copy transport delta: SRAM read -6089189062 B, SRAM write +31569740100 B; serialized byte-service +265424808 (+1.1516%). This is not a speedup.


The service ticks are a serialized byte-service estimate, not request arrival or system cycles. The CSV still requires validated dependencies, physical allocation, burst expansion, DRAMsim3, SRAM bank/port calibration, selector compute/control, liveness/fusion, and an RTL-exact attention schedule before any latency, energy, FPS, or speedup claim.
