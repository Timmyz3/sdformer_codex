# M22 ordered compressed SRAM/DRAM transaction ledger

This milestone assigns deterministic logical addresses, read/write direction, byte extents, and serialized logical timestamps to the real ten-sample ordered traces.

| identity | variant | records | DRAM read B | DRAM write B | SRAM read B | SRAM write B | logical service cycles |
|---|---|---:|---:|---:|---:|---:|---:|
| h67_ep35 | local_line | 24988 | 66482650680 | 54283830000 | 2106891710722 | 183088690680 | 24482955983 |
| h67_ep35 | motion_selector_line | 26682 | 66482650680 | 54283830000 | 2100535119544 | 183088690680 | 24416741522 |
| local_ep44 | local_line | 26077 | 66079450680 | 54283830000 | 1969713950182 | 182683600680 | 23047701307 |
| local_ep44 | motion_selector_line | 27878 | 66079450680 | 54283830000 | 1962399565020 | 182683600680 | 22971509875 |

h67_ep35 Motion-selector transport delta: SRAM read -6356591178 B (-0.3017%); serialized byte-service -66214461 (-0.2705%). This is not a speedup.

local_ep44 Motion-selector transport delta: SRAM read -7314385162 B (-0.3713%); serialized byte-service -76191432 (-0.3306%). This is not a speedup.


The logical service cycles are a serialized byte-service envelope, not system cycles. The CSV still requires compressed-pattern burst expansion and DRAMsim3, SRAM bank/port calibration, validated liveness/fusion, and an RTL-exact attention memory schedule before any latency, energy, FPS, or speedup claim.
