# M23 trace-live physical memory and bank/port envelope

M23 replaces M22's logical SRAM address span with deterministic trace-live reuse, then maps every M22 transaction to a compressed fixed-quantum bank/port service record. These figures are not DRAMsim3 or system speedup.

| identity | variant | M22 logical SRAM B | allocator B | peak live B | two-copy largest-stream bound B | allocator + extra stream B | DRAM bursts | SRAM words | M22 transport ticks | serialized port ticks lower..upper | conflict stalls lower..upper |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h67_ep35 | local_line | 132774408768 | 595968864 | 589824480 | 1179648000 | 1185792864 | 1886976570 | 572495100498 | 24482955983 | 24482955983..24482955983 | 0..0 |
| h67_ep35 | motion_selector_explicit_copy | 136193532608 | 595968864 | 589824480 | 1179648000 | 1185792864 | 1886976570 | 578896128224 | 24749667362 | 24749667362..25214223241 | 0..464555879 |
| h67_ep35 | motion_selector_shared_state | 132775164608 | 595968864 | 589824480 | 1179648000 | 1185792864 | 1886976570 | 571204800224 | 24429195362 | 24429195362..24893751241 | 0..464555879 |
| local_ep44 | local_line | 132365529408 | 595968864 | 589824480 | 1179648000 | 1185792864 | 1880676570 | 538099387785 | 23047701307 | 23047701307..23047701307 | 0..0 |
| local_ep44 | motion_selector_explicit_copy | 135873289408 | 595968864 | 589824480 | 1179648000 | 1185792864 | 1880676570 | 544469527251 | 23313126115 | 23313126115..23819861281 | 0..506735166 |
| local_ep44 | motion_selector_shared_state | 132366313408 | 595968864 | 589824480 | 1179648000 | 1185792864 | 1880676570 | 536578831251 | 22984347115 | 22984347115..23491082281 | 0..506735166 |

Attention boundaries remain fail-open for cost but fail-closed for claims: H67 is a nonzero abstract packed-summary lower bound; Local has at least the frozen module-call count with unknown nonzero bytes.

The live peak and best-fit allocator include every observed M22 SRAM category (stream tensors, weights, ATLIF input/output/state/parameters, Motion metadata/state, and observed abstract H67 attention) at 96-byte alignment. They are a boundary-materialized trace working set, not a proposed on-chip SRAM macro. The two-copy and allocator-plus-extra-stream values are analytical capacity bounds, not a scheduled ping-pong placement; tiling is still mandatory, and missing Local attention can only increase its bound.

Transport ticks reproduce M22's serialized byte ledger. Bank-service ticks are a separate sequential port envelope. Neither is compute-overlapped system latency, FPS, energy, or speedup.
