# R37: exact COUNT32 extraction and bounded integer-tree execution

This is a working cheap selector **control**, not a demonstrated response-supervision contribution. The algorithm's fixed LOSO experiment found ordinary y supervision stronger than p supervision, while fold-static y was stronger than both dynamic trees. The hardware result does not reverse that quality finding. No new GPU forward, training or quality evaluation was performed by this hardware path.

## What was actually executed

- All 64 fit identities: original finite `(10,480,640)` FP32 payload streamed once, 12,288,000 bytes each. Every one of 2,048 positive/negative voxel-occupancy counts equals an independently computed NumPy count and the algorithm's count. The resulting FP32 fractions exactly match X128's corresponding 32 columns. No tune/confirm input was accessed by this new implementation.
- All 128 actual OOF tree decisions (64 identities × ordinary y / projected p) were replayed through packed integer parameter memory with per-node reads, comparisons and branch services. Actions and comparison counts equal the algorithm output. Full-fit models are used only for the hardware configuration demonstration; their training-set actions are never substituted for OOF quality.
- First real fit input: a complete address trace of 1,152,061 memory records, all 189 DMA transactions including the 64B tree configuration, actual 128B IOS backing and 192B selector backing. Root independently audited this trace, raw counts and OOF traversal in `../independent_review/COUNT32_RAW_AND_TRACE_INDEPENDENT_V1.json`.
- The same first input was independently executed through actual RTL twice, once continuously and once with input idle gaps, held-valid backpressure and delayed output acceptance. All 192,000 beats and 32 final counters matched, and the configured ordinary tree action matched. The random case exercised 237,100 input-ready idle cycles and 620,598 input-valid backpressure cycles, with output held for 18 cycles. Premature `last` was rejected.
- New DC synthesis completed using the existing 28nm TT library and the same 4ns clock / I/O assumptions as R33. Total cell area is **6,360.690136 library area units**, worst setup slack **+2.71753ns**, no reported constraint violation. Area includes the actual new registers and integer control/datapath plus its idle-only counter debug interface. It excludes IOS/DMA/SRAM macros, PE, BN, stem and the rest of the accelerator. Constant-adder and high-fanout warnings remain visible in the reports; this is pre-layout synthesis without power results.

## Finite service result

| Resource point | Integer feature ready | Tree action ready |
|---|---:|---:|
| Sixteen sign/nonzero classifiers, one shared popcount16 and integer adder | 2,310,098 ticks | feature cost + `2 + 7 × visited_nodes` |
| One sign/nonzero classifier, same remaining resources | 5,190,098 ticks | 5,190,114 ticks on the representative depth-two tree |

All actual OOF node counts and exact totals are retained in `actual_v1/CPU_EXECUTION_V1.json`. The width-two output count bank is serialized; zero occupancy does not receive free skipped updates. Both y and p trees use identical hardware.

The primary point pays 12 ticks per input beat: one DMA ingress write, one IOS read, one classification, two four-tick read/popcount/add/write sequences, and one position advance. It also pays 188 raw-data burst startups, configuration transport and writes, 32 counter clears, and tree traversal. The physical counter bank is initialized at frame start; its first clear is combined with acceptance of that start, followed by 31 remaining clears.

RTL confirms an intrinsic acceptance interval of 11 cycles, including the input acceptance and ten internal cycles. The model adds the surrounding DMA-ingress cycle for each beat, giving 12. The last input acceptance is followed by ten internal cycles, root read and up to two seven-cycle nodes; the modeled output-acceptance cycle is paid as well. Testbench output backpressure adds explicit waiting beyond this minimum. IOS/DMA remain an address-level finite-service model, not RTL in this module.

The old R34 full-X128 feature service was 113,938,624 ticks for the usual nonzero-mass case, but this is **not a resource-matched speedup claim**: COUNT32 discards 96 features and uses new integer hardware. Its quality must belong to the new count32 policies. The existing 2KiB controller reservation could hold the state by capacity, but this round charges a separate implemented 192B register bank and does not assert free physical reuse. No raw-DMA sharing with the stem is credited.

## Same-budget static neighbor

The unchanged R34 finite supply source was copied with its SHA preserved and actually replayed for only two neighboring masks. Keep group2 tiles 302..452, then add lowest remaining complete tile IDs:

| Fixed mask | Four isolated conv service |
|---|---:|
| 155 upgraded tiles | 920,954,340 ticks |
| 156 upgraded tiles | 921,498,660 ticks |

For **every** one of the 128 OOF policy/identity pairs, dynamic151 plus its actual count-tree prefix can pay static155, and cannot pay static156. Static155 leaves 132,434–134,486 ticks under that particular budget. These are measured model services, including the new mask's descriptor/span behavior; no constant per-tile latency was substituted. Static155 AEE is unknown. The old static151 already beat the dynamic trees in the algorithm's diagnostic; this new service point supplies no accuracy rescue.

The sum remains serialized new predictor prefix plus four isolated conv services. The first controlled conv also needs the unmodeled shared stem; its start is `max(action_ready, shared_stem_ready)`. No whole-network latency, hidden overlap, energy benefit or static155 accuracy follows from this sum.

## Evidence and reproduction

- `IMPLEMENTATION_CONTRACT_V1.md`, `CONTRACT_FREEZE_V1.json`: pre-result contract and independent candidate ranking.
- `count_stream.cpp`, `run_cpu.py`: actual input streaming, byte-backed memory, finite ports, 64-input count check and 128 packed OOF traversals.
- `actual_v1/CPU_EXECUTION_V1.json`, `COUNT32_V1.npz`, `first.trace`, `first.trace.dma.csv`: raw source binding, counters, action services and full address trace.
- `rtl/count_tree.sv`, `rtl/test_count_tree.cpp`, `actual_v1/RTL_EXECUTION_V1.jsonl`: input/configuration/clear/backpressure protocol and real full-frame RTL tests.
- `rtl/synth_count.tcl`, `actual_v1/dc/`: actual synthesis source, command log, constraints, timing, area and mapped design. `actual_v1/RTL_DC_RECEIPT_V1.json` binds them and states scope.
- `static_neighbor_supply.cpp`, `actual_v1/STATIC155_SERVICE_V1.json`, `STATIC156_SERVICE_V1.json`, `STATIC_NEIGHBOR_BUDGET_V1.json`: unchanged operator model, real neighbor services and actual per-policy budgets.
- `finalize_evidence.py`: independently binds packed traversal results, static row/plane counts, RTL timing intervals and DC artifacts without any new GPU execution.

CPU reproduction uses `g++ -O2 -std=c++17 count_stream.cpp -o count_stream`, then the CPU environment's Python `run_cpu.py`. The source pin and existing input snapshots are required. RTL uses Verilator 4.028 (`--cc --exe`, then its generated makefile; this version has no `--build`), followed by the first input/config/count paths in `CPU_EXECUTION_V1.json`. DC runs `FLOW_TOP=count_tree FLOW_SOURCE=<absolute rtl/count_tree.sv> dc_shell -f <absolute rtl/synth_count.tcl>` in an empty result directory. These are reproducibility instructions, not automatic permission to overwrite frozen outputs.

Preserved failures and coverage corrections: an initial wrong local path before data processing, the unsupported Verilator `--build` invocation, and an initial random-offer schedule that exercised backpressure but did not create input-ready idle gaps. Their scripts/logs remain under `failures/`; the final randomized test explicitly exercises both.
