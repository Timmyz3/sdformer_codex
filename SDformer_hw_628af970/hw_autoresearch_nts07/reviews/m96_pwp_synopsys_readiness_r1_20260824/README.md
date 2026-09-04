# M96 PAFT/PWP Synopsys readiness audit (read-only, r1)

Date: 2026-08-24

## Decision

- **Most functionally complete existing top:** M86-R3, `phase_fsm_sync_banked_guarded_pwp_frontend`.
- **Best existing top for an immediate, honest 3.000 ns logic-only pre-macro DC run:** M85, `guarded_wordpacked_pwp_stream`.
- **No existing top satisfies both conditions.** M86-R3 still embeds eight `460 x 32` bank arrays inside M86-R1. With only standard-cell target DBs, a direct compile can register-map or otherwise uncontrolledly infer 117,760 storage bits, so its area is not a defensible controller or macro-inclusive PPA number. An ordinary outer wrapper cannot intercept those internal arrays.

The immediate recommendation is therefore to run M85 as a clearly labelled **logic-island diagnostic**. In parallel, refactor M86-R3 at the M86-R1 memory boundary, VCS-lockstep the result against R3, and only then synthesize the controller with eight SRAM blackboxes/DB macros.

No Synopsys tool was launched and no production file was changed by this audit.

## Immediate M85 DC target

Top and production-only file list, in dependency order:

```text
top: guarded_wordpacked_pwp_stream
rtl_m82/zero_bubble_elastic_pwp_stream.sv
rtl_m85/guarded_wordpacked_pwp_stream.sv
```

Frozen/default parameters:

```text
ROW_W=10
TAG_W=32
BUFFER_WORDS=3680
```

Clock/reset:

- `clk_core`: rising-edge clock, 3.000 ns period.
- `rst_core`: synchronous active-high reset. It must be constrained as an ordinary timed input; it is not an asynchronous reset and should not be false-pathed merely because its name is reset.

Memory boundary:

- External read-address output: `bank_row_addresses[8*ROW_W-1:0]`.
- External read-data input: `bank_words[255:0]`.
- The current functional interface assumes a combinational bank response. Thus the DC result measures the PWP metadata/mapping/masking/stream logic only, not a timed synchronous SRAM interface.
- No RTL wrapper is required for this standalone logic-only run. A DC file list, driver and SDC are still required. Analyze with `SYNTHESIS` defined to suppress frozen-geometry simulation-only checks.

Recommended constraint convention, matching the current 28 nm handoff:

```tcl
create_clock -name core_clk -period 3.000 -waveform {0 1.500} [get_ports clk_core]
set_clock_uncertainty -setup 0.200 [get_clocks core_clk]
set_clock_uncertainty -hold 0.050 [get_clocks core_clk]
# input delay 0.250 ns and transition 0.100 ns on every input except clk_core,
# including synchronous rst_core; output delay 0.250 ns and load 0.010.
# max fanout 32; fix multiple-port nets.
```

Use the existing TSMC28 setup/min library convention (`ssg0p9v125c` setup, `ffg1p05vm40c` min), an ideal/unpropagated clock and `ZeroWireload`. Preserve hierarchy (`compile_ultra -no_autoungroup`, or at minimum report hierarchy) so the M82 versus M85 contribution remains auditable.

## M85 claim boundary

M85 has strong functional evidence: the actual-record run covers 1,728 phases, 221,184 entries/outputs and 835,383 beats, with the full frozen catalog checked bit-exactly. This supports synthesis readiness, not a performance or PPA headline.

The immediate DC area includes metadata audit, bank-address mapping, word masking, M82 stream scheduling and associated registers. It excludes the eight SRAM banks, SRAM decoders/wiring/timing, producers, consumers, fallback/accumulator and the phase controller. The very wide standalone ports (592-bit metadata input and 1,152-bit output context) plus combinational metadata audit may dominate an isolated timing/area result. Report it as **M85 logic-only, pre-macro, ideal-clock, ZeroWireload**, never as full PAFT/PWP frontend area, paper-ready PPA or system speedup.

## M86-R3 readiness and required memory-boundary refactor

Production-only dependency order for diagnosis:

```text
top: phase_fsm_sync_banked_guarded_pwp_frontend
rtl_m82/zero_bubble_elastic_pwp_stream.sv
rtl_m85/guarded_wordpacked_pwp_stream.sv
rtl_m86/sync_banked_guarded_pwp_frontend.sv
rtl_m86_r3/phase_fsm_sync_banked_guarded_pwp_frontend.sv
```

R3 adds explicit LOAD/COMMIT/EXECUTE/DRAIN/FAULT sequencing and is the strongest functional candidate: the actual-record differential covers 460 unique row loads, metadata commit, 128 descriptors, all 1,728 phases, 221,184 descriptors and 835,383 issues with zero R1-cycle mismatch. Its known review limitations are descriptor count without a complete identity/order proof and fully serial phases with no load/execute overlap.

However, M86-R1 contains `logic [31:0] bank_mem [0:7][0:ROWS-1]` at default `ROWS=460`, plus a 460-bit written-row bitmap and response FIFO. The eight banks total 117,760 bits. A direct standard-cell-only R3 compile is permitted only as an explicitly labelled flat diagnostic whose storage inference and area are uncontrolled; it should not be the first paper-PPA run.

A meaningful R3 pre-macro run needs a synthesis-equivalent M86 memory-boundary refactor, not merely another outer wrapper. The boundary should expose:

- aggregate write enable/valid, common write row `[9:0]`, and write data `[255:0]` split across eight banks;
- read enable, eight independent read rows `[79:0]`, read data `[255:0]`, and the frozen one-cycle response latency;
- either eight blackbox/DB SRAM instances of logical `460 x 32`, or the chosen physical organization (likely `512 x 32`), held `dont_touch` and reported separately.

The current FSM serializes load and execute, so one single-port/1RW macro per bank is sufficient at this boundary: writes share a row, reads use independent per-bank rows, and reads/writes need not overlap. The extracted implementation must be VCS lockstepped against the sealed R3 behavior before DC. Even after macro extraction, the R3 `row_seen[459:0]` and R1 `row_written[459:0]` bitmaps, response FIFO and metadata duplication remain legitimate controller FF area.

## Candidate matrix

| Candidate | Functional position | Memory/timing boundary | Immediate 3 ns DC decision |
|---|---|---|---|
| M84 mapper | Exhaustively verified combinational mapper | External `bank_words`; no clock/register boundary; variable-select/prefix path | Do not use as the primary synchronous top. Add a registered wrapper or constrain as a combinational max-delay microblock. |
| M85 guarded stream | Full actual-record catalog, independent oracle | External bank address/data; no internal large memory | **Run now**, logic-only and pre-macro. No production wrapper required. |
| M86-R1 | Actual records, but simultaneous payload/descriptor valid can deadlock | Internal 8 x 460 x 32 arrays | Reject. |
| M86-R2 | Only directed one-descriptor evidence; three-way deadlock/starvation remains | Inherits internal arrays | Reject. |
| M86-R3 | Most complete phase-FSM and actual-record differential | Inherits internal arrays; outer wrapper cannot intercept | Refactor memory boundary first; then VCS lockstep and DC. |

## Frozen input identities

The machine-readable JSON records exact RTL, contract, VCS file-list, completion and independent-review SHA-256 identities. Existing VCS file lists include SVA and testbench sources and must not be reused as DC file lists.

All current M84/M85/M86 contracts forbid promoting these results to DC/PPA/system-speedup headlines. This audit does not change those claim gates.
