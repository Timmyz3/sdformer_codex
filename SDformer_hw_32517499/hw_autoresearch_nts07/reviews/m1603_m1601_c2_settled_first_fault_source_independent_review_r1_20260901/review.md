# M1603 — M1601 C2 settled first-fault source independent review

Verdict: **PASS source-only review. Authorize exactly one compile and one
`k8_case0` simulation under a new M1604 result identity. Do not execute from
this review.** M1603 ran no VCS, `simv`, DC, or PTPX.

## Exact source delta

After normalizing five identity-only replacements (top module, trace token,
first-stop token, and two watchdog labels) and deleting the single authorized
settle block, the M1601 testbench is byte-identical to the frozen M1578
testbench. The block contains exactly one `#1ps`, with comments documenting its
purpose. There is no other 1 ps delay.

Within the sole `always @(posedge clk_core)` checker, the order is:

1. increment `cycle_ordinal`;
2. wait exactly one declared timeprecision step (`#1ps` under
   `` `timescale 1ns/1ps``);
3. emit `trace_edge()`;
4. evaluate first-difference, fault/X, difference, clean-done, and watchdog
   decisions.

Thus the repair implements M1594's minimum checker-only change. DUT bindings,
mapped binding, two independent memories, reset schedule, case0 header and raw
stimulus, request/response gating, four-state comparisons, internal taps, stop
priority, and watchdog limits are unchanged. No resynthesis is needed for this
source repair.

## Filelist and authority

The old and new filelists each contain 16 entries. Entries 1--15 are identical
and their content hashes match the frozen standard-cell library, all RTL files,
the M872 mapped netlist, and the reset-safe memory model. Only entry 16 changes
from the M1578 TB to the M1601 TB.

The complete M1594 tree and outer seal pass. Its decision remains
`CHECKER_ACTIVE_REGION_SAMPLING_DEFECT_WITH_COMBINATIONAL_X_OBSERVATION`, with
exactly one post-posedge 1 ps settle as the minimum repair and no resynthesis
required. The M1601 contract also passes its inner and outer seals and grants
no execution before this different-author review.

The author Python suite passes 3/3 and its source checker passes under CPython
3.10. The independent checker rejected 22/22 semantic, placement, filelist, and
contract mutations under both CPython 3.6 and 3.10; the reports are
byte-identical. No simulator or EDA tool was invoked.

## Narrow execution authorization

The only admitted future execution identity is
`m1604_c2_rtl_mapped_k8_case0_settled_first_fault_r1_20260901`, using filelist
`date_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault_source.f` and top
`tb_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault`. Its budget is exactly
one VCS compile and one `k8_case0` `simv` run. UCLI, initreg, SAIF, PTPX,
force/release, a second compile, a second simulation, reuse of M1593 binaries,
or retry after failure is not authorized. The attempt must use a fresh result
directory and be consumed before compile.

If both DUTs remain clean through done, a separate review may consider a later
production-activity source. If a stable X or RTL/mapped difference remains,
the next action is RTL valid/reset isolation followed by DC; no further mapped
simulation is authorized first. Either outcome requires an independent result
review. This source review creates no RTL/mapped PASS and no citable timing,
power, energy, PPA, speedup, or paper claim. docs/359 remains `dedde7ce...`.
