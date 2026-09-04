# M1000 C1 same-ledger storage physical-closure review

## Verdict

`PASS_M1000_STORAGE_RECONCILIATION__147246UM2_COMPONENT_ONLY_AFTER_PROMOTION__MAIN_TABLE_BLOCKED`

The CPU and physical evidence are individually useful but do not yet share an
area denominator. M528's `435,293,339` cycles and `1.746753x` versus strong
zero were computed at a `213,376 B` macro-rounded storage coordinate. M962's
`147,246.392090 um2` integrates the C1 logic and exactly nine parent SRAM
macros, but not the full psum store, psum-valid sidecar, weight/PWP tile store,
abstract 16-KiB reserve, source store, or DMA.

Therefore the current area must not be paired with the CPU speedup to claim
throughput/mm2. It is not a full-C1 or system-PPA number.

There is a second, independent publication gate: M989 explicitly marks the
M962 quarantine source as not directly citable until the M990/M991/M992/M993
promotion/release/result-hammer chain completes. After that chain, the area may
enter a **component implementation table**, with the full boundary below. It
still may not enter the main system table without a matched memory boundary.

## Exact reconciliation

M528 reports:

- logical storage: `155,480 B`;
- macro-rounded storage: `213,376 B`;
- 240-KiB margin: `32,384 B`.

M962 physically binds one item exactly: the parent scratch. It uses nine
`128x128` 1RW macros (`18,432 B` physical), while the hard-wired upper address
bit exposes only 64 rows (`9,216 B` logical). The measured macro area is
`78,825.243164 um2`; flattened standard-cell logic is `68,421.148925 um2`.

The remaining `194,944 B` is an accounting obligation, not a statement that
every byte is absent:

- `171,648 B` of macro proxy is definitely external or uninstantiated:
  `119,808 B` psum, `50,688 B` weight payload, and `1,152 B` psum-valid;
- directory, mask and liveness have `5,760 B` of CPU macro proxy but are
  physically present as flattened standard cells;
- the `1,152 B` active-bitmap proxy has no sealed one-to-one RTL identity;
- the `16,384 B` FIFO/control reserve is analytical. M962 contains the actual
  two-entry 1152-bit response queue and scheduler metadata, not a 16-KiB RAM.

The top-level ports make the large omissions observable. The prior psum enters
on `issue_psum_prior[1823:0]` and leaves on `psum_write_data[1823:0]`; no resident
psum array exists. Product/residual data enters on
`issue_residual_data[1151:0]`; no weight/PWP tile memory or DMA exists.
`row_acc_q` and `psum_acc_q` are only current-row state, not the 116,736-B
logical resident psum store.

The source store is also outside the 213,376-B capacity total even though M528
charges `103,680,000 B` of logical source-SRAM traffic. Weight DRAM/DMA is
external while M528 charges `9,069,207,552 B` of weight traffic. These resources
must be common-charged identically or integrated in both comparison tops.

## Why naive macro extrapolation is unsafe

Using the available `128x128` 1RW geometry, a psum row is 1,824 bits. A naive
independent mapping needs 15 macros across each of eight banks: 120 macros and
245,760 B before any other state. A capacity-only depth packing can reduce this
to 60 macros by placing two 64-row banks in one 128-row group, but only if an
exact access trace proves the paired banks never require two operations in the
same cycle.

Similarly, one 3,072-bit weight record is 24 macros wide. Both half-slots fit in
32 rows of one depth group only if their accesses are mutually exclusive. If
they are concurrent, the weight group must be duplicated.

The best capacity-only arrangement is `214,912 B` including the parent, packed
psum, single-group weight, the 16-KiB reserve and all small CPU proxies. It fits
240 KiB by `30,848 B`. The concurrent-weight case is `264,064 B` and exceeds the
budget by `18,304 B`. Neither is a measurement or an admitted design. No area,
timing or power may be obtained by linearly scaling the measured parent macro.

## Minimal DATE-acceptable closure

The lowest-risk path is a strict common-charge matched-component comparison:

1. complete the M990--M993 copy-only promotion and result review;
2. freeze identical external psum, weight, source and DMA memories, ports and
   latencies for the C1 candidate and its baseline, excluding those areas from
   both sides;
3. synthesize a matched baseline top with the same libraries, SDC, ports and
   debug/counter policy;
4. replay the M528 address-timed operation stream through the RTL boundary;
   until then `1.746753x` remains a CPU same-ledger model result;
5. replace the abstract 16-KiB reserve with audited implemented queue/control
   state, or make a real shared buffer an identical common charge;
6. after the boundary is frozen, run hold repair/PT fast, Formality and
   SAIF/PTPX in that order if timing, equivalence or energy claims are needed.

The higher-cost alternative is full-storage integration. Before any RTL or EDA,
export a cycle-addressed access trace and construct a lifetime/conflict graph
for psum banks and weight half-slots. Only a proven 1RW organization should be
bound to foundry macros and sent through VCS plus one controlled DC/PT run.

Directory, mask, liveness, response queue and current-row accumulators should
remain in standard cells for the minimum closure: they are already included,
and their parallel access semantics make a simple 1RW SRAM substitution a new
scheduler rather than a storage-only change.

## Legal paper row after promotion

> C1 matcher/product-capture island; 28-nm pre-route DC; 3-ns ideal clock;
> ZeroWireload; 9x128x128 1RW parent macros included; directory/mask/response
> queue/current accumulators in standard cells; full psum, weight/PWP, abstract
> 16-KiB reserve, source store and DMA excluded; setup only; hold/power false.

This row belongs in a component table. The main system table remains blocked
until either full storage is physically integrated or the identical
common-charge boundary and matched baseline are demonstrated.

## Claim boundary

No EDA was run. No RTL, result, M962 quarantine, M975/M989 evidence, or
`docs/359` was modified. This review adds no measured performance, area, timing,
power or energy point. The packing arithmetic is explicitly capacity geometry,
not physical estimation.
