# M99 phase-slack sequential metadata audit — direction and draft preflight r1

Date: 2026-08-24  
Mode: read-only architecture review and static draft inspection; no RTL, testbench, SVA or production flow file was modified by this review.

## Decision

**GO for one minimal architecture only: serialize the existing M85 metadata audit at one entry per cycle and retain the existing lookup prefix computation. Do not precompute a 128-entry base table in M99-r1.**

This is the smallest change causally matched to M97. The in-progress, non-final M97 compile exposes exactly 504 32-bit add operators (`126 + 378`), 254 32-bit compares and 128 4-bit compares in the precompile reference report. A later first-compile progress row showed approximately `2,245,469.75` area and a `1.26` path/progress metric; these are observed intermediate values, not sealed PPA. The multiplicities match the 128-entry integer-unrolled load audit. The retained lookup prefix loop has at most seven prior entries and therefore does not justify adding a 128 x 13-bit base table, 1,664 FFs and its read mux before a post-M99 critical-path report says that lookup is the limiter.

The reviewed draft is directionally correct and matches the selected architecture. During this review its only static P1 was closed: simultaneous `phase_load_valid` and `lookup_valid` had been able to accept a phase load and suppress the intended early-lookup fault through the load-priority `if/else`. Current draft SHA `93c638f69a2a50f4d020f4a2d0b974d620574e80b05f96c0a0358008c8883353` gates `phase_load_ready` with `!lookup_valid`, so an old-phase lookup has priority and metadata replacement cannot race it.

Verdict: **GO_DIRECTION_AND_DRAFT_STATIC_PREFLIGHT / NO-GO_RTL_ADMISSION until exact-latency VCS/SVA and same-flow 3 ns DC gates close.** Open static severity after the fix: P0=0, P1=0, P2=3.

## Frozen r1 architecture

### State

Retain the existing 592-bit `metadata_q`; it remains the only code/base metadata store used by lookup. Add only:

- `parse_active_q`: 1 bit;
- `parse_poison_q`: 1 bit;
- `parse_index_q`: 7 bits, exact range 0..127;
- `parse_cursor_q`: 14 bits;
- no compiled-base table and no second metadata image.

The sequential-audit increment is therefore approximately 23 state bits, not 1,664+ base-table bits. `phase_loaded_q`, `phase_poison_q` and `lookup_error_q` already exist in M85.

### Exact edge timing

Let `E0` be the rising edge on which `phase_load_valid && phase_load_ready` is accepted.

- At E0 NBA: latch all 592 metadata bits, clear `phase_loaded`, metadata poison and lookup error, set parser active, index=0 and cursor=0.
- At rising edges E1..E128: audit entries 0..127 respectively, exactly one entry per edge.
- At E128 NBA, while processing entry 127: clear parser active and register `phase_loaded=1` plus the final accumulated metadata poison. Thus `phase_loaded` is visible during the E128→E129 cycle.
- The first legal lookup can be driven after observing that post-E128 value and can handshake at E129. A lookup held before E128 is an early-lookup protocol attack, not a zero-latency request.

There is no extra idle COMMIT state in M99-r1. In edge-sampled SVA, the E128 NBA is naturally observed at the E129 sample; documentation must not mislabel that sampling fact as a second hardware commit cycle.

### One-entry audit recurrence

For entry `i`, `pattern=i[6:3]`, `block=i[2:0]`, `code=metadata_q[3*i +: 3]`, and `cursor` is the packed-word start before this entry.

| code | meaning | used words | fetched words |
|---:|---|---:|---:|
| 0 | signed8 | 24 | 24 |
| 1 | signed9 | 27 | 32 |
| 2 | signed10 | 30 | 32 |
| 3 | signed11 | 33 | 40 |
| 4 | escape | 0 | 0 |
| 5..7 | reserved | 0, plus poison | 0, plus poison |

Every audit edge performs:

1. `next_cursor = cursor + used_words` using a typed 14-bit datapath.
2. `fetch_end = cursor + fetched_words`.
3. Poison if `code>4`, `fetch_end>3680`, or `next_cursor>3680`.
4. If `block==0`, poison unless the supplied 13-bit pattern base equals the current cursor.
5. Accumulate poison monotonically and set `cursor=next_cursor`; a reserved code leaves cursor unchanged, matching M85.

For entry 127 only, also form `rounded_terminal=(next_cursor+7)&14'h3ff8` and poison if it is zero or `next_cursor>3680`. This is equivalent to M85's final terminal rule. Errors are accumulated internally; metadata status changes only at the deterministic final commit.

The reviewed draft implements the same recurrence. Its dynamic code part-select is safe because `parse_index_q` starts at zero, increments only while active, stops at 127 and never wraps. Its dynamic pattern-base select uses `index[6:3]`, giving exact patterns 0..15. Typed local constants would reduce signed-width ambiguity, but the frozen geometry is functionally equivalent.

## Interface, illegal traffic and fault semantics

- `phase_load_ready = !parse_active && !m82_busy && !lookup_valid` is the required r1 arbitration. A held load while parsing or while output/M82 state is busy is ordinary backpressure, not a fault; metadata must remain stable until acceptance.
- A load acceptance atomically replaces the old phase only when no lookup is presented and no M82 transaction/output is resident. It clears prior metadata and lookup faults, preserving M85's load-based recovery.
- `busy = parse_active || m82_busy`.
- `phase_loaded=0` throughout audit. Any `lookup_valid` during that interval is illegal: `lookup_ready=0`, no beat/output is accepted, and `lookup_error` becomes sticky until reset or the next accepted phase load.
- Simultaneous load and lookup must never be silently swallowed. With the required ready gate, a valid old-phase lookup has priority and the load waits; an invalid/unloaded lookup faults and the load waits.
- An invalid metadata image still reaches the deterministic E128 commit, with `phase_loaded=1`, `metadata_error=1`, `protocol_error=1`, and `lookup_ready=0`. Keeping `phase_loaded=1` on a poisoned image preserves the existing M85 externally tested state semantics; loaded does not mean usable.
- Reserved code, wrong canonical base, fetch overflow, cursor overflow and zero terminal all poison. There is no data-dependent early completion.
- For a legal committed phase, descriptor width, escape, 3/4/4/5 beat count, bank-row mapping, final-word mask, tag and 96 x signed12 output are unchanged from M85.
- A legal lookup held while M82/output is backpressured is not a fault merely because `lookup_ready=0`; it faults only when `mapper_valid=0`. Lookup inputs and bank data must remain stable under ready/valid backpressure.
- `output_valid/tag/width/escape/values` remain entirely owned by unchanged M82. They must stay stable while `output_valid && !output_ready`, and the next transaction remains blocked exactly as in M85.
- Synchronous active-high reset aborts an in-progress audit, clears loaded/poison/lookup fault and destructively flushes M82; no ghost phase or output may appear.

## Why r1 does not precompute bases

The observed M97 explosion is the phase-load audit, not the lookup prefix:

- audit: 128 entries with 32-bit `integer` cursor/fetch operations produced the 504/254/128 resource signature;
- lookup: at most seven prior-code contributions, already represented by only seven 14-bit prefix additions in the precompile report.

Sequential audit should reduce the audit cone to roughly two or three 14-bit additions/comparisons plus the dynamic metadata select. The draft adds about 23 FFs. A base-precompute variant would add 1,664 FFs for 128 x 13-bit starts, a lookup read mux and verification/ownership complexity. It is explicitly **NO-GO for r1**. It may be reconsidered only if a sealed M99 DC report both meets the audit resource-reduction gate and identifies the retained lookup-prefix path as the failing 3 ns path.

## Expected hardware effect — estimate, not admission

- 32-bit audit adders: 504 observed in M97 precompile → target at most 16 total module 32-bit add references, with no replicated 128-entry audit family.
- 32-bit compares: 254 → target at most 16.
- 4-bit compare family: 128 → target at most 8.
- Added sequential state: approximately 23 FFs; no base-table FFs.
- Audit critical path: dynamic 3-bit code select + small decode + 14-bit cursor/fetch arithmetic/comparison. The final entry also has terminal rounding. The likely module critical path should migrate to retained lookup/M82 logic.
- Directional area expectation: a 70–90% reduction from the observed 2.245M intermediate M97 area is plausible, but no absolute area or Fmax is admitted until same-SHA 3 ns DC completes. The minimum acceptance is relative and report-based, not this estimate.

The draft currently computes next cursor, fetch end and rounded terminal combinationally every parse cycle. This remains only a few 14-bit operators and is acceptable for r1. Terminal arithmetic can later be gated to entry127 or reduced algebraically only if timing requires it; such a rewrite is not needed to establish the main win.

## M88 cycle accounting

The production M88 result already has:

- `metadata_parser_cycles=128`;
- preparation 786..847 cycles;
- `minimum_compute_minus_prepare_margin_cycles=12645`;
- zero modeled midstream phase-refill stalls.

M99 has exactly 128 parser edges from E1 through E128 and no additional idle commit state. Therefore **M99 adds zero cycles to the M88 model only under the existing M88 schedule assumption**: metadata acceptance/parser start occurs at the modeled preparation start and overlaps shared DRAM plus the 460-row writer. Parser 128 is already below the minimum 786-cycle preparation.

This zero is not automatic for the current M86-R1 wiring. M86 currently gates phase metadata acceptance until all 460 rows are written. Dropping M99 behind that gate would serialize its 128 cycles after row loading; the large 12,645-cycle double-buffer margin would likely preserve zero midstream refill stalls, but phase-0/startup accounting would gain 128 cycles and could not be called zero-new-charge. M88 admission therefore requires the future loader/double-buffer controller to start M99 on the inactive slot in parallel with row/DRAM preparation, and to activate the slot only when row-ready and metadata-loaded are both true.

M88's existing `+1 metadata commit` may remain as conservative slot ownership/activation cost. It must not be accompanied by another newly inserted M99 idle commit cycle.

## Actual-record TB changes

1. Replace the one-edge load expectation with an edge counter: accept at E0, require `phase_loaded=0` through post-NBA E127, and require `phase_loaded`/final poison at post-NBA E128. Drive the first legal lookup after E128 so its first handshake is E129.
2. Replay all 1,728 phases, 221,184 entries/outputs and 835,383 beats. Preserve exact tags, widths, escape, signed values, final masks and no-bubble checks after phase commit.
3. For every regular beat, independently compute and compare all eight `bank_row_addresses`, not only reconstructed output data. This directly covers retained lookup-prefix equivalence.
4. Add a latency-normalized old-M85 versus M99 miter: hold lookup traffic until M99 commits, then require cycle-identical ready/address/output behavior under the same randomized backpressure. The old M85 instance is a functional reference after alignment, not a timing reference.
5. Poison campaigns must include reserved codes 5/6/7 at early/middle/final indices; wrong bases at patterns 0/4/15; fetch/cursor overflow near the capacity boundary; and all-escape zero terminal. Every case must take the same 128 audit edges and commit loaded+poisoned.
6. Attack lookup at audit entries 0, 63 and 127; prove no accept/output, sticky lookup fault, and recovery only on reset/new load.
7. Hold a second `phase_load_valid` throughout audit, mutate unaccepted metadata only in a producer-protocol negative test, and prove no premature accept. Check clean handoff after commit.
8. Test simultaneous load/lookup arbitration, reset at entries 0/63/127, long output stalls, and metadata changes after an accepted load to prove the captured image—not live input—is audited.

## Required SVA

- exact 128-entry progress and no early loaded; final-entry edge implies loaded on the following sampled clock;
- `parse_active -> parse_index<=127`, index increments exactly once per parse edge, final index terminates, and no wrap;
- cursor delta is exactly 24/27/30/33/0 for codes 0/1/2/3/4 and zero for reserved codes;
- poison is monotonic during audit and final poison equals the OR of every entry/base/fetch/cursor/terminal violation;
- metadata is stable internally from E0 through completion;
- `!phase_loaded || metadata_error || lookup_error -> !lookup_ready`;
- lookup presented during audit eventually raises sticky protocol error and cannot create M82 beat/output activity;
- phase load and lookup accepts are mutually exclusive; stalled phase-load payload is stable by environment contract;
- phase-loaded/poison remain stable until reset or a new accepted load;
- existing M85 output accept, held-output stability, escape-zero and protocol-reflection assertions remain;
- covers for legal commit, every poison class, early lookup at first/middle/final audit cycle, held next load, simultaneous-valid arbitration, reset abort and first lookup at E129.

Because SVA samples before NBA, the latency property should be written around internal final-entry completion (for example, final-entry-active implies loaded at the next sampled edge) and paired with a TB post-edge `#1` check that loaded becomes visible after E128. This avoids an off-by-one claim.

## Minimum acceptance gate

M99 may replace M85 only when all are true:

1. The now-closed simultaneous load/lookup seam is asserted in VCS/SVA.
2. VCS exact-source compile/simulation passes the full actual-record replay with 0 value, tag, width, escape, mask, address, accept and output mismatches; counts remain 1,728 / 221,184 / 835,383.
3. Exact load-to-loaded timing is E0 accept, E1..E128 audit, E128 NBA visible, first legal lookup handshake E129; all poison cases are fixed latency.
4. Independent negative tests cover all metadata, early-lookup, held-load, simultaneous-valid, reset and output-backpressure classes listed above.
5. Same TSMC28 3.000 ns, ideal-clock, ZeroWireload logic-only DC meets setup and hold, has no unresolved production references/macros, and preserves the M97 claim boundary.
6. Precompile/resource audit shows 32-bit add count <=16, 32-bit compare count <=16 and 4-bit compare count <=8, or an independently attributable >=95% removal of the M85 audit operator family.
7. Mapped area is <=35% of the sealed M97 logic-only area under identical libraries/constraints. If M97 never seals, M99 must instead provide an independently reviewed audit-cone hierarchy comparison; the observed 2.245M intermediate number is not a denominator.
8. The M88 analyzer/receipt continues to charge parser=128 and existing commit=1, proves zero extra parser charge only under concurrent inactive-slot preparation, and records that current post-row M86 integration is not that schedule.

Passing these gates admits only a standalone/loader-integrated logic-island optimization. It does not admit complete PWP timing, SRAM PPA, energy, full-module/system speedup or a DATE headline.
