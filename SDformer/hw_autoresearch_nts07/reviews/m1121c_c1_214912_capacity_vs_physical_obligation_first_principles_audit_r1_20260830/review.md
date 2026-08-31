# M1121C C1 214,912-B capacity versus physical-obligation audit

## Verdict

`PASS_SCOPE_CORRECTION__M1119C_STOP_APPLIES_TO_PATH_A_ONLY__GO_PATH_C_OR_B_WITH_MATCHED_BOUNDARY`

The `214,912 B` number has been over-interpreted when every byte is treated as
mandatory live storage.  In M1064/M1102 it is a derived, same-ledger capacity
coordinate under a `245,760 B` ceiling.  Those sources explicitly set
`capacity_only_214912B_admitted=false`.  M1000 independently calls the 16-KiB
FIFO/control entry an analytical reserve and says the best-case 93-macro
packing is capacity geometry, not a port-proven implementation.

M1119C/M1120C remain correct under one conditional: if the paper claims that
the complete `214,912 B` coordinate was physically integrated, every byte must
have a live semantic owner or an identical common charge.  Their STOP must not
be generalized to force unused analytical headroom into dummy RAM.

The frozen M1114 opportunity remains legal and unchanged:

- candidate: `434,242,823` raw CPU-model cycles;
- strongest-zero and same-coordinate bit: `763,908,050` cycles;
- ratio: `1.7591725401987818x`;
- boundary: four bottleneck Conv layers, ten samples, `812,160` tasks, exact
  service digest, raw CPU same-ledger only.

This ratio is not RTL cycles, system speedup, throughput/mm2, SRAM PPA or
energy.  Physical closure may support it later but cannot silently promote it.

## Two distinct meanings that must remain separate

1. **Capacity ceiling / ledger reservation.**  `214,912 B` is the arithmetic
   best-case total used to reject designs that exceed the 240-KiB resource
   coordinate.  An unused part of this allowance is headroom, not hardware.
2. **Implemented live storage.**  Only state with a proven producer, consumer,
   lifetime, width/depth and port requirement contributes to physical
   area/power.  Padding a design to the ceiling is neither required nor fair.

The `190,464 B` parent + depth-packed psum + single-group weight number is also
an analytical macro organization, not yet a measured integrated SRAM point.
It assumes conflict-free sharing of 1RW depth groups.  M935 seals two live
`1152-bit` response payload slots (`288 B` payload); it does not establish that
the remaining `16,096 B` of the FIFO/control reserve is live.

## Three legal closure paths

### A. Full semantic mapping and macro-inclusive PPA

This path deliberately promotes all `214,912 B` from ledger arithmetic to a
physical implementation claim.  M1119C/M1120C therefore apply in full.

Required evidence:

- a one-to-one live owner for all `24,448 B` metadata/reserve, including the
  full `16,384 B` FIFO/control reservation;
- address-timed lifetime and 1RW conflict proofs for the 60 psum macros and 24
  weight macros;
- no dummy, tied-off or invented state;
- exact RTL replay plus matched baseline; and
- macro-inclusive DC/PT/hold, equivalence, SAIF/PTPX as claimed.

This is legal but highest cost and is not the recommended DATE-window path.
If the physical ports or schedule differ from M1102, cycles and the ratio must
be rerun.  If they are proven identical, the raw CPU ratio need not be
recomputed, but it still cannot be labeled RTL speedup until replay closes.

### B. Physicalize actual live storage; leave unused allowance as headroom

This path implements only storage that the architecture actually consumes.
The 240-KiB value is an upper bound; unused bytes are explicitly reported as
unallocated headroom and contribute zero area and zero power.  The physical
total is:

`port-proven parent/psum/weight macros + audited live metadata/control state`.

The M935 response payload contributes `288 B`; the rest must come from an
inventory of real RTL state, not from the old proxy sizes.  `190,464 + 288 =
190,752 B` and `55,008 B` headroom are therefore only a provisional arithmetic
lower coordinate, not a final physical result: other live control exists and
the 93-macro 1RW packing still needs a conflict proof.

This path is fair without dummy hardware if candidate and baseline are both
charged for their actual live state under the same technology, ports and
measurement rules.  It does not change the `1.7591725402x` raw CPU result while
the trace, service model, ports and schedule remain unchanged.  It does require
a new actual-live byte inventory, macro packing/port proof, matched RTL replay,
and new storage-inclusive area/timing/power numbers.  Throughput/mm2 must use
the resulting actual area, never an area extrapolated from `214,912 B`.

### C. Identical external common charge

This is M1000's lowest-risk preferred closure.  Put the large source, weight,
psum and DMA stores outside both measured compute islands with identical
capacity, technology, ports and latency.  Either exclude their area
symmetrically or add the identical area to both rows.  Charge each design's
actual accesses using the same per-access latency and energy model; do not
erase differing traffic merely because the memory instance is common.

Candidate-only parent capture, matcher, response queue and control remain
inside the candidate boundary.  The baseline must include its corresponding
interface/adapter logic and use the same library, SDC and counter/debug policy.

This path is fair, requires no dummy storage and preserves the raw CPU ratio if
the frozen external service schedule and latency are unchanged.  It still
requires a matched baseline top, address-timed RTL replay, differential
logic/live-state PPA, and an explicit common-memory latency/energy charge.  It
does not support the phrase "214,912 B physically integrated."

## Recommendation

Use **Path C** for the fastest defensible DATE component comparison, because it
matches M1000 and avoids inventing storage.  Use **Path B** as the stronger
upgrade when the full actual-live SRAM/control inventory and 1RW conflict graph
are available.  Do not pursue Path A merely to preserve the old ledger number.

M1119C should be retained as a valid fail-closed guard for Path A, with its
scope narrowed rather than retracted.

## Rerun matrix

| Quantity | A | B | C |
|---|---:|---:|---:|
| M1102/M1114 raw CPU cycles and `1.7591725402x` | only if ports/schedule/latency change | only if ports/schedule/latency change | only if external service model changes |
| Live-byte inventory | all `214,912 B` | actual state only | candidate/baseline internal state only |
| 1RW conflict/address replay | required | required for physically integrated stores | required at island/external interface |
| Matched RTL cycles | required | required | required |
| Macro/logic area and throughput/mm2 | rerun full | rerun actual-live | rerun differential island; common area shown/excluded symmetrically |
| Setup/hold/equivalence/SAIF/PTPX | rerun as claimed | rerun as claimed | rerun as claimed plus common access energy |

## Publication-safe wording

> Under a common 240-KiB capacity ceiling, exact ten-sample CPU replay exposes
> a 1.759x C1 opportunity.  Physical results use either audited actual-live
> storage or an identical external-memory boundary; unused capacity is reported
> as headroom and is not instantiated as dummy state.

Forbidden until independently closed: `214,912 B physically integrated`,
`1.759x RTL speedup`, macro-inclusive throughput/mm2, energy, or system speedup.

## Scope

This is an additive read-only audit of sealed evidence.  It authors no RTL,
filelist or EDA Tcl, launches no VCS/DC/PT task, modifies no prior evidence, and
does not modify `docs/359`.
