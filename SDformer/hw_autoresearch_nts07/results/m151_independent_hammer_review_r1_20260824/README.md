# M151 independent hammer review

Verdict: **68/100, conditional pass only for the standalone two-slot multicast frontend; repair and integration are required before M150 cycle admission.** P0/P1/P2 = **0/6/6**.

## Bottom line

The sealed evidence is real and reproducible at its declared cut. The production VCS input, output, and runner manifests all verify; the exact same RTL SHA is used by the sealed Synopsys DC run, whose input, evidence, and runner manifests also verify. The directed legal workload proves 48 keys, 96 descriptors/outputs, 48 releases, 22 output-stall samples, 42 simultaneous load/descriptor accepts, 84 adjacent descriptor-accept pairs, all four legal destination counts, and four isolated protocol attacks.

M151 nevertheless does **not** close M150 as a cycle-accurate accelerator. It emits one signed-11 vector plus four destination/negate metadata entries, but it omits the four signed accumulator applications and four conflict-free write ports that M150 assumes. It also accepts an atomic 1,056-bit source-vector load, while the frozen M150 evidence states that signed-width-11 vectors require two PWP1024 beats. There is no two-beat assembly path, PWP SRAM/reconstruction, global transpose, address-timed storage, or real heldout replay.

The independent VCS attack found one integration bug that production testing missed. If a previously accepted multicast is stalled and a missing-resident descriptor is presented, combinational `protocol_error` immediately suppresses `multicast_valid`, then the sequential fault branch clears the pending output. The production stall-stability SVA fails at 55.5 ns. Thus fail-closed behavior is not transaction-atomic: an illegal new request can erase an older legal transaction.

M150's **1.805357581x** is a heldout recurrence/cycle opportunity relative to M143r2 B4. Both the M150 and M151 contracts explicitly set cycle-speedup/system-speedup admission false. It is neither an M151 measured throughput ratio nor a full-network/system multiplier.

## Evidence admission

| Evidence | Independent result | Admitted meaning |
|---|---:|---|
| Production VCS source hashes | all 6 pass | exact frozen RTL/SVA/TB/filelist/M150/M151 contract identity |
| Production VCS output hashes | all 4 pass | compile, simulation, assertion report, and receipt unchanged |
| Production VCS runner hash | pass | frozen runner unchanged |
| Production VCS result | PASS | directed legal stream and four isolated attacks only |
| Production SVA covers | 70 both-slot, 42 load/descriptor overlap, 93 full4, 1 each tail1/2/3, 11 stall-recovery, 84 II1, 47 release-with-other-live | cover reachability, not exhaustive lifecycle proof |
| Independent attack VCS | compile/sim rc 0; expected production SVA failure reproduced | pending legal output can be erased by later illegal input |
| Production DC source hashes | all 8 pass, including exact VCS receipt | same RTL and declared scripts/constraints |
| Production DC evidence hashes | all 20 pass | reports/netlist/DDC/SVF/receipt unchanged |
| Production DC runner hash | pass | frozen runner unchanged |
| `docs/359` | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` | unchanged |

The VCS seal is sufficient for the narrow directed claim, not exhaustive correctness. The DC seal is sufficient for the narrow logic-only synthesis report, not PPA, power, macro feasibility, four-write-port feasibility, or system speedup.

## Two-slot lifecycle audit

The normal lifecycle is internally coherent:

1. A load selects only a currently invalid slot.
2. Accepting `last_for_source` marks that slot retiring but leaves it valid.
3. A retiring slot is excluded from further descriptor lookup and is not considered free.
4. The slot clears only when its terminal multicast is accepted; release identity is coupled to that accepted multicast.

The independent test confirms that a full two-slot state cannot accept a third load on the same edge that the terminal output is released. `load_ready` becomes true only after that edge. This is conservative and prevents overwrite, but it is a real transition bubble, not zero-bubble free/reallocate behavior.

For M150 this corner is rare but nonzero. Since a source has at most two K4 descriptors, the frozen counts imply 23,518,182 two-descriptor keys and 4,413 one-descriptor keys. The latter are 0.018761% of active keys. Production TB drives exactly two descriptors for every one of its 48 keys, so it does not measure the one-descriptor transition penalty.

The main lifecycle defect is fault composition. RTL lines 139-155 make `protocol_error` combinational in a new illegal request and suppress the existing output; lines 200-203 then latch the fault and clear the output register. This contradicts the production `ap_multicast_metadata_stable_under_stall` property when the cases are composed. Quarantine must reject/record the bad input while preserving the accepted output until consume or an explicit architectural flush/abort handshake.

Detailed rows are in `lifecycle_audit.csv`.

## M150 composition and interface cuts

| Boundary | M150 requirement | M151 status | Admission |
|---|---|---|---|
| Descriptor ordering | global raw-row/source/destination transpose | M151 consumes an already ordered descriptor | absent |
| Context identity | heldout records, 432 partitions, row, output block, source | only 32-bit sequence + 9-bit row + 4-bit source; packing is undefined | unproven |
| Source delivery | width-aware PWP1024; signed-11 needs two beats | atomic signed-11 96-lane/1,056-bit load | adapter absent |
| Residency | hold one source over up to two descriptors | two 1,056-bit resident slots | implemented at register cut |
| Multicast | one source to one-to-four distinct destinations | four destination IDs and negate bits | metadata implemented |
| Signed application | independent sign per destination | negate is metadata only | absent |
| Commit | up to four destination updates/cycle | no accumulator datapath/write ports | absent |
| Numeric replay | real heldout source vectors and signs | generated signed-11 vectors only | absent |
| SRAM/PWP | storage, ports, width assembly, energy | none | absent |
| Speed | matched integrated recurrence | conditional one-descriptor/cycle frontend cut | not admitted |

M151 is therefore composable with the *idea* of M150, but not plug-compatible with the frozen producer and consumer contracts. A minimum next integration must add a width-tagged 1/2-beat PWP1024 assembler, explicit context packing, four signed apply/accumulate paths with a bank-conflict contract, and a heldout stream miter through release.

## Throughput and area boundaries

Production VCS proves initiation interval one only when the one-entry multicast stage is ready and the referenced source is resident. It does not prove four destination commits per cycle. `multicast_ready` abstracts all omitted downstream capacity, so any four-write-port conflict, signed-add latency, accumulator hazard, or SRAM stall can reduce delivered throughput.

Production DC reports at TSMC28 HPC+, 3.000 ns, ideal clock, ZeroWireload, zero macros:

| Metric | Sealed value |
|---|---:|
| Cell area | 10,858.050069 um2 |
| Cells / sequential cells | 10,545 / 3,331 |
| Logic levels / critical path | 20 / 1.10 ns |
| Setup / hold slack | +1.4266 / +0.0006 ns |
| Ports | 2,352 |
| Macros | 0 |

This is the frontend register/control island only. It includes 2,112 resident vector FFs and another 1,056 egress vector FFs, but excludes PWP delivery, SRAMs, sign application, four accumulator write paths, routing, CTS and power. The 1,056 egress FFs are 31.70% of reported sequential cells and are rewritten for every descriptor, including twice for a typical two-descriptor source. A slot-ID-tokenized egress that keeps the source resident through output acceptance may remove this duplicate register/write activity, but timing, fanout, routing and area must be re-synthesized; no saving is admitted here.

The +0.0006 ns hold result is technically MET only in the ideal-clock/fix-hold setup and is not a physical margin. The 10,858 um2 number must not be added to or compared against a full accelerator until all omitted cuts are priced at a matched corner.

## DATE innovation assessment

The workload observation is useful: M150 shows that source-stationary grouping retains 99.9957% of unrestricted mosaic packing while reducing the live contribution payload to one source vector per descriptor. M151 turns that observation into a small, auditable ping-pong/multicast frontend.

As a standalone DATE contribution, however, dual buffering plus multicast is conventional. The potentially publishable novelty is the *joint* event-matrix transposition, PWP width-aware source residency, K4 destination multicast, and conflict-safe signed commit under real SNN traces. M151 currently implements only the middle metadata/register slice. Until the producer/consumer cuts and real heldout replay close, this is an implementation milestone, not a best-paper-level architecture result.

## Severity list

### P1

1. Illegal input during an already stalled legal output erases that output and fires the production stall-stability SVA.
2. Four per-destination sign/accumulator write paths are absent, so one frontend descriptor/cycle is not four delivered updates/cycle.
3. Atomic 1,056-bit load is not connected to M150's width-aware PWP1024 producer; signed-11 two-beat assembly is absent.
4. Global transpose/context mapping and heldout numeric/ordering replay are absent; M150's 1.805x remains model-only and non-system.
5. The 1,056-bit egress register duplicates resident data per descriptor, weakening the intended residency area/energy story.
6. Production verification omits single-descriptor keys, terminal-output stall plus protocol attack, full-slot release/reload admission, and explicit per-slot transition assertions.

### P2

1. `release_valid` has no independent ready/ack; its lossless coupling to multicast acceptance needs an explicit consumer contract.
2. The 32-bit sequence field has no frozen packing for record/operator/partition/output-block context.
3. No RTL-to-netlist Formality result is sealed for M151.
4. Hold slack is only +0.0006 ns under ideal-clock ZeroWireload assumptions.
5. VCS stores coverage data but seals only named cover matches, not line/condition/toggle percentages.
6. The pass string calls four destination metadata entries `destination_ports=4`; it correctly also says accumulator write ports are false, but paper-facing wording should remove this ambiguity.

No P0 is assigned because no legal-input functional mismatch was reproduced; the data-loss issue requires a subsequent illegal request and is therefore P1 integration severity.

## Required next admission sequence

1. Repair fault atomicity and add SVA proving pending output preservation across all illegal load/descriptor combinations.
2. Replace or justify the per-descriptor 1,056-bit egress copy; synthesize the slot-ID-tokenized alternative at the same exact SHA/corner.
3. Add the PWP1024 1/2-beat assembler and replay real width8/9/10/11 vectors.
4. Add four signed accumulator update ports with bank-conflict/backpressure behavior, then run an integrated VCS miter.
5. Replay the frozen M150 heldout stream cycle by cycle, including the 4,413 single-descriptor keys and all stalls; only then recompute the kernel ratio.
6. Run macro-aware DC/PT/SAIF/PTPX and an Amdahl/full-network model before any system or DATE headline.

Production files and `docs/359` were not modified.
