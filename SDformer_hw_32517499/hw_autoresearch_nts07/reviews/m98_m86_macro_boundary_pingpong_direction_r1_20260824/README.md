# M98 M86 macro-boundary and ping-pong direction review

Date: 2026-08-24

Status: `GO_TWO_STAGE_BOUNDARY_THEN_PINGPONG_NO_PERFORMANCE_ADMISSION`

This is a read-only design audit.  It does not modify production RTL and it
does not admit a module, full-network, system, accuracy, PPA, DATE, or headline
speedup.

## Executive decision

Implement one architecture in two admission steps:

1. **M98A: boundary-equivalent single slot.**  Replace M86-R1's behavioral
   `bank_mem` only with eight explicit one-cycle `512x32 1RW` macro wrappers.
   Preserve M86-R3's serial `LOAD -> COMMIT -> EXECUTE -> DRAIN` protocol and
   prove exact cycle lockstep against frozen M86-R3.
2. **M98B: two-slot ping-pong using the same wrapper.**  Instantiate two
   independently owned payload slots, permit writes only to the inactive fill
   slot and reads only from the active slot, attach phase/slot-generation tags,
   and prove transaction equivalence plus a bounded overlap timeline in VCS.

Do not implement ping-pong before M98A passes.  Directly combining a new macro
latency, changed read-during-write semantics, new ownership, and overlap makes a
failure impossible to localize and weakens the evidence chain.

## What exists, line by line

### M85 is a single-metadata consumer

- `rtl_m85/guarded_wordpacked_pwp_stream.sv:46-47` contains one
  `metadata_q`, one `phase_loaded_q`, and one poison bit.
- Lines 89-120 audit the presented 592-bit metadata combinationally.
- Line 124 admits metadata only when M82 is idle; lines 133-137 overwrite the
  one metadata image.
- The lookup path therefore has no payload-buffer ID, phase ID, or generation
  tag.  It assumes the metadata remains stable for the one active phase.

### M86-R1 is one behavioral payload image, not a macro boundary

- `rtl_m86/sync_banked_guarded_pwp_frontend.sv:52-55` contains one
  `bank_mem[8][460]`, one row bitmap, one metadata copy, and one commit bit.
- Lines 163-170 make payload/metadata loading mutually exclusive with active
  descriptor, pending read, response FIFO, and M85 activity.
- Lines 200-206 write all eight 32-bit banks for one row in one accepted cycle.
- Lines 209-213 commit metadata and clear the one row bitmap.
- Lines 231-253 register eight row addresses and enqueue the eight synchronous
  bank responses one cycle later.  The four-entry FIFO supplies independent
  output backpressure.
- There is no macro wrapper, no second payload image, and no safe path for a
  next-phase write while an active-phase read is in progress.

### M86-R3 deliberately serializes the phase

- `rtl_m86_r3/phase_fsm_sync_banked_guarded_pwp_frontend.sv:48-52` defines
  `LOAD`, `COMMIT`, `EXECUTE`, `DRAIN`, and `FAULT`.
- Lines 63-75 forward exactly the request class selected by the current state,
  even if external valid signals overlap.
- Lines 98-117 require exactly 460 unique row accepts and one metadata commit.
- Lines 119-133 require 128 descriptor accepts and full R1 drain before the
  next `LOAD`.
- R3 wraps the unchanged single-image R1 at lines 140-171.  It is therefore a
  sound serial protocol shell, not a latent ping-pong implementation.
- The previous M86-R3 hammer also showed that `128 accepts` is a count contract,
  not an identity/order contract: repeated descriptor identities can satisfy
  the phase count.  M98 must add a phase identity contract even if canonical
  descriptors remain externally ordered.

## The M88 abstraction is not the M86-R3 machine

`analyze_m88_bounded_sync_bank_double_buffer.py` assumes:

- two finite phase slots (`slot_free = compute_end[i-2]` at lines 121-123);
- a serial shared 32-byte/cycle DRAM port;
- 12,288 weight bytes plus the canonical PWP record per phase;
- a 460-cycle row writer and 128-cycle parser running in parallel with DMA;
- preparation `max(DRAM, row writer, parser) + one commit` at lines 103-109;
- compute starts once both the previous compute and this abstract preparation
  have completed at lines 124-132.

M86-R3 instead receives already formatted 256-bit rows and a 592-bit metadata
word.  It implements neither the record DMA/parser nor either weight buffer.
Its visible ideal ingress cost is 460 row accepts plus one commit, and it cannot
overlap those accepts with execution.  Consequently:

| Property | M86-R3 RTL | M88 model |
|---|---:|---:|
| PWP payload images | 1 | 2 |
| weight phase images | outside module | 2 |
| record DMA/parser | absent | modeled |
| active read + next fill | forbidden | assumed |
| phase preparation | preformatted 460 rows + commit | 786-847 cycles |
| output backpressure | executable FIFO/M85/M82 | always-ready compute duration |

The M88 result's `1.409375695x` is versus the same-bandwidth bit-sparse compute
model, not the contribution of double buffering.  The 1,728 preparation values
sum to 1,411,655 cycles per sample and phase 0 costs 838 cycles.  Since M88
reports zero midstream refill stalls, changing the exact five-sample M88
timeline back to serial preparation gives:

- two-slot bounded candidate: 790,706,475 cycles;
- same-work derived serial preparation: 797,760,560 cycles;
- preparation overlap contribution: 7,054,085 cycles, or only
  `1.008921243x` serial/overlapped.

This derived ratio remains an isolated M88 model result.  It is not executable
RTL evidence.  It also proves that M98 ping-pong is mainly evidence closure and
latency hiding, while the larger performance advantage remains the M78/M88 PWP
compute mechanism.

## Single recommended architecture

### Payload macro boundary

Use **16 independent `512x32 1RW` logical SRAM macros**:

- 2 slots x 8 banks/slot;
- one 32-bit write to every bank of the fill slot for each accepted row;
- one 32-bit read from every bank of the active slot for each bank issue;
- one-cycle registered read response;
- validated external row range `0..459`; physical addresses `460..511` are
  padding and never issued;
- no reliance on read-during-write behavior because a slot is never read and
  written under the same ownership epoch.

This organization sustains one 256-bit active read and one 256-bit inactive
write in the same cycle using simple 1RW macros.  The logical two-slot payload
is 29,440 bytes.  A `512x32` binding is 32,768 bytes, adding 3,328 bytes of
padding (11.30% over logical payload).  M88's two payloads plus metadata were a
logical 29,588 bytes; the padded payload plus 148 metadata bytes is 32,916
bytes before row bitmaps, tags, FIFO, ECC, and control.

Do not use an eight-macro `1024x32 1R1W` organization as the reference design.
It is only a fallback if the target memory compiler has a materially better
qualified 1R1W macro.  No extra port, area, frequency, or energy benefit may be
credited without the exact `.db`/macro model and the same-cycle cross-slot
read/write contract.

The wrapper boundary for each bank is:

```text
clk, reset
req_valid, req_write, req_addr[8:0], req_wdata[31:0]
req_ready
rsp_valid, rsp_rdata[31:0], rsp_error
```

All eight banks in a slot accept or stall atomically.  Partial row writes are
forbidden.  A request is counted only when every selected bank is ready.

### External phase-aware interface

Retain the M86 data fields and add explicit identity:

```text
payload_load_valid/ready/accept
payload_phase_seq[31:0], payload_load_row[9:0], payload_load_words[255:0]

phase_seal_valid/ready/accept
phase_seal_seq[31:0], phase_metadata[591:0]

descriptor_valid/ready/accept
descriptor_phase_seq[31:0], descriptor_pattern[3:0],
descriptor_block[2:0], descriptor_tag[31:0]

output_valid/ready/accept
output_phase_seq[31:0], output_tag[31:0], width, escape, values
```

`phase_seq` is distinct from the existing descriptor/output tag.  Internally,
every read request and FIFO entry also carries `slot_id` and `slot_generation`.
These fields prevent a delayed response from an old slot incarnation being
accepted after a swap or reset.

Freeze the actual-record descriptor identity as pattern-major order: accepted
ordinal `k=0..127` must have `pattern=k[6:3]` and `block=k[2:0]`.  This matches
the sealed 1,728-phase replay and closes the duplicate-descriptor hole found by
the M86-R3 hammer without adding a 128-bit descriptor-seen structure.

### Slot state and ownership

Each slot owns its eight banks, 592-bit metadata, 460-bit row-seen bitmap,
9-bit unique-row count, 32-bit phase sequence, generation counter, poison bit,
and one state:

```text
FREE -> FILL -> SEALED -> ACTIVE -> DRAIN -> FREE
```

Only these operations are legal:

- `FILL`: writes and the matching metadata seal;
- `SEALED`: no macro access; waits for the active slot to drain;
- `ACTIVE`: reads and descriptors only;
- `DRAIN`: no new descriptor, only already accepted read/FIFO/M85 work;
- `FREE`: allocation only.

There is at most one `ACTIVE/DRAIN` slot and at most one `FILL/SEALED` slot.
The initial phase fills slot 0.  Once sealed, it becomes active and slot 1 is
allocated for the next sequence.  Thereafter the roles alternate.

A swap is atomic and legal only when:

1. the active slot has accepted the exact 128-descriptor contract;
2. no active descriptor beat, macro response, FIFO entry, M85/M82 transaction,
   or external output remains;
3. the other slot is `SEALED` with 460 unique rows and matching metadata;
4. no fault is sticky.

Keep the existing M85 instance for the first implementation.  At activation,
present the sealed slot's stored metadata through one explicit `ACTIVATE`
cycle, wait for M85's ready/poison result, then permit descriptors.  Charge this
cycle in every simulator; do not silently hide it inside M88's old commit term.
This is smaller and safer than simultaneously refactoring M85 metadata storage.

### Backpressure

- Output stalls may fill the response FIFO and stop active macro read issue.
  They must not stop writes to the other slot because those writes use different
  macros.
- Once the fill slot is sealed, next-phase ingress is backpressured until that
  slot becomes active and the old slot becomes free.  A third phase is never
  accepted.
- If active drain completes before the next slot seals, descriptor readiness
  remains low and execution waits.  M88 predicts this does not occur for its
  always-ready workload, but RTL must implement the stall.
- If the active consumer stalls indefinitely, the sealed slot remains intact
  and ingress remains backpressured; there is no overwrite or ownership steal.
- Macro-bank readiness is reduced across all eight banks so a row accept is
  indivisible.

### Fail-closed contract

For M98B revision 1, any of the following sets one global sticky fault until
reset, suppresses all later accepts/outputs, and flushes logical in-flight
state while ignoring late macro responses:

- duplicate or out-of-range row;
- payload/seal/descriptor phase mismatch;
- seal before exactly 460 unique rows;
- descriptor identity/order violation under the selected 128-entry contract;
- metadata poison;
- write to `SEALED/ACTIVE/DRAIN`, read from `FREE/FILL/SEALED`;
- swap with any active read/FIFO/M85/output work;
- unexpected, late, wrong-slot, or wrong-generation macro response;
- any bank `rsp_error` or non-atomic eight-bank acceptance.

Slot-local abort/retry is deliberately deferred.  A reset-only recovery policy
is easier to verify and is sufficient for the first paper module.  Reset must
invalidate both generations, clear all slot/FIFO/active state, and ensure that
no pre-reset macro response can become visible.

## VCS admission plan

### M98A exact lockstep oracle

Run frozen M86-R3 and M98A with identical legal inputs and identical
`output_ready` cycle by cycle.  Replace direct behavioral-array observation
with an independent one-cycle macro model.  Compare on every cycle:

- all ready/accept signals and FSM counters;
- bank request valid/address and response valid/data;
- FIFO level and enqueue/pop;
- descriptor issue beats;
- output valid/tag/width/escape/value/accept;
- busy and protocol error.

Replay all 1,728 actual M83 phases: 221,184 descriptors/outputs, 835,383 bank
issues including the one escape, the frozen 128 backpressure phases, and the
existing negative attacks.  Any cycle mismatch is a failure.  Also bind SVA to
the macro wrapper for one-cycle response, no request to padded rows, and
all-eight-bank atomicity.

### M98B overlap oracle

Cycle lockstep at the top input is no longer meaningful because M98B accepts
phase `N+1` while R3 intentionally refuses it.  Use two independent checks:

1. **Transaction/reference scoreboard:** queue the same phase records into
   serial M98A and overlapped M98B, align by phase sequence, and require exact
   descriptor, bank-word, output ordering and values.  Relative to each active
   phase start, the compute/read/output subsequence must match M98A under the
   same output-ready schedule, except for the explicitly charged activation
   cycle.
2. **Independent two-slot timeline oracle:** reconstruct allocation, each of
   460 row accepts, seal, activation, 128 descriptor accepts, drain, and free
   events without importing the DUT scheduler.  Check conservation and legal
   ownership for both alternating slots.

Required attacks/covers include duplicate/OOB row, wrong phase sequence, early
and late seal, 129th/wrong-order descriptor, long output stall, fill completion
before and after drain, both slots occupied, reset during fill/execute/drain,
late/unexpected macro response, and swap with pending/FIFO work.  The central
positive cover is simultaneous `active_bank_read_issue && fill_row_accept` for
both slot polarities over actual records.

## Cycles that may be replayed after admission

M98A admits no speedup.  It admits only cycle equivalence of behavioral SRAM and
the macro wrapper.

After M98B VCS passes, the following may be reported for the isolated frontend:

- measured serial M98A versus ping-pong M98B cycles under identical record,
  descriptor, output-ready, macro latency, and reset assumptions;
- actual accepted-row/seal/activation/compute/drain timestamps;
- overlap cycles, exposed refill stalls, and simultaneous read/write cycles;
- the exact two-slot storage and exact compiled-macro PPA once available.

The following may not be replayed from M98 alone:

- M88's 786-847 cycle full preparation, because it includes unimplemented
  record DMA/parser and weight-buffer traffic;
- M88's `1.409375695x`, because that is the full PWP compute-model comparison,
  not ping-pong speedup;
- zero total stalls, since M88 only proved zero abstract refill stalls under an
  always-ready consumer;
- system/full-network/accuracy/DATE/headline speedup.

To re-admit M88 later, supply a separate executable DMA/parser/weight-buffer
producer, timestamp its completed phase preparation, and connect that producer
to the same two-slot ownership contract.  Do not substitute a synthetic
`prepare_done` pulse for this missing hardware.

## Score and risk

This review scores the direction, not an implementation:

| Dimension | Score / 100 |
|---|---:|
| evidence quality | 96 |
| architecture specificity | 94 |
| implementation readiness | 82 |
| hardware innovation potential | 74 |
| performance-advantage potential of ping-pong alone | 58 |
| scoped milestone overall | 90 |

Risk counts: `P0=0`, `P1=5`, `P2=4`.

P1 risks are the missing target macro/DB, absent executable ping-pong RTL,
single-image M85 metadata activation seam, missing descriptor identity/order
contract, and the M88 DMA/parser/weight-preparation gap.  P2 risks are 512-row
macro padding, avoidable zero-tail fills, reset-only recovery, and unmodeled
ECC/control/clock-gating energy.

The direction is GO for M98A followed by M98B.  It is NO-GO for any new
performance claim until the corresponding VCS timestamp receipts exist.
