# M1967 — M1966 independent-load-handshake TB source hammer

## Verdict

**FAIL source gate, 78/100; P0/P1/P2 = 0/4/0.** M1966 correctly separates the baseline and TSBG load valids, wires each valid to the corresponding DUT and SVA instance, latches each side's acceptance independently, keeps one shared payload until both sides have accepted, and adds an exact 10,000-cycle per-descriptor bound. However, it does not satisfy four mandatory M1965 controls: descriptor assertion can still race a DUT sampling posedge, there is no whole-test watchdog before the first load, both completion blocks retain unscoped `join_any` watchdogs without `disable fork`, and the required phase/timeout observability is incomplete. No fresh runner or EDA attempt is authorized.

## Frozen parent binding

The exact M1965 failure review is bound at SHA256 `c0f90256dff7a39c9f2d64e9887b212aad3045fbbcf41f38ccf35d72b13d060d`; its inner and outer seals verify. M1965 permits only an additive successor TB and requires per-side valid/latched accept, drive away from active posedge, per-load plus whole-test watchdogs, named completion forks with cleanup, and complete phase tokens before runner authoring.

M1966 is additive. Its TB SHA256 is `5bf8e7d66396d36f5e287e56cc9667766530a8d595030f0f571a439d91087d41`, and its filelist SHA256 is `93e424c315c650f6607a43328940ffe69b984f16987f5cf9737d95e5940cef7e`. The filelist changes only the TB path relative to M1942. Adapter, RTL, SVA, and `docs/359` remain frozen.

## What M1966 fixes correctly

- `load_valid_base` drives only the baseline DUT and baseline SVA; `load_valid_tsbg` drives only the TSBG DUT and TSBG SVA.
- `base_seen` and `tsbg_seen` latch acceptance independently. An accepted side is deasserted with an NBA while the other side may remain valid; there is no longer a same-cycle `base.load_accept && tsbg.load_accept` requirement.
- The payload is shared and no descriptor field is modified inside the wait loop. The next call to `prepare_descriptor` occurs only after both latched accepts and a negedge, so split acceptance does not by itself mutate the waiting side's payload.
- Each full and recovery descriptor uses one loop with the exact bound `load_wait_cycles < 10000`.
- The frozen arithmetic/reference functions, scoreboard, two attack classes, work/cache ledgers, local 1.15x gate, reset-recovery checks, and unique PASS token are unchanged. The original 31 fatal sites remain text-identical; M1966 adds exactly one new timeout fatal, for 32 total.

## Blocking findings

### P1-1 — The first descriptor of each phase is still driven on an active posedge

The main thread deasserts reset at a `posedge clk_core` and immediately enters `load_workload`; the task asserts both valids with blocking assignments before its first `@(posedge clk_core)`. The recovery load similarly starts immediately after a `repeat (2) @(posedge clk_core)`. SystemVerilog active-region process order is unspecified. A DUT may therefore sample the first descriptor on that same edge while the task begins waiting only for the next edge. If that happens, the descriptor stays valid for a second edge and the already-advanced DUT can enter `ST_FAULT`. M1965 explicitly required descriptor payload/valid drive on negedge or an explicitly skewed clocking block. The NBA deassert after an observed accept is sound, but it does not cure the initial assertion race.

### P1-2 — No whole-test watchdog covers reset release through the first load

M1966 adds a per-descriptor 10,000-cycle loop, but it never forks a bounded whole-test watchdog before reset release or the first load call. The only 300,000-cycle watchdogs are still instantiated after the full and recovery load tasks return. This fails the explicit M1965 liveness contract and leaves unexpected hangs outside the descriptor loop uncovered.

### P1-3 — Both completion forks leave orphan watchdogs

The two original `fork ... join_any` blocks remain, and the file contains no `disable fork`. When either `bundle_done_valid` branch completes, the other completion waiter and the 300,000-cycle fatal thread survive. M1965 required named forks plus `disable fork`, or an equivalent bounded wait for both done signals. The current source does not meet that requirement.

### P1-4 — Phase and timeout diagnostics are incomplete

M1966 emits only four tokens: `full_load_begin`, `full_load_complete`, `recovery_load_begin`, and `recovery_load_complete`. It omits required BEGIN/END observability for reset, full execute, retired replay attack, stale attack, recovery execute, and final checks, and emits no dedicated timeout token. Its per-load fatal is only `M1966 per-side load handshake timeout`; it does not report phase, context/group, per-side valid/ready/accept/pending, `protocol_error`, or `busy` as required by M1965.

## Integrity and claim boundary

- Adapter: `cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156`
- M1880 RTL: `8524f6a7a6d09e1aaab55ee91515bd1fce9ea57fa2a478a9817f637685299a05`
- M1880 SVA: `e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2`
- `docs/359`: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

This review performed read-only source and seal inspection only. It ran no license query, compile, VCS, simv, DC, PT, or other EDA. M1966 admits no functional, performance, PPA, energy, system, or paper claim.

## Required next gate

Author one additive successor TB that preserves the correct M1966 split-valid logic but: (1) drives every new descriptor at negedge or through a clocking block; (2) arms a whole-test watchdog before the first load; (3) replaces both `join_any` blocks with named forks plus cleanup or a bounded both-done task; and (4) adds the complete M1965 phase and timeout diagnostics. A different-author source hammer must bind the exact sealed M1965 failure review and approve that successor before any fresh runner may be authored. M1967 explicitly authorizes no runner and no EDA attempt.
