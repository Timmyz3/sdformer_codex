# M860 — C1 M849/R20 held-final failure hammer

Verdict: **PASS100 as a failure audit; R20 remains permanently `FAILED_DO_NOT_CITE`.** R20 fixed the R19 epoch-triplet defect: compile/link completed, all 13 normal minima passed, and P2 reached 19 consecutive-distinct-read pairs plus 189 response-identity checks. The next directed test then failed because the TB sampled a ready level one cycle after the legal handshake had already consumed the held final. This does not demonstrate a foundry-model or RTL parent-authority defect, but R20 still lacks the held-final completion token, six attacks, and final PASS, so C1 RTL remains unverified.

## Package and authority

- The result `SHA256SUMS` and outer seal verify. Its inventory names exactly 141 nonterminal entries; an independent byte/SHA pass verified 120 regular files, two internal symlinks and 19 directories.
- The terminal receipt is self-consistent: runner rc 1, simv child rc 0, phase `functional_and_coverage_gate`, `FAILED_DO_NOT_CITE`, and every claim flag false. The child returns zero because the TB's `$fatal` path calls `$finish`; the runner correctly detects the fatal token and fails closed.
- Compile and link completed for the foundry model, macro adapter, RTL r2 and TB r9. No exact M849 runner or simv orphan remains. One unrelated simulation from another project was excluded by command identity.
- The exact source/candidate/release/final-hammer bindings match the sealed receipt and their double seals verify. The one R20 attempt is consumed.
- `docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` and was not modified.

## What R20 proved before failing

The sealed log contains, in order:

```text
COVERAGE_M533_M528_DW1RW_R8 ... minima=1 normal_covers=13
P2_STRENGTH_M533_M528_DW1RW_R3 consecutive_distinct_reads=19 response_identity_checks=189 minima_pairs=1 minima_responses=2
Fatal ... tb...r9.sv, 1341 ... at time 6381000 ps
later authoritative parent did not release held final
```

TB r9 differs from r8 only in the P2 epoch triplet: reference/load/wait move from 3/3/3 to 14/14/14. The normal frontier is 13 and P2 completes, so the R19 stale-epoch self-attack is resolved and is not causal here. No attack task is reached and no final PASS token exists.

## First-cause cycle reconstruction

The held-final test creates only two active rows: row 0 mask `0001` and row 1 mask `0003`, so row 1's exact parent is row 0 and row 1 itself is not live. Once row 1 reaches its final residual beat, the TB snapshots counters and forces both response slots plus `read_pending_q` nonauthoritative for three cycles. The three stale-hold checks correctly observe no ready, commit, overflow or protocol fault.

At the third check, on the 6,378,000-ps negative edge, TB r9 releases the stale payload and forces:

- `slot0_valid_q=1`, parent 0, consumer 1;
- the exact row-0 signed12 parent vector;
- both architectural sinks ready.

With row 1 not live, `deadline_hold_w=0`. The forced slot exactly satisfies `matching_parent_authoritative_w`; residual/parent arithmetic is in range; therefore `issue_data_ready` becomes the legal **pre-edge** handshake level. The next edge is the 6,379,500-ps positive edge, where the RTL consumes that parent and accepts/completes the final beat.

The TB does not inspect ready before that edge. Instead it executes `@(negedge clk_core)` and checks `issue_data_ready` at 6,381,000 ps. By then the accepted row is complete and `issue_request_valid` has fallen, so `issue_data_ready` must also fall. The fatal therefore asks a ready/valid source to keep `ready` asserted after the transfer it enabled. That is a testbench post-handshake observation error, not an RTL held-final failure.

The foundry macro cannot cause this event: the test forces the authoritative value directly into `slot0_data_q`, keeps `read_pending_q=0`, and never needs a macro response. The epoch repair also cannot cause it because P2 has already printed its strength token before this reset-isolated test begins.

## Minimal successor

Use a new TB-only identity and preserve RTL r2, SVA r2, macro adapter/model, normal/P2 vectors, oracle equations and six attacks byte-for-byte. In `test_held_final_stale_parent_then_legal`, after forcing the legal slot and sink readiness:

1. allow a deterministic one-picosecond combinational settle and check `issue_data_valid && issue_data_ready`, parent identity, zero overflow and zero protocol error **before** the accepting edge;
2. hold all forces through exactly one positive edge;
3. release after the edge (outside the active-edge race), then check exactly one psum commit and one row completion with no protocol error;
4. emit a dedicated held-final recovery token and require it before the six attack counts and final PASS.

Do not change RTL/SVA or add a reset. A fresh source hammer, release chain, one new exact VCS identity and a fresh result hammer are required. This review authorizes source authoring only; it authorizes no VCS, simv, license query, DC, Formality, timing, cycle, speedup, PPA, energy, system or paper claim.
