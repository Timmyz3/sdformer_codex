# M847 — C1 M831/R19 P2 stale-epoch functional-failure hammer

Verdict: **PASS100 as a failure audit; R19 remains permanently `FAILED_DO_NOT_CITE`.** The sole released R19 VCS compile/link completed and the normal suite reached all 13 coverage minima, but the run did not complete P2, the held-final test, the six attacks, or the final PASS token. Therefore C1 RTL is still unverified and the M528 `1.746753x` point remains CPU same-ledger only.

## Package and authority

- The canonical result's `SHA256SUMS` and outer seal both verify. The inventory names exactly 141 nonterminal filesystem entries; an independent byte/SHA pass verified every regular file and internal symlink target.
- The terminal marker and receipt agree: runner rc 1, simv child rc 0, phase `functional_and_coverage_gate`, message `functional token`, all claim flags false.
- The exact runner/source/candidate/release/final-hammer bindings match the receipt, and every review/release seal checked here verifies. The one R19 attempt is consumed.
- No exact M831 runner or its simv remained alive. A process-wide search found one unrelated long-running `simv` from another project; it is not an R19 orphan.
- `docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` and was not modified.

## First-cause reconstruction

The sealed log reports all normal minima before the fatal:

`COVERAGE_M533_M528_DW1RW_R8 ... minima=1 normal_covers=13`

The normal task order in TB r8 is 1, 2, 4, 10, 11, 12, 13. Immediately afterward the TB checks `protocol_error==0`, prints that coverage line, and starts P2 **without reset** using:

```systemverilog
build_reference(16'd3);
normal_score_enable = 1'b1;
load_task(16'd3);
```

The RTL start contract accepts a new bank only when `!epoch_seen_q || prep_epoch > newest_epoch_q`. At this point `newest_epoch_q=13`; P2 row 0 otherwise has a legal start/last/reserved envelope and reaches `prep_ready`, but `3 > 13` is false. Thus `prep_accept_w && !prep_semantic_ok_w` is the first fault producer. That accepting edge sets sticky `fault_q`; the next oracle edge sees `protocol_error` and fires line 641 at 5,551,500 ps.

This is a normal-test self-attack by a stale P2 epoch, not a UNIT_DELAY response issue and not an RTL product-capture datapath failure. It is also not attack-state leakage: the explicit attack tasks occur later and were never reached.

The final logged `RAW_OBS` is normal epoch 13, consumer 63, parent 47, age 1, with `forward=1`; that token is recovered on that same cycle. No RAW token survives into P2, and P2 fails during prep before any source/parent payload is issued.

## Minimal successor

Make a TB-only successor that changes the two P2 literals from epoch 3 to the same monotonic epoch 14. Do not reset between the normal suite and P2, because a reset would weaken the intended cross-task isolation test. Freeze RTL r2, SVA r2, foundry macro/model, masks, oracle equations, held-final case, attacks, and runner policy.

A successor is admissible only after a new exact source/release chain and one directed VCS attempt produce, in order:

1. all 13 normal coverage minima;
2. `P2_STRENGTH` with at least one consecutive distinct-read pair and at least two response identity checks;
3. the held-final legal recovery;
4. exactly six attacks;
5. exactly one final PASS token, with no fatal/error/assertion failure;
6. a fresh result hammer.

This review authorizes source authoring only. It authorizes no VCS, simv, license, DC, Formality, timing, speedup, PPA, energy, system, or paper claim.
