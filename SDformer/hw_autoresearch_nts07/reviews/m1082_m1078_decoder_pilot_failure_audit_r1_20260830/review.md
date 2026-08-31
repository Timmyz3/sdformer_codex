# M1082 independent failure audit of M1078

Verdict: **M1078 consumed its sole attempt and failed closed; do not retry. A new-namespace additive validator repair is allowed.**

## Frozen runtime state

- Permanent attempt: `.m1078_m1076_decoder_exact_bool_pilot_attempt_consumed`, exact `attempt.json` SHA `7ea0a3c9...`.
- Published result: absent.
- Original work directory: absent after quarantine move.
- Quarantine: `m1078_m1076_decoder_exact_bool_pilot_r1_20260830.failed_or_incomplete.2631940.23079.18872`, return code 1, exact three files only.
- The attempt and quarantine have no manifest/outer seal under the frozen implementation. Their member identities were rehashed here, but the quarantine remains non-citable. M1077's authoritative seal recomputed exactly to manifest `4999b94b...`, outer `a293c6c6...`.

## First concrete failure

The first failure is D0 window 0, `M1048:D0:SOURCE:000000000`, `SOURCE_INIT_CENSUS`:

- candidate cycles = baseline cycles = 623;
- candidate/baseline exact `total_cycles` = 623;
- both sides independently pass the M768/M861/M890/M896 exact scheduler miter;
- the frozen validator nevertheless requires the entire `candidate_exact` and `baseline_exact` dictionaries to be equal.

The first unequal field is `terminal_readiness_sha256`: candidate `cd7b8dd4...`, baseline `97e9f0ac...`. The second is `transaction_address_sha256`: candidate `be3f74b4...`, baseline `ecfe8c85...`. Reset `boundary_ready_token_sha256` and reset transaction census hashes also differ because candidate/baseline reset transactions are intentionally side-tagged.

This is a **validator identity bug**, not an algorithm/numeric mismatch, transform bug, or M1076 bool/int repair side effect. `transform_layer` preserves both exact dictionaries verbatim; M1076 reaches the inherited M1052 predicate without mutating them.

## Additive repair

Keep M1076/M1078 frozen and use a fresh result/attempt/lock namespace. Validate each exact result independently, bind each `total_cycles` to its side's cycle field, and compare an explicit side-normalized semantic projection plus paired-reset service semantics. Side-specific provenance hashes must remain separately recorded and individually valid, but must not be required equal.

The repaired source needs a different-author hammer and may consume one new attempt only after that hammer. No decoder cycle, speedup, completeness, Table-A, power, or paper claim is admitted by this audit. `docs/359` remains `dedde7ce...`.
