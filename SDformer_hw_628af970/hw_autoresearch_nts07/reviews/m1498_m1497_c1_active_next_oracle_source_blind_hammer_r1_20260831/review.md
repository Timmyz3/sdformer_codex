# M1498 different-author blind hammer — M1497 C1 active-next oracle

## Verdict

**FAIL / DO NOT CITE. M1499 release authoring is forbidden.**

The intended RTL-facing change is good: the new TB differs from frozen R13 in
exactly one oracle fragment, accepts `weight_accepted=0` and
`psum_accepted=!request_first_q`, binds public and latched first/source, rejects
the served source, and fails closed on X/Z. Author tests pass 16/16. The raw
VCS build and four-member clean result are separated, and a symlink attack is
rejected.

The authority chain is not release-safe. Independent local-only mutation
testing produced **22 false negatives in 48 checks**:

1. Ten resealed contract mutations are accepted, including altered identity,
   oracle semantics, raw-result policy, two VCS compiles, automatic retry,
   simv/EDA authorization and an extra top-level field.
2. Runtime validation does not exact-read the parent/M935/wrapper RTL, R3 SVA,
   R15 witness, foundry model, checker/tests, or VCS binary. Exact-pinning the
   M1459 Python file does not execute its frozen-input validation.
3. A mocked zero-return simulation containing only the two PASS tokens is
   accepted without phase/coverage/witness records. The same log plus an
   explicit `Error: assertion failure` line is also published as PASS.
4. The attempt is canonically consumed before `RAW_BUILD.mkdir`, while the
   failure guard starts later. A post-attempt raw-stage collision leaves no
   sealed failure quarantine.

All runtime attacks replaced the tool call with local mocks. No VCS, simv,
synthesis, STA, power, SSH, GPU, or license query was run. `docs/359` remained
`dedde7ce...`; the pre-existing modified `ucli.key` remained byte-identical at
`1107aa2b...`.

The minimum successor is narrow: exact-validate the entire contract and live
frozen corpus, restore M1459's log cardinality/error gates, cover all
post-attempt operations with the quarantine guard, and add regressions for all
22 false negatives. Then request a new different-author blind hammer.
