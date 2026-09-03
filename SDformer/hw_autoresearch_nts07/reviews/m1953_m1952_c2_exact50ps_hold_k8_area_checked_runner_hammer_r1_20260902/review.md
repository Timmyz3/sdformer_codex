# M1953 — M1952 C2 exact-50-ps K8 area-checked runner hammer

## Verdict

**FAIL / DO NOT AUTHOR M1954 RELEASE.** This was a static-only review: zero license queries, zero attempts, and zero EDA runs.

M1952 substantially improves the predecessor. It is K8-only, binds the exact M1939 Tcl and the M1944 failure review, has one `lmstat` call site and one `dc_shell` call site, publishes the attempt before the license query, blocks same-UID EDA, owns an atomic lock, quarantines failures without retry, and exactly parses the planned M1954/M1955 authority chain. Its independent area path now requires exactly one ordinary decimal `Total cell area:` row, extracts the fourth field with a fixed `awk` rule, checks finiteness, and applies the exact 137,363.9139348 µm² ceiling (= 1.05 × 130,822.775176 µm²).

Two predecessor requirements remain open.

## Blocking area finding

The trusted predicate uses `value >= 0.0`; therefore a malformed zero-area report passes. M1944 explicitly required rejecting nonpositive values. Moreover, `area_posthold_summary_machine.txt` contains the observed area, but the raw `receipt.txt` contains neither that value nor the frozen baseline or observed/baseline ratio. It only states `area_ceiling_percent=5.0` and `post_area_independently_parsed=true`.

A fresh runner must use `value > 0.0`, compute the ratio against 130,822.775176 µm² in the trusted parser, and write the observed area, baseline, ratio, and absolute ceiling directly into the raw receipt. The existing exact-one-row and absolute-ceiling checks should remain.

## Nonblocking census finding

The attempt marker correctly separates authorized and observed labels, but its observed values are a permanent pre-license zero snapshot. The success receipt hard-codes both observed counts to one, while `finish()` writes no authorized or observed counts into a failure receipt. A failure after `lmstat` or after `dc_shell` launch therefore cannot be distinguished from a pre-license failure by authoritative machine fields.

A successor should maintain explicit shell counters, persist each increment immediately before the corresponding process launch, emit those counters from `finish()`, and generate the success receipt from the same counters.

The PASS identity expected by the current parser is internally exact, but this review is intentionally a FAIL artifact and must be rejected by that parser. No M1954/M1955 artifact, license query, attempt, or EDA launch is authorized from this review.
