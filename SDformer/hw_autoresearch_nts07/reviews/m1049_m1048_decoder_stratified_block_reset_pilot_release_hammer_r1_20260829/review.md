# M1049 independent receipt-blind hammer of M1048/M1050

Verdict: **STOP, 41/100, P0/P1/P2 = 2/1/0.** M1048/M1050 must not be executed. The current exact payload identities and the synthetic selector/transaction logic are sound, but two release-protocol failures make the one-shot result non-authoritative.

The first P0 is ordering. The runner calls `validate_source` before consuming the canonical attempt. That validation calls the full M699 sealed-directory verifier, which opens and hashes every payload member. This directly contradicts `attempt_consumed_before_real_payload_open=true`. A tripwire reproduced the call while leaving the canonical attempt absent; this review opened zero real bitpacks.

The second P0 is publication integrity. `assemble()` does not strongly validate `raw_windows.json`; it accepts any raw body with the expected schema and zero top-level mismatch if `result.json` carries the updated raw SHA and false headline flags. A synthetic attack injected `candidate_mean_cycles`, `point_speedup`, a forged status, and a forged completion token. `assemble()` accepted and sealed it. Thus the contract's no-prefill/typed-CI boundary is not enforced at the final sealing boundary.

A P1 remains in contract parsing: the D1 object and arbitrary extra semantic fields are unbound. The current frozen contract does state D1 diagnostic-only and schedules only D0/D2/D3, but a modified caller-pinned contract can rewrite D1 and add point fields while `contract_value()` still accepts it.

Positive checks remain useful: exact M699/M705/M1042 identities passed; selected sample routes are D0/D2/D3 exact binary and D1 exact scaled-binary diagnostic-only; selection is frozen before replay; an independent synthetic census assigned all 85 unique compressed transactions exactly once; missing/wrong pins, direct run bypass, and wrong namespaces were rejected; static runner ordering contains the fixed flock, same-UID EDA collision, and 16 GiB memory/commit gates. These positives do not override either P0.

Required additive repair: move full payload-member verification behind canonical attempt consumption; add strict exact-key recursive raw/result validation before sealing; bind D1 and reject extra contract semantics; then obtain a new independent hammer in a new namespace. M1049 authorizes repair source only, not M1050 execution.
