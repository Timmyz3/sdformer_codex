# M1975 independent hammer of M1974 bounded TSBG runner

## Verdict

**FAIL — no release, no attempt, no EDA.** The source is close, but the exact runner SHA `85a4632e...d5f6a` is not launchable and does not fully implement M1972's ignored-option gate.

| Severity | Count | Result |
|---|---:|---|
| P0 | 1 | Bound M1967 path does not exist |
| P1 | 1 | Compile/runtime assertion-option diagnostics can evade the negative gates |
| P2 | 0 | — |

## P0 — predecessor path is wrong

M1974 binds:

`reviews/m1967_m1966_m1965_c2_tsbg_b4_independent_load_source_hammer_r1_20260902`

The double-sealed predecessor exists only at:

`reviews/m1967_m1966_m1965_c2_tsbg_b4_independent_load_handshake_source_hammer_r1_20260902`

The missing `_handshake_` means `sha_exact "${M1967}/review.json"` exits before freshness, attempt, license, compile, or simulation. The digest written in the runner is otherwise the correct review digest (`8f39a78a...`). A release for this exact runner is forbidden.

## P1 — known ignored-option diagnostics escape

The M1956 compile log provides the concrete attack text:

- `Warning-[SVAA-RNF] Invalid compile time argument to -assert`
- `Ignoring -assert global_finish_maxfail=1.`

Neither line is matched by M1974's compile regex. The simv negative regex also lacks `SVAA-RNF`, ignored/invalid/unknown assertion-option, and `global_finish_maxfail` alternatives. Thus the receipt could say `sva_runtime_maxfail=1` even if VCS ignored it. The external 180-second timeout and native SVA failure grep remain useful, but M1972 explicitly requires rejection of ignored-option diagnostics.

The successor should reject in **both** logs: `SVAA-RNF`; `global_finish_maxfail` combined with ignored/invalid/unknown wording regardless of word order; and generic ignored/unknown `-assert` diagnostics.

## What passed static attack

- Runner SHA, TB/filelist/RTL/adapter/SVA/docs359 hashes are exact.
- Sealed M1965, actual M1967, M1971, M1972, and M1956 failure/attempt artifacts verify.
- Compile uses only `-assert svaext`; simv receives runtime `-assert global_finish_maxfail=1`.
- GNU `timeout` directly wraps simv for 180 seconds, then TERM and 10-second KILL.
- `set -e` routes timeout or any other nonzero simv exit to sealed quarantine.
- Native property-started/failed, assertion-failed, SVA error, `$error/$fatal`, and fatal diagnostics are rejected.
- Exactly one PASS, ten BEGIN/COMPLETE phase pairs, 52 load begins, 52 completions, and zero load timeouts are required.
- Fresh namespace, same-UID EDA collision, memory, one-license/one-compile/one-sim, no-retry, no-replace publication, and double sealing are present.

No license query, attempt, VCS, simv, DC, PT, signal, or EDA action was performed by this reviewer.

## Required next gate

Create an additive corrected runner with the exact M1967 directory and comprehensive assertion-option diagnostic gates. A different author must hammer that new exact SHA before a release can be authored. M1975 does not authorize a release for M1974.
