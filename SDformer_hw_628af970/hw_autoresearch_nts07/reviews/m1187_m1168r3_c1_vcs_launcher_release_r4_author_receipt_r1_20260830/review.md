# M1187 additive C1 R3-source / R4-launch author receipt

Verdict: **SOURCE READY, 99/100, P0=0, P1=0. A fresh different-author
M1188 release hammer is mandatory before launch.**

This additive package leaves the frozen M1162/M935 RTL, R3 SVA, R3 testbench,
R3 source checker, filelist, old R3 runner/release, and all old namespaces
untouched.  It repairs only the launch chain:

1. The release and executable pre-attempt gate now use the same canonical
   `source_contract_sha256` field.  The exact gate was executed source-only on a
   synthetic recursively sealed release-hammer fixture and completed without a
   `KeyError`.
2. A fresh M1188 release-hammer review and outer seal are now required exact-SHA
   environment inputs.  Before attempt creation, the runner recursively verifies
   that directory and checks schema, PASS status, GO verdict, score, P0/P1,
   release/runner/contract identity, zero EDA execution, and one-compile/one-simv
   authorization.
3. R4 uses fresh M1187 attempt/result/work/quarantine names.  Consumed R2 and
   unconsumed R3 coordinates remain non-reusable.

The static checker passed the exact pre-attempt parse and rejected 12 mutations.
The runner retains one foundry-`UNIT_DELAY` compile, one simv run, the same-UID
EDA scan, 64-GiB memory gate, recursive failure quarantine, success sealing, and
the frozen normal-mask/coverage token.

No runner, VCS, simv, EDA executable, or license client ran.  No attempt or
result was created.  All performance/PPA/power/system/paper claims remain false.
`docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
