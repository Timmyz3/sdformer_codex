# M1077 independent receipt-blind hammer of M1076

Verdict: **GO for exactly one M1078 diagnostic attempt**, with an independent
M1079 result hammer still mandatory.

The additive M1076 exact-type boundary rejects all four Python JSON aliases at
arbitrary nesting depth: `False -> 0`, `0 -> False`, `True -> 1`, and
`1 -> True`. Independent attacks covered contract, canonical context, payload
receipt, raw rows, result rows, and re-entered both assemble and publish from
mutated on-disk files. A refreshed flat seal did not legitimize any mutation.

M1060 protections were replayed independently: all-fake SHA identities,
nonexistent selected paths, relabel-and-rehash, double-seal publication,
wrong authority/status/schema, extra fields, non-finite values, direct runtime
bypass, and namespace attacks were rejected.

The pre-attempt tripwire observed zero open, zero stat, and zero hash of real
`calls/*` payload members, and the full payload verifier was never called.
M1078 was not run; no canonical attempt/result, cycle, EDA, GPU, or remote work
was created. `docs/359` remains at its frozen SHA.

This receipt authorizes only a diagnostic pilot. It does not admit decoder
completion, continuous-row cycles, local/system performance, or a Table-A row.
