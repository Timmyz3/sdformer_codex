# M1476 independent blind hammer — M1475 configuration compatibility

Verdict: **FAIL**, P0=1, P1=0. Launch and M1477 authoring remain forbidden.

The intended compatibility change is narrow and otherwise held. The selected
configuration's frozen selection mapping must still match every field exactly.
Only the observed file's recreated entity metadata may differ; its absolute
path, regular non-symlink type, size, SHA-256, and before/after stat stability
remain mandatory. All selection, observed-file, label-generalization,
checkpoint/profile-bypass, context-restoration, and M1458 result/attempt/log
replacement attacks failed closed.

The final-authority verifier has a P0 type-confusion false negative. It compares
the complete authorization mapping using Python equality, under which `True ==
1`, `False == 0`, and `1.0 == 1`. Consequently, five malformed M1478 authority
mappings were accepted: integer `launch=1`, boolean `runs=true`, float
`runs=1.0`, integer `automatic_retry=0`, and integer
`controller_restore=0`.

The minimal repair is an additive successor with explicit exact-type predicates:
`launch is True`, `type(runs) is int and runs == 1`,
`automatic_retry is False`, and `controller_restore is False`. M1475 and this
failure must remain immutable, and the successor needs a fresh different-author
zero-false-negative hammer before any release is authored.

This hammer used no SSH, real GPU query, capture, production attempt, controller
operation, or EDA operation.
