# M1701 source-author handoff

Status: `PASS_M1701_M1695_C1_TOOL_ENTITY_REPAIR_SOURCE_AUTHOR_HANDOFF__NO_EDA`.

M1695 did not reach its attempt or launch boundary. Its generic regular-file
hash helper rejected the installed, official, direct `dc_shell -> snps_shell`
entry before caller pins, resource locking, license probing or `mkdir` of the
attempt identity. The result, attempt, work and launch-lock identities are all
absent.

M1701 changes one source gate only. It accepts the entry only when all of the
following remain exact: raw link text `snps_shell`, normalized direct target
`/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell`, resolved path equal to that
same target, target is a non-symlink regular file, and target SHA-256 is
`23a4101...e6d2`. Stat stability and the link entity are checked again across
and after hashing. Arbitrary, absolute, parent-traversing, chained, path-drifted
or SHA-drifted entities remain fail-closed.

The M1695 TCL is byte-identical. The frozen M1665 DDC/SDC, libraries,
3.000/0.200/0.050 ns reported point, 0.081 ns optimization-only guard, one
`set_fix_hold`, one hold-only incremental compile, nine macros, area/DRC gates,
shared resource lock, one attempt/no retry and positive/negative result sealing
are unchanged.

Both CPython 3.6 and 3.12 pass 14/14 source tests; six entity mutations are
rejected. No EDA, attempt, result, RTL, document, GPU, commit or push action was
performed. M1701 remains inert until a different-author M1702 hammer and a
separately sealed M1703 release exist and are caller-pinned.
