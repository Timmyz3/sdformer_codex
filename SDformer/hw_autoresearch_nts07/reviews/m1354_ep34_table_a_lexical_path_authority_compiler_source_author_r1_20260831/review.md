# M1354 — lexical-path Table-A authority compiler source author review

## Verdict

`PASS_SOURCE_AUTHOR__FRESH_DIFFERENT_AUTHOR_BLIND_HAMMER_REQUIRED__NO_PRODUCTION_ALLOWLIST`

M1354 leaves M1351 frozen and changes only the path-validation boundary.  It
first establishes an absolute lexical workspace root, rejects `..`, proves
lexical containment, and uses `lstat` on every existing lexical component
through the candidate leaf.  Only after that walk does it resolve and prove
resolved containment independently.  This closes the exact M1353 accepted
symlink-config attack and also rejects symlink ancestors and broken symlinks.

The M1340/M1342/M1351 regression suites pass 10/10, 16/16, and 13/13.  The six
new tests pass, including the original M1353 attack and a legal ordinary-file
fixture.  Source self-check passes; invoking the author-stage CLI without
`--source-self-check` fails closed.

This is source authoring, not Table-A evidence.  The production allowlist has
zero entries and production rows remain zero.  No production build, capture,
GPU, VCS, DC, PT, PTPX, EDA or remote task ran.  A fresh different-author
hammer is mandatory before source admission.  `docs/359` remains unchanged.
