# M2007 independent source hammer: FAIL CLOSED

M2006 does not authorize process capture or M2008. Score: **76/100**, with **P0/P1/P2 = 0/2/2**.

M2006 genuinely closes most of M2004's direct defects: the process set is non-vacuous and has exactly five roles; the overall attempt is O_EXCL and precedes the first archive open; the archive path is opened once and one descriptor carries SHA, population validation, and extraction; all 4,200 remote rows are verified before plan publication and canonical merge; every local row is explicitly pinned to M1706; and receipt equality uses the exact key set minus only RSS. Both official and independent tests pass under CPython 3.6 and 3.12. Traversal, duplicate, symlink, hardlink, FIFO, character-device, late-corrupt, missing-key, unknown-key, wrong-release, empty-PID, and unrelated-PID attacks fail closed.

Two executable transaction gaps still block release:

1. A crash after creation of `<result>.m2003_import_work` strands the only manual plan-resume path. M2006 neither recognizes nor quarantines that inherited work namespace, so the second copy attempt rejects it forever.
2. Canonical publication remains check-then-rename. A target directory created after the absence check can be replaced by `Path.rename`; the synthetic plan-publication race reproduced the overwrite.

Two audit defects also remain: a resumed result reports cumulative archive opens as zero instead of one, and the five-role classifier accepts a stored `cmdline_sha256` inconsistent with its command text.

The minimum successor must make every canonical publication atomically no-replace, recover/quarantine interrupted import-work from the sealed plan, distinguish cumulative and resume-leg archive-open counts, and enforce exact self-consistent process records. These mutations must become permanent CPython 3.6/3.12 regressions.

No production archive, canonical shard/payload, merge, reducer, GPU, or EDA was opened or executed. M2006 source/test/contract and docs/359 were not modified.
