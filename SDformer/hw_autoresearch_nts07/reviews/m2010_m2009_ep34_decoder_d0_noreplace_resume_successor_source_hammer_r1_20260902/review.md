# M2010 independent source hammer: FAIL CLOSED

M2009 does not authorize process capture or M2011. Score: **86/100**, with **P0/P1/P2 = 0/1/1**.

M2009 genuinely closes three important M2007 gaps. Every canonical tree publication now reaches Linux `renameat2(RENAME_NOREPLACE)`; a raced target is preserved and the sealed source work remains recoverable. A fully copied, sealed import-work tree is checked against the sealed plan before promotion, while a malicious plan-mismatched tree is rejected without overwrite. Campaign archive opens are reported as one and the resume-leg delta as zero. Process records now enforce exact keys and bind raw cmdline bytes, decoded text, and SHA-256. The exact M2006 single-FD, attempt-before-open, all-4200-before-mutation, explicit-M1706 and exact-minus-RSS mechanisms remain pinned. Official and independent tests pass under CPython 3.6 and 3.12.

One original P1 remains at the crash point that motivated it. If `shutil.copytree` is interrupted before the import tree is fully sealed, the fixed `<result>.m2009_import_work` directory remains incomplete. Every manual resume verifies and rejects that same incomplete directory; it is never quarantined and no fresh exclusive copy is started. The synthetic regression reproduced two consecutive failures. The minimum repair is to preserve and atomically no-replace quarantine the incomplete orphan, reverify staged data against the sealed plan, and restart into a fresh work name. A completely sealed plan-matching orphan should remain directly promotable; a malicious sealed orphan must remain rejected.

One audit defect also remains. `capture_process_identity` reads five `/proc` records once and publishes after classification without the second PID/starttime identity read required by M2007. The synthetic changing-starttime seam still emitted `captured_all_five_live=true` after exactly five reads. The successor must re-read and compare every PID/starttime/ppid/raw-command hash/cwd immediately before publication.

No production process, archive, shard/payload namespace, merge, reducer, GPU or EDA action occurred. M2009 source/test/contract and docs/359 were not modified.
