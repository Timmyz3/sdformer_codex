# M1296 author receipt — atomic decoder diagnostic publisher

Verdict: **AUTHOR_PASS__10_OF_10_SYNTHETIC_TESTS__DIFFERENT_AUTHOR_HAMMER_REQUIRED**.

M1296 is an additive successor to frozen M1284.  The public production entry is zero-argument and has one canonical destination.  A persistent `O_EXCL` marker prevents repeat publication.  The publisher locks and holds the result parent and canonical result directory, records every canonical member through directory-rooted file descriptors, stages and seals an M1296-native payload/token, then repeats the full FD identity, runner seal/schema, completion-token, root-inode and marker-inode checks immediately before atomic no-replace rename.

Synthetic attacks confirm that alternate destinations, repeat publication, result-file mutation after stage, result-root replacement, contract claim promotion, Table-A/full-network/system/headline promotion and source digest promotion all fail closed.  A failed publication deliberately retains the marker and sealed stage; there is no automatic rollback or retry.  A committed annex is never silently removed.

This author check used temporary synthetic fixtures only.  It did not read a live decoder prefix, run canonical preflight/publication/replay, launch EDA/GPU/remote work, or admit any decoder performance result.  A different-author receipt-blind hammer must replay the attacks before release.

`docs/359_DATE终局冻结_20260813.md` remains SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
