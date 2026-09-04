# M851/C2 R24 source-author handoff

R24 closes the M850 P0 without weakening population checking. The inherited flat exact-root verifier remains unchanged; the new local verifier walks the complete nested tree with `O_NOFOLLOW`, requires exactly 15 payload files, both seal files and exactly the implied `attack/` and `equalbw/` directories, binds every payload to the manifest and the manifest to the outer seal, then repeats verification after `renameat2(RENAME_NOREPLACE)` publication.

The synthetic test exercises the full work → whitelist stage → recursive seal → exact recursive verify → no-replace publish → canonical postverify path. It also proves that the old flat API rejects the same legitimate nested stage, and rejects extra empty directories, recursive symlinks, payload mutation and destination collisions.

M803 RTL/SVA/TB/filelists, compile/run implementation, attack/equal-bandwidth commands and cycle gates are byte-identical to R23. No VCS, license query or EDA was run. The source is ready only for a fresh independent hammer.
