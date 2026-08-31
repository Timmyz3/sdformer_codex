# M859/C2 R25 source-author handoff

R25 closes the M856 receipt-key mismatch with one guard-owned 15-key whitelist. The real runner no longer writes its receipt inline: it delegates to `write-pending-receipt`, which uses the same receipt filename/schema/status authority consumed by staging, recursive verification, and recursive publication.

The positive test does not populate work by iterating the whitelist. It explicitly emits the 12 phase files, launch identity and RUN_COMPLETE exactly like the runner, calls the same receipt writer as the real runner, proves that independently produced population equals the shared authority, then completes stage → seal → recursive exact verify → no-replace publish → canonical postverify.

The obsolete R24 receipt name and wrong schema/status are rejected. M803 and compile/run/cycle semantics are byte-identical; R24 recursive safety is unchanged. No VCS, license query or EDA was run.
