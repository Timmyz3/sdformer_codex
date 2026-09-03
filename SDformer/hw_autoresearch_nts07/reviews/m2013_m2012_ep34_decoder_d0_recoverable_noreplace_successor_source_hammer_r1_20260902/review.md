# M2013 independent source hammer of M2012

Verdict: **FAIL CLOSED, 80/100**. No production process capture and no M2014 release are authorized. One additive M2015 source successor is authorized.

M2012 does close most of M2010's two named gaps. A strict-subset partial copy is quarantined without overwrite and restarted; a valid sealed orphan promotes; valid-but-plan-mismatched, alien, symlink, and special orphans are preserved and rejected; publication collisions preserve both trees; and a second exact read rejects PID exit/reuse. The inherited M2009 runtime constants and callables are bound to M2012 at import time.

Two P1 findings remain:

1. **Three allowed names with a truncated seal strand recovery.** `inspect_import_work_topology` equates the presence of all three allowed names with a complete transaction. A normal `copytree` interruption can occur after the last pathname is created but before its bytes are complete. The hammer created `result.json`, `SHA256SUMS`, and a truncated `SHA256SUMS.seal.sha256`; two consecutive resume calls rejected the same preserved orphan and never published the target. A successor must distinguish a cryptographically valid sealed tree from a merely name-complete copy. Invalid-seal all-allowed-name trees must be preserved in no-replace quarantine and restarted from the reverified staged tree. A cryptographically valid but plan-mismatched tree must remain a hard rejection.
2. **The process-capture gate does not authenticate M2013.** `capture_process_identity` uses only `strict_json(FUTURE_REVIEW/review.json)` and checks status/identity/authorization. It neither verifies the sealed review tree nor enforces score >=95 and zero P0/P1/P2. The hammer supplied an unsealed review with score 0 and nine findings at every severity; the function still sealed and published a synthetic PRESTOP receipt in a TemporaryDirectory. This can consume the no-replace production receipt namespace before a legitimate review. A successor must verify the M2013 sealed tree and require exact identity, score, severity counts, and authorization before reading any PID or creating `.work`.

All attacks were isolated under `TemporaryDirectory`. No production process, archive, shard, payload, merge, reducer, GPU, or EDA state was opened or changed. `docs/359_DATE终局冻结_20260813.md` remained at the protected SHA.
