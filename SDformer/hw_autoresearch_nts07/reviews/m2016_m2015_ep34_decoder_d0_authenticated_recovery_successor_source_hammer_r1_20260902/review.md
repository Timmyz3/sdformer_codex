# M2016 independent source QA: FAIL CLOSED

M2015 does **not** authorize production process capture or M2017. Score: **86/100**, with **P0/P1/P2 = 0/1/1**.

The intended control-plane repairs are real. A future review passes only when the complete review tree is sealed, its score is at least 95, all P0/P1/P2 counts are zero, and identity plus narrow authority match exactly. A stable synthetic five-process topology was read exactly ten times, and a PID/starttime change on the second pass was rejected before canonical publication. A strict-subset copytree interruption was preserved in a numbered no-overwrite quarantine and recovered from reverified staging. Correctly sealed but plan-mismatching evidence was preserved and rejected. All these tests ran under CPython 3.6 and 3.12 in temporary directories only.

The central M2013 crash boundary is still broken. When all three allowed filenames exist but `SHA256SUMS.seal.sha256` is truncated, M2015 returns `partial_unsealed`. It then reuses `Q.quarantine_partial_import_work`, whose own precondition reclassifies the directory by filenames alone as `complete` and refuses to move it. The independent hammer called the real `_promote_result_resumable` twice; both resumes failed, the fixed orphan remained, and no target was published. The official M2015 test checks only the classifier and therefore misses this composition failure.

The minimum repair is an additive successor with a local no-replace quarantine helper that accepts all topology-safe `partial_unsealed` states, including a complete allowed filename set with an invalid seal. It must retain the evidence, reverify staging, fresh-copy exclusively, verify against the plan, and publish no-replace. An authenticated sealed plan mismatch must remain a hard preserved failure.

A lower-severity namespace issue also remains. Effective publication through `P` and the `Q` quarantine root point to M2015, but six other `Q` runtime constants still point to M2012. They are inert in the current exported path, yet contradict the broad nested-binding comment and are a maintenance hazard. Bind them all or explicitly narrow and test which path-free Q helpers may be reused.

No production process, remote endpoint, archive, canonical shard/payload namespace, merge, reducer, GPU, or EDA action was accessed. M2015 source/test/contract, predecessor evidence, and docs/359 were not modified.
