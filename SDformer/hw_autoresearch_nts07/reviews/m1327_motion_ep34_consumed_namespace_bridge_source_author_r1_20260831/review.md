# M1327 consumed-namespace bridge — author receipt

Status: `PASS_SOURCE_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_PRODUCTION`.

M1327 closes the M1326 P0 without copying or relaxing the identity validator.
Only during unchanged M1319 exact validation, one callback is temporarily
replaced: old M1249 “fresh namespace” is checked as the exact sealed consumed
failure state.  The old attempt must be regular, non-symlink, non-writable and
contain the exact sealed token; old result and canonical log must be absent;
the transferred failure temporary log must remain exactly empty.  `finally`
restores the original callback on success and failure.

The runtime projection remains exactly four keys with 100 attention windows
per call, the exact M1313 cohort, and fresh M1327 contract/output names.  The
author regression passed 10/10 plus source self-check.  Negative tests cover
missing/writable/wrong old attempts, unexpected old result/log, restoration on
exception, actual (unmocked) consumed-state callback execution, new namespace
freshness, runtime drift, output propagation, and delegate restoration.

No remote command, GPU, capture, attempt consumption, production release, or
hardware/paper metric was executed or authorized.
