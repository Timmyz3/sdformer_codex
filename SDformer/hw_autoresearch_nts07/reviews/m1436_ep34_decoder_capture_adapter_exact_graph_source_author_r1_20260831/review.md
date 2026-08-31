# M1436 exact-graph decoder adapter source review

Status: `PASS_SOURCE_ONLY__FRESH_DIFFERENT_AUTHOR_HAMMER_REQUIRED`.

M1436 is an additive wrapper around frozen M1321. It proves that all 9,880
ordered records have an exact integer `global_order` equal to their file
ordinal and rejects booleans, duplicates, missing rows and non-terminal JSONL.
It also rejects boolean `module_ordinal` values before delegating the existing
payload, support/sign, dynamic-D1-theta and checkpoint-weight checks to M1321.

Seven directed tests pass. They include both M1322 duplicate-order attacks,
the boolean ordinal attack, missing rows and terminal-newline corruption.

No production capture was consumed, no payload was written, and no replay,
cycle, traffic, power, PPA, Table-A, remote, GPU or EDA action was performed.
A fresh different-author source hammer is mandatory before use.
