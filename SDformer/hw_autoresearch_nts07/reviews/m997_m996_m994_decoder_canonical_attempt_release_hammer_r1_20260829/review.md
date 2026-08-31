# M997 independent static hammer of M996 release

## Verdict

`PASS_M997_M996_M994_CANONICAL_ATTEMPT_RELEASE_HAMMER`

`GO_ONE_M998_RUN_ONLY`

Release SHA256:
`7140608515b165db358c1ccee23c3a23712aff5abd0172b3793993c89bc6fc03`.

M996 exactly matches the M994 driver schema and authorizes one run containing
only D2 prefix 10K followed by D3 prefix 10K. `launch_now=false` and
`max_attempts=1`. Retry, 100K, full-row, production and EDA/GPU/remote
authorization are all false.

The release binds the exact M994 contract, M994 driver, M998 runner, sealed M995
review, and original M982 STOP identity. M995's manifest and outer seal verify.
The future runner checks release authority before consuming the canonical
ATTEMPT and requires exact M995/M996/M997 SHA identities.

Ten negative mutations were independently rejected, including reversed row
order, prefix expansion, retry, second attempt, full-row, production, external
tool expansion, paper-citable promotion, and M995 identity drift.

The canonical ATTEMPT, result, work prefix and failure prefix were absent before
and after this static hammer. No 10K model call or external tool ran.

This GO authorizes exactly one M998 run after the final M997 review, manifest
and outer-seal hashes are injected. The resulting M998 output will still require
an independent result hammer and will not be paper-citable on creation.
