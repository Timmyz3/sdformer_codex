# M528 r3 recovery static-hammer withdrawal / red-team addendum

## Immediate verdict

The earlier double-sealed PASS review at `reviews/m528_r3_recovery_static_hammer_r1_20260827` is **withdrawn**. It is preserved byte-for-byte as review-error evidence; it must not be edited, deleted, renamed, used to sign an admission, or treated as authorization.

**R3 is permanently NO-LAUNCH.** Do not create a smoke admission, do not run the r3 smoke runner, and do not create a production admission or run r3 production. No smoke or production was launched before this withdrawal.

Red-team score: **86/100, P0=0, P1=2, P2=0**. The exact slow-area fix, JSON live-path audit, wrapper-before-legacy return, and frozen compute semantics remain statically sound. The blocking defects are fail-closed authorization-chain omissions in both runners.

## Withdrawn review identity

| Object | SHA256 |
|---|---|
| withdrawn review `review.json` | `78bdcb888d3b88e21ba2a4cf52358cd2ab70b738e0fa603b16ce67aa4d844d7c` |
| withdrawn review `review.md` | `a5a966d3fd059cc67c5c0100baaef0fcc69fc1f100d61502996d9a3c51ac5c60` |
| withdrawn review manifest | `87e60986efa16d6b8a9ef7b35aa17d025dc77cc53d57b4e5a1218aba604d4379` |
| withdrawn review outer-seal file | `3b5abf952a9e3c1b8485a78e1a4faa59c06b602da4199489fb78dfcc95018c72` |
| r3 analyzer | `a52b4e21bbbe2ab2123763ba0dba7353217fec85f4e8be1c1c24396f2211c0ae` |
| r3 smoke runner | `cf9aaca2178b1e5290490ff720011649f1775493ea06993f27607671e362c126` |
| r3 production runner | `68fed5f590b2c716b000ff94cd79dc7a4646209d0b95786f37752dacf5566685` |
| r3 execution contract | `680a351618fb0cd6e653bc6b2c770d14effa717048bdce67bf9ab98846b8ae65` |
| docs/359 | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` |

## Blocking findings

### P1-01 — smoke runner verifies the static review seal but not the verdict

`run_m528_r3_schema_smoke_r1_exact_sha.sh` verifies that the fixed static-review directory has a valid manifest/outer seal and that the admission-supplied outer-seal hash matches the file. It never parses the sealed `review.json` to require:

- the expected static-review schema and PASS status;
- `p0_count == 0` and `p1_count == 0`;
- the exact analyzer/runner/execution-contract identities;
- explicit authorization for root to sign one smoke-only admission;
- an unwithdrawn claim boundary.

Therefore a sealed FAIL/withdrawn review can satisfy the runner if an admission is mistakenly signed around its outer seal. A seal proves immutability, not semantic admission. Root admission is a necessary authorization layer, but it must not substitute for runner-side verification of the prerequisite review content in this fail-closed chain.

### P1-02 — production runner verifies four outer seals but not their PASS contents

`run_m528_h67_single_port_same_ledger_recompute_r3_exact_sha.sh` verifies the author/static/smoke/smoke-hammer directories and admission-supplied outer-seal hashes. It does not parse the sealed static review, smoke receipt, or smoke-hammer review to require:

- static-review PASS with P0/P1 zero and matching r3 identities;
- smoke receipt PASS status;
- positive exit 0 with exactly one token;
- wrong-pointer and wrong-corner nonzero exits with zero tokens;
- no production output/pool/row replay and zero CPU-production/EDA/GPU/RTL activity;
- smoke-receipt identities matching the runner/analyzer/execution/admission actually consumed;
- smoke-hammer PASS with P0/P1 zero, matching receipt outer seal, and explicit permission for root to sign one production admission.

Consequently a double-sealed failed or incomplete smoke chain could pass the production runner if wrapped in a mistakenly signed production admission. The repeated live positive smoke before the attempt does not repair this gap: it does not repeat the two negative controls or independently prove the receipt/hammer verdict.

## Why P1 rather than P0

No r3 smoke admission, smoke attempt, production admission, or production attempt existed when the defect was caught. The current r3 source cannot be launched under the required P0=0/P1=0 review gate. Thus no result identity or paper evidence was contaminated. The omission is nevertheless blocking and requires new runner identities; it is not a documentation-only issue.

## Mandatory r4 repair gates

R4 must use new runner, execution-contract, author-handoff, static-review, admission, canonical, attempt, and quarantine identities. The r3 runner SHAs above are permanently non-launchable.

### R4 smoke runner must directly parse the sealed static review

After verifying both seals and before consuming the smoke attempt, require with fail-closed `jq -e` or equivalent:

1. exact expected review schema/status/verdict;
2. `p0_count == 0` and `p1_count == 0`;
3. review identity analyzer/smoke-runner/execution-contract SHAs equal the live caller-pinned r4 identities;
4. review binds the author-handoff outer-seal file SHA used by the admission;
5. `authorization.root_may_create_one_new_double_sealed_smoke_only_admission == true` and production authorization is false;
6. no withdrawal/addendum marks that review identity NO-LAUNCH.

The admission must still be caller path/SHA pinned and double sealed. Both semantic review checks and root authorization are required.

### R4 production runner must directly parse all sealed prerequisite payloads

Before dynamic gates and certainly before attempt creation, require:

1. the static-review checks above;
2. exact smoke-receipt schema/status and live identity SHAs;
3. positive `{exit_code: 0, exact_pass_token_count: 1}`;
4. both negative cases with `exit_code != 0` and `pass_token_count == 0`;
5. forbidden activity fields proving no process pool, row replay, production result, CPU production, EDA, GPU, or RTL;
6. exact smoke-hammer schema/status/verdict, `p0_count == 0`, `p1_count == 0`, receipt outer-seal identity match, and explicit permission for root to sign exactly one production admission;
7. production admission fields matching all values parsed above, not merely containing syntactically valid 64-hex strings.

R4 must retain the exact live positive smoke after dynamic resource/collision gates and before production attempt consumption. That repetition is additional defense, not a substitute for the prerequisite semantic checks.

## Boundaries unchanged

R2 remains permanently consumed and non-citable. R3 now joins it as permanently NO-LAUNCH. `docs/359` remains unchanged. There is no r3 cycle, speedup, RTL, VCS, Synopsys PPA, energy, full-network/system speedup, or DATE headline.
