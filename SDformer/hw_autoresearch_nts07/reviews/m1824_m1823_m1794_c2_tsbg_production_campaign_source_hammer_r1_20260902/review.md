# M1824 independent source hammer of M1823 TSBG governance

Status: `FAIL_CLOSED_M1824_M1823_TSBG_PRODUCTION_CAMPAIGN_SOURCE_HAMMER__P1_1__NO_VCS_NO_EDA_NO_LICENSE`

Score: 90/100. Findings: P0=0, P1=1, P2=0. M1825 and the M1823 VCS campaign are not authorized.

## Verified closure

M1823 really closes the nine exact M1813 escapes and adds the self-runner-pin
attack. Python 3.6 and 3.10 each pass the source checker, parse the runner,
checker, and test, and reject all 58 declared mutations. The concrete runner's
main path currently calls authority validation, source validation, two namespace
checks, both flocks, collision and resource gates in the intended order. The
tool wrapper repeats source and collision validation. `ATTEMPT.mkdir()` is
immediately followed by `state["attempt"] = True`; the compile command retains
`-assert svaext`. M1794, M1795, M1812, M1813, and docs359 identities match.

## Blocking equivalent bypasses

The AST checker proves direct-call names, not all of their security-relevant
arguments or bodies. A separate twelve-attack probe was accepted 12/12. It can:

- downgrade either flock or use the wrong handle;
- make canonical atomic publication unreachable;
- bind the source-contract authority pin to the runner;
- remove `ATTEMPT` from the freshness tuple;
- empty the same-UID collision set or zero a resource threshold; and
- remove the M1812 contract or M1813 review/manifest/outer-seal keys from the
  future-release identity.

The present frozen runner is written correctly, but the claimed semantic hammer
cannot distinguish those equivalent bypasses. Exact contract SHA protection is
not counted as semantic proof under the project's own M1795 rule. This is one
P1 and blocks the sole no-retry VCS launch.

Do not overwrite M1823. An additive successor must parse the exact flock
handles/modes, namespace tuple, collision set, resource thresholds, authority
target/pin pairs, reachable atomic publication calls, and all M1812/M1813
identity keys and values; then explicitly mutate and reject each boundary before
another different-author review.

No EDA or license was invoked, no release/attempt/result was created, and no
predecessor or docs359 file was modified.
