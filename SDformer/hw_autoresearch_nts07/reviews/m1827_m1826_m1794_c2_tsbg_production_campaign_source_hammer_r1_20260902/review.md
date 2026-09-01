# M1827 independent M1826 TSBG governance hammer

Status: `FAIL_CLOSED_M1827_M1826_TSBG_PRODUCTION_CAMPAIGN_SOURCE_HAMMER__P1_1__NO_VCS_NO_EDA_NO_LICENSE`.

M1826 does close the twelve exact mutations reported by M1824. The current runner also contains the intended strong queue/local flock calls, exact namespace tuple, collision set, resource predicates, authority calls, predecessor identity dictionary, and direct private/result publication sequence. All M1794/M1795/M1812/M1813/M1823/M1824 identities and directory or file double seals verify, and `docs/359` remains at `dedde7ce...`.

That is not enough to authorize M1828. On both CPython 3.6.8 and 3.10.18, the positive checker passes and all 73 declared mutations are rejected, but a separately designed fifteen-attack suite escapes **15/15**. The checker validates several call sites and predicates without validating the provenance of their handles/paths or the effective bodies of the called primitives. In-memory changes can redirect the shared lock, alias the local lock, reopen locks with the wrong mode, make the attempt latch reusable or point elsewhere, short-circuit `renameat2`, neutralize collision/resource/failure actions, return early from exact/seal verification, or turn the release-identity mismatch into a no-op while retaining every current token and AST anchor.

This is P1 rather than P0 because the frozen concrete runner is presently strong and no launch has occurred. It is nevertheless launch-blocking: the source hammer still cannot justify the sole no-retry VCS attempt under the project’s semantic-mutation standard. M1828, license query, VCS, simv, attempt, and result are not authorized.

Required successor work is narrow: pin exact path assignments and handle-opening provenance; require argument-free `ATTEMPT.mkdir`; validate the bodies and direct raises of exact/seal/publish/collision/resource/release/quarantine primitives; and add all fifteen M1827 probes before another different-author review.
