# M1884 independent hammer: M1882 TSBG-B4 VCS campaign source

Verdict: **FAIL CLOSED (76/100, P0/P1/P2 = 0/3/0).** M1882 must not produce M1885, M1886, a license query, an attempt, VCS/simv execution, EDA evidence, or a paper claim.

The frozen M1880 RTL/SVA/TB/filelist and the M1880/M1881/M1866/M1871/M1875/docs359 chain all re-hash and re-seal correctly. Both CPython 3.6 and 3.12 run the author checker successfully and pass its 70/70 tests with byte-identical checker output. The current runner text also contains one explicit `lmstat`, one SVA-enabled VCS compile, one simv call, fresh namespaces, same-UID locks/resource gates, no-replace publication, and false paper flags.

Those positive checks are insufficient for release because three P1 findings remain:

1. The actual M1882 runner executes `lmstat` at lines 382--391 and only creates the durable attempt latch at lines 393--395. The required order is attempt first, then the first license/tool use. The current order permits an unledgered failed/repeated license preflight.
2. Seven independent early-return mutations in `verify_authority`, `namespaces_fresh`, `collision_gate`, `resource_gate`, `run_tool`, `seal_dir`, and `publish_no_replace` are all accepted by the official checker on both interpreters. Presence and call-site order therefore do not prove live helper effects.
3. Three more mutations are accepted on both interpreters: a second uncounted license query, a faked successful license result with no query, and an extra uncounted simv execution. The 1/1/1 counter text is not a proof of exact tool invocation cardinality.

The independent hammer therefore observes 10/10 escapes on CPython 3.6 and 10/10 on CPython 3.12 while the official suite remains green. No license or EDA tool was queried or run, and no attempt/result/release namespace was created.

Required next action: preserve M1882 as failed review evidence. Author an additive successor that consumes the attempt before its sole license query, guarantees a sealed terminal namespace for every post-attempt failure, and adds AST/reachability mutations for helper bodies and subprocess call cardinality. That successor needs a new different-author review before any release chain.
