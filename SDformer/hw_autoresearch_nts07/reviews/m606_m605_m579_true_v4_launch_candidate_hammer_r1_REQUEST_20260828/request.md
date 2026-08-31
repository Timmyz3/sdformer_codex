# M606 request｜M605 M579 nonproduction template + launch-now-false candidate hammer

Perform a fresh independent static hammer of the exact double-sealed M605 template, admission candidate and author
handoff. Do not run the formal 80-record CPU replay, GPU, EDA or remote; do not create a production true-v4 contract,
true release, result, attempt, consumed marker or quarantine; do not modify any reviewed object or `docs/359`.

## Frozen subjects

- template SHA `cdd2fdd07f5b5adfdec32d66ec1cd52fed7e8ed61f2a0eba3c155a3a9ea75a65`
- template outer-seal-file SHA `1c01263427f00db6479aea8a2d8cda350d0015cdbed3af8b266b83d533c4b450`
- admission-candidate SHA `55b3c951df3714a964836e13b3d5bc07f043b7deb74fde95826e44a0fba09c5e`
- candidate outer-seal-file SHA `d3323bd8c97f93377828dc0a5ca305935a82a72e8e5e5c8e4435f5e61e28c998`
- M603 manifest SHA `6503dcb68ce889cb2efe8ae1694769b3460b8718be0fd95f3420e550be407b3d`
- M603 outer-seal-file SHA `8aeea0e49148544ef960159552c9ff68a087b81afd36f1fcabe88fd19862a60f`

## Mandatory attacks

1. Strictly parse all JSON and verify every member/outer seal and exact M601/M603 identity.
2. Prove the template schema is not production v4 and both analyzer and runner reject it before attempt. Check
   authorization is the exact closed dictionary with launch_now/run_cpu/max_attempts/execution_release = false/false/0/false.
3. Compare template `.inputs` byte-for-byte after canonical JSON sorting with M601 candidate `.inputs`; require exactly 15
   keys and the future validator obligation of 80 payload hashes and zero formal records.
4. Verify the production true-v4 path, result, attempt, consumed, quarantine and PID staging are absent under lexists
   semantics. No test may create an attempt.
5. Verify same-parent result/attempt/quarantine state, terminal rehash, member/outer seals and RENAME_NOREPLACE are frozen
   from M601/M603; the candidate may not weaken them.
6. Check resource policy exactly: 3 workers/spawn; future root live precheck of 3x2s, 48-GiB commit, 128-GiB
   MemAvailable, 32-GiB SwapFree, clean cgroup and zero UID-local collision. Confirm the honest boundary
   `runner_enforces_memory_or_collision_policy=false`; the candidate must not pretend this is runner-enforced.
7. Recheck M255 64-frame PAFT regression 1.0189020311889285%, no Pareto, nine-row capacity 213,376 B, and all
   system/RTL/PPA/energy/headline claims false.
8. Recheck `docs/359` SHA is unchanged.

Give score/P0/P1/P2 and PASS/FAIL. PASS requires score>=95 and P0=P1=0. A PASS authorizes only M607 authoring of a
production true-v4 contract and true release synchronously; it does not authorize CPU execution. The true release must
then receive another fresh independent hammer before one exact invocation.

