# M522/M514 logic-only DC narrow static hammer r3

Verdict: **STATIC GO — exactly one new positive DC run is authorized for runner `58ae904ed019d27690544c474da1df03fd4b3eb69752d3f5e78fae21a1a6402f`.** Score 96/100, P0=0, P1=2, P2=2.

## Narrow normalization repair

The historical M514 VCS `SHA256SUMS` has 94 members and all 94 labels begin with `./`. The previous exact-topology comparison retained that spelling in its expected set while `relative_to(root).as_posix()` produced labels without `./`, so the sets could never match. The r3 runner applies `Path(name).as_posix()` before duplicate detection and set comparison. An independent in-memory replay confirms `Path('./x').as_posix() == 'x'` and closes the real 94-member package.

The repair remains fail-closed: absolute labels and labels containing a `..` path component are rejected before file access; `x` and `./x` collide after normalization and are rejected as a duplicate; digests, regular-file/no-symlink checks, exact inventory equality, manifest identity, and the outer seal are unchanged.

## r2 attempt boundary

The reported r2 attempt failed at historical sealed-manifest topology verification. Independent control-flow inspection places that verification at byte offset 7747, before r3 launch authorization (7955), resource admission (9344), staging creation (9791), and the resolved `snps_shell -f` invocation (12924). The workspace contains no M522 canonical output, staging directory, quarantine, `dc.log`, `dc.rc`, or DC receipt, and no M522/DC process was active during review. The old comparison failure is independently reproducible from the sealed 94-member manifest. Thus the failed preflight did not consume a positive DC run.

## r2 hard-gate drift check

All 12 contract-bound frozen inputs match. The M514 VCS evidence and receipt-blind hammer both pass their member and outer seals; the r2 M522 review seal also passes. The launcher remains a `dc_shell -> snps_shell` symlink, while the resolved regular executable remains bound to SHA `23a4101c...`.

The runner still checks its own exact SHA, the sealed r3 schema/status/P0/authorization and literal runner SHA before resource admission or DC. `SYNTHESIS`, slow target plus fast min library, 3 ns constraints, `ZeroWireload`, explicit ideal clock, the three-source precompile TIM-209/OPT-150 gate, five constraint classes, strict finite receipt readback, staging verification, atomic rename, canonical reverification, and completion-after-verification remain present and correctly ordered. Bash syntax, contract JSON, and all four embedded Python blocks pass static syntax checks.

The two r2 P1s and two P2s remain nonblocking: receipt gate counts rely on earlier exact shell gates; seal exclusions are basename-based; the ideal-clock report is declarative; and the canonical `mv` has a narrow theoretical TOCTOU window.

## Authorization boundary

This review authorizes exactly one new positive execution of the reviewed runner, solely for standalone M514 logic-only 3 ns DC/STA and additive decoder-support area/timing cost. It does not admit cycle or system speedup, energy, full-decoder execution, physical SRAM, Formality, paper-ready PPA, or a DATE headline. The resulting sealed canonical output must pass an independent receipt-blind DC hammer before any area or timing number is admitted.

No runner, DC, or VCS command was executed during this review. Production files and `docs/359` were not modified; `docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
