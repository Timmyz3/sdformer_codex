# M2034 independent M2033 VCS-runner source review

## Verdict

**PASS, 96/100; P0/P1/P2 = 0/1/1.** This review authorizes exactly
one execution of the runner with SHA
`7a3f7340955edcdb5eb68e28c1b92a6fbf3f2fe2baeba8037f254978322ea41d`:
one VCS compile, then one `simv` run, with no automatic retry. The earlier
`a0812f...` source is superseded and receives no authority.

The additive revision closes the blocking findings from the first runner
audit. It starts through a clean `/bin/bash` shebang, removes shell-startup
injection variables, and uses `env -i` allowlists for both VCS and `simv`
without inheriting or redefining `HOME`. Its own SHA is bound independently by
this review and `launch_release.json`; the double-sealed M2032 and M2034
directories, all RTL/TB/fixture/tool identities, result/attempt paths, and the
one-compile/one-simulation budget are checked before the attempt is consumed.

## One-shot and failure behavior

The runner takes an exclusive same-UID M2033 flock and identifies VCS-family
processes through real UID plus `comm`, executable, and `argv[0]`. The blocked
set covers `vcs`, `vcs1`, `vlogan`, `simv`, and truncated
`common_shell_ex*` wrappers associated with a VCS path. It scans once before
attempt consumption and again immediately before VCS.

The immutable attempt directory is created once and before the stage or either
tool. Compile is bounded to 900 seconds and simulation to 180 seconds, each
with a kill-after margin. Any post-stage failure is marked
`FAILED_OR_INCOMPLETE_DO_NOT_CITE`, sealed, and quarantined. Success requires
exactly one full terminal line plus no compile/simulation error, fatal,
assertion failure, watchdog, counter mismatch, numeric mismatch, or protocol
error. The stage is sealed before canonical publication.

Independent checks passed under Python 3.6 and 3.12: 30 static predicates and
38/38 mutations covering input/tool/authority identity, clean environment,
collision axes, lock and dual scan, budgets/timeouts, exact terminal,
UNIT_DELAY, failure quarantine, retry, payload boundary, and forbidden
performance promotion. `bash -n`, a poisoned-caller clean-shebang smoke, and
the M2031 source audit under an empty environment also pass. No EDA, license
query, or GPU process was launched by this reviewer.

## Nonblocking boundaries

The lock is M2033-specific rather than the project-wide cross-campaign EDA
queue. The two scans are adequate for this single controlled launch, but the
operator must not concurrently start another same-UID VCS campaign. Also,
publication uses `mv -T` rather than an explicit no-replace primitive, and the
generic seal verifier does not reject unlisted members. The result reviewer
must therefore verify fresh canonical publication and exact topology,
including no extra, nested, symlink, duplicate, or unsealed member.

## Claim boundary

A future PASS calibrates event counters and signed arithmetic only for one
real ep34 64-row **mask** tile on the frozen M528 r2 island. Lane values remain
synthetic deterministic signed12 values, and prior psum is zero. It does not
calibrate real weights or real prior psums and cannot promote the M1590
`1.694510x` CPU-model ratio to RTL cycles. Same-area performance, timing,
power, energy, full-network/system speedup, and headline claims remain false.
An independent result review is mandatory. docs/359 remains unchanged.
