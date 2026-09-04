# M944 | M943/M941/M938/M935 C1 VCS runner source hammer

## Verdict

`PASS_M944_M943_VCS_SOURCE_HAMMER`, score 97/100, verdict `GO`.
P0=0, P1=0, P2=2.  GO authorizes only creation of the separately sealed M945
exact-SHA launch release; M944 does not release or run VCS.

`bash -n` passes.  M944 ran only the M938 pure static checker, which passed.
No VCS, simv, DC, EDA, GPU, remote, network, or license action was invoked.

## Identity chain

The runner SHA is `e9ee27befaef16afeb14f93aa474e31bcc10ef45ff13abe994cdd9605996059b`
and the M943 source contract SHA is
`454f748eba467fa3ee38411a9f3e9051e27b35302ec9f929e0dc4b0009f5fccc`.
The contract's two sidecars validate: the first binds the JSON; the second
binds the first sidecar.

Runner and contract agree on exact SHA for M935 RTL, macro wrapper, inherited
execution SVA, repaired M938 match SVA, M938 TB/checker/source contract,
foundry UNIT_DELAY model, VCS binary, and `docs/359`.  All on-disk values match.
The real M941 manifest is `3aae9106...`; its outer-seal file SHA is
`8bac72dc...`; both are correctly distinguished and recursively verified.

The future M944/M945 loop is fail closed: the runner requires three nonempty
environment SHAs, recursively verifies this M944 directory, checks exact
review and outer-seal hashes, checks the exact release SHA and both release
sidecars, and then asserts M944 status/GO/score/P0/P1 plus M945 runner,
contract, hammer review/manifest/outer identities and one-compile/one-sim
authorization.  The current absence of M945 therefore prevents launch.

## Attempt, collision, and resource gates

- RESULT, ATTEMPT, and PID-unique WORK must all be absent before consumption.
- ATTEMPT is created before WORK/tool launch and survives every later failure,
  enforcing at most one released attempt.
- The EXIT trap writes a failure marker, recursively seals the incomplete work,
  and moves it to a unique quarantine.  A success is sealed before atomic move
  to RESULT.
- The same-UID `/proc` scan excludes runner ancestry and blocks VCS/simv and
  Synopsys DC/PT/FM/ICC/common-shell processes by `comm` or argv basename.
- `MemAvailable` must be at least 64 GiB.
- Source contains exactly one VCS compile pipeline and one timeout-bounded simv
  pipeline.  M945 must authorize exactly those counts and zero other EDA runs.

## Shell and result admission

Each tool pipeline saves `PIPESTATUS` immediately into an array and requires
both the tool/timeout and `tee` status to be zero.  The 900-second sim timeout
is fail closed.  `set -euo pipefail` remains active.

The four exact-count greps match actual TB line prefixes: one main PASS, one
reset coverage, one exact-match coverage, one inherited metadata coverage,
and one foundry-response-strength line.  The final anchored regex follows the
TB field order and requires attacks=6, nonzero bank-distinct overlap, all three
reset counts equal one, functional-only=true, and timing/speedup/PPA/headline
false.  It cannot be satisfied by the coverage lines or a partial PASS token.

The receipt admits only functional VCS and keeps timing, measured workload
cycles, speedup, PPA, power, energy, system speedup, and paper citation false.
Its contract hash transitively binds the exact tool/model/source set, and the
sealed result binds compile/sim logs and receipt.

## Issues and claim boundary

- **P0=0:** no bypass, second attempt, tool-count, PIPESTATUS, PASS-regex, or
  seal-integrity failure found.
- **P1=0:** all prerequisites for a separately authorized single attempt are
  source-complete.
- **P2=2:** the future receipt should directly record M944/M945/foundry/VCS
  hashes for easier standalone audit; OS helper binaries (`python3`, `rg`,
  `timeout`, coreutils) are not pinned, though their actions are fail closed.

Functional VCS remains false until the future run completes and its result is
independently hammered.  Timing, cycles, speedup, PPA, energy, system, and
headline claims remain false regardless of a functional PASS.
