# M965 | M964 C1 macro-aware DC launch-release audit

Verdict: `GO`, score 98/100, P0=0/P1=0/P2=2. The exact M964 release
authorizes one and only one M962 macro-aware setup/area DC attempt. M965 did not
launch it.

## Admission result

The M964 payload and both SHA sidecars validate. It pins the exact M962 runner,
source contract, Tcl, SDC, two-file source list, M935 RTL, nine-macro wrapper,
`docs/359`, and the recursively sealed M963 review. M963's narrowly superseding
functional admission remains exact: only the single M923 wrong-parent negative
test is accepted; unexpected assertions and fatal/errors have zero tolerance.

The runner independently requires the M963 and M964 seals, checks the caller's
runner and release SHA environment pins, rejects a consumed/colliding namespace,
blocks same-UID DC processes, checks live resources and licenses, then consumes
the attempt before invoking DC. One `compile_ultra` is allowed; incremental
compile, false paths, multicycle paths, disabled timing arcs, case analysis and
path-specific delay exceptions are absent and forbidden.

The canonical result policy is fail-closed but does not discard a valid negative
timing result. A complete 3 ns setup violation is published as a double-sealed
negative with WNS, TNS, violating-path count and top-100 report. Only tool/link,
macro-binding or incomplete-report failures enter quarantine. A setup pass is
required before any positive timing claim.

At audit time the result, attempt, lock, work and failure-prefix namespaces were
all absent; same-UID DC process count was zero. MemAvailable was approximately
398.7 GiB, SwapFree 51.7 GiB, and commit headroom 110.7 GiB, all above the frozen
runner gates.

## Required launch identity

The caller must use:

- `M962_EXPECTED_DC_RUNNER_SHA256=7ec1138696c40b923d6841dc21749aed35e93da266e00910b6715278c51da7fd`
- `M962_EXPECTED_DC_RELEASE_SHA256=9d47a2c204bf89204ec124214ed64935a8fcc401d2ed34f5a881006f8c3bb1d2`

and invoke the exact no-argument M962 runner. Any changed identity or failed live
gate must STOP without consuming a DC attempt.

## P2 and claim boundary

- The frozen memory floor is `100663296 KiB = 96 GiB` (about 103.1 GB decimal),
  not 100 GiB. It must be labeled 96 GiB.
- Live resources and collision state are snapshots; the runner must recheck all
  gates at launch.

This release is not a timing result. Timing, hold, cycles, speedup, PPA, power,
energy, system and paper claims remain false until the separately hammered M962
result exists. M965 ran no DC, VCS, PT, PTPX, Formality, GPU or remote workload
and did not modify RTL, source packages or `docs/359`.
