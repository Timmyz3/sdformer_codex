# M1364 independent blind review: FAIL_DO_NOT_CITE

M1363 correctly closes the 16 exact-contract holes reported by M1355. The
source-absent checker passed, all 23 directed tests passed before this M1364
namespace existed, all 16 prior exploit families were rejected, and 24
missing/malformed/uppercase external digest attacks failed closed.

The launch protocol nevertheless has one P0 reachability defect. The runner
requires the M1364/M1365/M1366 chain with `--mode runtime_present`, then runs
the same source-only test suite before consuming the attempt. Test 22 requires
those future paths to be absent. With this M1364 namespace present the suite
runs 23 tests and fails exactly `test_22_source_absent_and_residue` with
`AssertionError: future release residue`. A complete release chain therefore
cannot reach the attempt marker, license export, VCS, or simv.

The static runner controls themselves are sound: exactly one compile call,
one simulation call, two bounded timeouts, two collision gates, attempt
publication before license/tool reachability, and recursive failure quarantine.
No EDA or license command was invoked by this review.

The minimal additive successor should split source-only author tests from a
separately exact-pinned runtime-present launch suite. It must retain the 16
M1355 regressions and all one-shot safeguards, use a fresh namespace, and
receive new different-author source/final hammers. M1363 must remain immutable.
