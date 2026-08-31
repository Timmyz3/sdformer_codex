# M814 / M533 R18 coverage-repair source handoff

This is a source-only handoff. It does not authorize or report a VCS run.

R17 remains permanently consumed and `FAILED_DO_NOT_CITE`. M812 found no numeric, scoreboard, protocol, SVA assertion, or timing error before the normal-run fatal; the mandatory `pending_plus_forward` cover was zero and the old TB ping-pong count was only a prep handshake proxy.

R18 keeps M528 RTL r2 and SVA r2 byte-identical. TB r8 adds the single authorized six-row P0/A/P1/C0/C1/CA witness, counts real `dut.prep_active_q && dut.exec_active_q` overlap, and uses legal sink backpressure to keep both banks active. The normal 13-cover gate executes before P2, held-final, and all six protocol attacks; the final PASS token still requires all later phases.

The earlier four-row P0/P1/C0/C1 construction is withdrawn because P0 completion forwards its consumer too early. It must not be revived.

Author-only source checks passed under pinned Python 3.6.8: TB static proof, runner closure positive, three closure negatives, strict JSON and double-seal verification, and runner-owned rc86 pre-mkdir dry-run with zero VCS/license/compile/simv/result side effects. The exact actual `require_regular_sha` literal-call count is 83 (76 inherited plus seven new edges); an earlier 84 count incorrectly included the function definition and has been corrected throughout the frozen package.

Next step: a fresh independent source hammer must recompute every identity and rerun all source tests. It must not query a license, run VCS/simv, create a result, or author a launch release.
