# M871 / M863 C1 R21 launch-candidate hammer

Verdict: **PASS, 100/100, P0/P1/P2 = 0/0/0**.

The closed R21 candidate has exactly eleven typed authorization keys: one prospective VCS compile, one prospective simv run, and zero for every other run class. `launch_now` and `authorization_effective_now` remain false. Under both Python 3.6.8 and 3.10.16, all eleven missing-key mutations, all eleven bool/int type-confusion mutations, one extra-key mutation, two duplicate-key mutations, and four nonfinite/overflow mutations fail closed.

The fixed M864 source hammer is double-sealed PASS100. Runner, source contract, candidate, TB r10, RTL r2, SVA r2, macro adapter, binding plan, foundry UNIT_DELAY model, timeout binary, all source tests, and `docs/359` remain byte exact. R20 remains permanently `FAILED_DO_NOT_CITE` under M860's `TESTBENCH_POST_HANDSHAKE_READY_SAMPLE_SELF_ATTACK` classification.

Both interpreters pass the exact TB-delta test, synthetic event-order model and four negatives, full function-closure test and three negatives, external-command whitelist, fake fast/TERM/KILL/tee/orphan suite, and exact pre-mkdir runner dry-run. Each dry-run exits 86 at the live VCS/license boundary and creates no probe, license query, VCS compile, simv run, result, attempt, or quarantine.

This PASS authorizes only a separate author to create an inert, conditional release bound to this double-sealed review. It does not authorize VCS, simv, a license query, EDA, a workload, or any functional/cycle/performance/paper claim. A fresh independent final-launch hammer remains mandatory.
