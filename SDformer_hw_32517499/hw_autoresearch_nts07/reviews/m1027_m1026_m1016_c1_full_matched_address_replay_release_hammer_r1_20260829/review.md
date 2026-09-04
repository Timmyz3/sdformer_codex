# M1027 independent M1026/M1028 execution-chain hammer

Verdict: `PASS_M1027_M1026_M1016_C1_FULL_REPLAY_RELEASE_HAMMER`; one future CPU-only M1028 attempt is authorized.

The M1025 P1 is closed. M1028 hardcodes the M1016 contract and engine, M1025 source hammer, M1026 release and this M1027 hammer. The caller supplies exact pins for the runner and the two hammer outer seals but cannot select an authority path. M1026 binds the runner, contract, engine, checker, tests, M1025 identities and M410 ledger identity.

Seventeen release mutations were rejected, covering identity, launch/max-attempt, caller-path, retry and forbidden-tool fields. Two live preflight faults—missing caller environment and a wrong runner SHA—both exited with code 3 before creating an attempt or result. The one-shot code has atomic attempt consumption, failure quarantine, recursive seals and a no-overwrite final publish.

This hammer did not execute M1028 or read/replay the full 51.84M rows. It ran no EDA, GPU or remote tool. A future raw result must remain pending an independent result hammer; this release admits neither 214,912B, cycles nor speedup.
