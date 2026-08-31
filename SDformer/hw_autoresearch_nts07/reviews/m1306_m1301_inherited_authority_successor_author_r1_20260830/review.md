# M1306 inherited-authority successor — author receipt

M1306 fixes only M1303 P1. After its own M1301/M1303 seal gate and frozen
M1301's exact-claim gate, it calls frozen
`M1297.M.verify_frozen_authorities()` exactly once, then delegates to unchanged
M1297 `execute_once`.

The injected inherited-gate failure produced one preflight call, zero delegate
calls and no attempt. M1297/M1301 policy, interpreter entity, retained fd,
`/proc/self/fd` child, exact four pass-fds, eleven snapshots, three sealed
sources, F1--F4, E0--E8, O_EXCL/no-retry and seven false claims are unchanged.

Local temporary-fixture tests pass 11/11. A fresh different-author hammer is
required. This receipt does not authorize transfer, remote preflight, attempt
consumption or production.
