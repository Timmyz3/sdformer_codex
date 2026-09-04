# M1301 claim-only authority successor — author receipt

M1301 restores the exact seven-key M1292 false claim map and removes the
out-of-scope `paper_ppa_ready` key. The new zero-argument execution entry pins
M1297 source/test/contract and M1298's double seal, validates the exact claims,
and only then delegates to the unchanged M1297 fd-bound executor.

M1297 `PRODUCTION_POLICY` is reused as the same object. Interpreter identity,
fd probing, `/proc/self/fd` execution, pass-fds, eleven snapshots, three sealed
sources, candidate selection, F1--F4, E0--E8, O_EXCL and no-retry semantics are
unchanged.

Local temporary-fixture tests pass 11/11. This author receipt does not authorize
transfer, remote preflight or production. A fresh different-author hammer is
required. No remote, production, checkpoint, GPU or EDA action occurred.
