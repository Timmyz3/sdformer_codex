# M1283 receipt-blind hammer of M1278

**STOP, score 80/100; P0/P1/P2 = 0/3/0.** Eleven malformed-state attacks were rejected, including forged live 120 rows, missing PID/lock/result/work transitions, bad seal, final-checkpoint identity, duplicate/nonfinite JSON and a boolean call ordinal.

Three P1 escapes remain: `true` passes the integer-one attempt check; `publish_annex` accepts a mutated final-checkpoint/Table-A/system payload; and that publisher has no completed-gate capability, so a direct caller can write from an incomplete synthetic state. The zero-argument main path did not itself promote a claim, hence no P0. Repair scalar types and add last-mile validation/capability binding before another hammer.

This hammer was receipt-blind and synthetic-only. It did not open the growing live work file, run M1278 live preflight, or launch replay/EDA/GPU/remote work.
