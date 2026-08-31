# M293 binary patch-Conv bounded destination-group screen

The audit decodes 60 exact-binary M51 payloads for six patch Conv2d modules,
reconstructs 1,774,268,587 receptive-field source contributions, and binds the
frozen ep35 checkpoint weights.  Each output row is quantized to symmetric
INT8.  A source/destination-group task is eligible for omission only when every
weight magnitude in that destination group is at most beta.  Beta zero retains
the exact engine.

The six modules cover 172,321,077 of the frozen 620,302,905-cycle compute
envelope; two nonbinary patch modules with 27,099,543 cycles are excluded.
No low-budget standalone point crosses the 1.15 screening gate.  Group-4
beta-48 reaches only 1.11463x while removing 37.02% of weighted tasks.
Group-4 beta-64 reaches 1.22520x, but it removes 66.17% of weighted tasks and
is not accuracy-admitted.  Other first crossings are similarly aggressive.

The useful next test is not a higher patch-only beta.  It is a scope-correct
combination of group-4 beta-48 across the six patch modules and the ten binary
FC1 modules, because both use the same destination-group contract at a lower
budget.  That combined arithmetic must be independently checked before any
paired S10 run.

Every reported sensitivity is an ideal task-compaction opportunity.  There is
no router, bank-conflict-free executable schedule, accuracy, RTL, PPA, power,
energy, system-speedup or headline admission.
