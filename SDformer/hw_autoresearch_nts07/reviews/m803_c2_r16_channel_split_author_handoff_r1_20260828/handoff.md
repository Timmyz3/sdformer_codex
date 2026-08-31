# M803/C2 R16 author handoff

M803 is a source-only additive repair of the M800 K8 TIM-209 root cause. It
does not alter frozen M490, M499, M519 K1/K1x8, or docs/359. The new adapter
retains M490's same-cycle response-retire/request-reuse expression and migrates
only M499 R5's independent request/response channel enables plus state-update
ordering. The new K8 top retains the frozen M519 arithmetic/control core; the
matched shell still binds frozen M519 K1 and K1x8.

The attack TB/SVA now hard-gate multi-bank request-fault plus legal completion,
illegal-response closure of both channels, pending/backpressure behavior,
same-slot same-cycle reuse, sticky/reset recovery, and bundle/bank conservation.
The full-workload TB hard-gates the five frozen K8/K1x8 cycle pairs and all old
numeric, protocol, stall, and coverage counters. Frozen K1-vs-K1x8 is only
source-SHA-bound and was not duplicated or changed.

Author-only source checks passed under Python 3.6.8. The exact-SHA runner's
wrong-SHA negative failed before trace creation; its positive source dry-run
stopped with rc=86 immediately before the live VCS/license boundary with zero
VCS, license, simv, result, or attempt side effects. No VCS/DC/EDA execution is
authorized by this handoff. A fresh independent source hammer is the only next
step.
