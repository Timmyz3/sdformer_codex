# M1628 decoder compact L2 retained-ledger successor — author review

Status: **PASS source-only author repair; M1629 different-author review remains mandatory.**

M1628 repairs exactly the three M1620 P1 findings. First, it retains every accepted return and derives the 16-queue active projection from those returns and each destination's `last_cycle`; an earlier long return cannot be overwritten or disappear. The per-block psum-write readiness vector is also derived from accepted writes and therefore cannot move backward. Cache state now binds both its predecessor identity and the cumulative digest of accepted weight requests; clearing valid content or changing content without activity is rejected.

Second, every request is restricted to D0/module0/timestep0, the current destination, and output block 0..3. Request count, kind count, byte count, transaction-address digest, commit digest, dense commit population, port calendar and active returns are internal request-derived ledgers. A destination row is evidence only when it exactly matches those ledgers.

Third, `finish()` returns an immutable per-session receipt authenticated over configuration, resource, population, coverage and final state. The sealer additionally requires the exact completed miter owner. Bundle validation accepts only three registered, distinct, fresh receipts in frozen configuration order, requires dense coverage and one shared commit stream, then consumes all three receipts so replay fails.

The payload-free suite passed 18/18 under CPython 3.6 and 3.10. It kills all eight M1620 survivors plus direct-seal, cache predecessor/activity, port/bank, receipt clone/tag, session reuse/order, commit mismatch and replay attacks. This package opens no ep34 payload and executes no L2/L3, GPU or EDA work. It creates no performance or paper result. Only M1629 independent source review is authorized; M1633 release remains absent.

