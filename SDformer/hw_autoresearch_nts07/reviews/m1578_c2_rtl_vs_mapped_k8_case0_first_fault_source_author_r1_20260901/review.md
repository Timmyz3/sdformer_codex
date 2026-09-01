# M1578 C2 RTL-vs-mapped K8 case0 source author review

Decision: **PASS source-only; ready for an independent-author hammer.** This is not a simulation result and is not paper evidence.

The additive diagnostic places the frozen RTL matched wrapper in `ARCH_MODE=1` beside the frozen mapped `ARCH_MODE1` netlist. Both see the exact hard-wired M979 K8 case0 stimulus and the same external backpressure schedule. Each DUT owns a separate instance of the same reset-safe memory model, preventing one endpoint from changing the other's request or response schedule.

The first-fault trace preserves four-state information. It prints `protocol_error`, `numeric_overflow`, and `stale_response_seen` independently as 0/1/X, all eight endpoint fault bits, and six retained internal fault taps per DUT. Header, source, endpoint-request, memory-response, commit, and done events are logged edge by edge. The diagnostic records both the first fault/X cycle and first RTL-mapped difference cycle, then stops immediately.

Static validation passed for 16 ordered filelist entries and 20 frozen identities. Nine mutation tests rejected comment substitution, X-to-zero coercion, missing DUTs, missing first-difference/event reporting, prohibited runtime mechanisms, claim promotion, and duplicate JSON keys.

Boundary: no VCS compile, no `simv`, no UCLI, no initreg, no SAIF, no PTPX, and no attempt was consumed. Therefore this package proves only that the source is ready for independent review. It proves no RTL/mapped correctness, timing, power, PPA, speedup, or headline claim.

The independent hammer must decide whether to authorize the contract's single future compile plus single case0 simulation. It must not reuse the consumed M1502 binary.
