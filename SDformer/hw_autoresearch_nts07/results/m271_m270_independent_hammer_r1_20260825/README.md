# M271 independent hammer review of M270

Verdict: **93/100, GO for closure of M268's malformed-IDLE-header P1 only**.
There are no P0 findings.  The exact M270 author identity and r2 VCS seal verify,
and a fresh Synopsys VCS compile at independent seed `2710825` reproduces the
canonical `6 tiles / 26 descriptors / 22 clean cycle checks / 40 commits / 7
attacks` PASS with all six stall classes nonzero and no assertion failure.

The independent testbench uses the exact M270 RTL/SVA but not the author TB.  At
seed `2711825` it retires 3,081 descriptors, including a complete legal 3,072-
descriptor traversal from factor address `0xff400` through `0xfffff`.  It also
checks the legal `count=1, base=0xfffff` edge, exact `6+3*popcount` for
popcounts one through eight, four cycles of empty-done backpressure, four
tagged sticky malformed-header attacks, and `abort_ready=1` on the first visible
abort cycle.  The result is zero non-abort side-effect mismatch and zero SVA
failure.

The M262 performance numbers are byte-for-byte numerically unchanged:
`110840148144 / 66282442128 = 1.6722399565476673x`, and
`12126285024 / 4700000688 = 2.580060265727348x`.  M270 changes malformed-header
legality/tag capture only; it adds no speedup and does not alter clean datapath
or state-cycle recurrence.

One inherited interface caveat remains.  `header_valid` while busy is treated
as a protocol fault, so the header input is request-only-when-ready rather than
a decoupled ready/valid queue.  A development stimulus that delayed legal
`header_valid` deassertion by 0.2 ns exposed the corresponding combinational
`protocol_error` pulse.  The sealed source uses handshake-edge NBA deassertion.
Document/assert that source contract or add a one-entry header skid before
integration.

The pre-existing performance P1 also remains open: `1.672240x` and `2.580060x`
are small-width aggregate lifecycle mappings without address-timed SRAM,
queueing, bank conflicts or physical timing.  No 96-lane, full-trace RTL,
complete-FC1/FFN, system, headline, DC, energy or paper-PPA claim is admitted.
No DC was run and docs/359 was not modified.

Only `canonical_recompile_r7_seed2710825` and
`independent_attack_r7_seed2711825` are sealed evidence.  Earlier `r1` through
`r6` directories are retained as unsealed testbench-development audit trail and
must not be cited.
