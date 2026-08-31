# M259 independent hammer of M241r2

Score: **91/100**. Severity: **P0=0, P1=3, P2=4**.

Verdict: **GO for the bounded standalone eight-lane elastic protocol repair.**
The independent exact-SHA VCS compile and simulation pass, and the assertion
report is byte-identical to the formal sealed run. A second same-seed replay is
deterministic. A wrong RTL SHA makes the exact production runner exit `10`
before VCS compile and leaves a do-not-cite failure marker.

Independent vector decoding confirms 126 ordered full4 descriptors, 504 writes
and 4,032 lane checks per latency mode. Fixed 1/2/3-cycle modes and a varying
1-to-3-cycle mode pass with zero mismatch. Stale responses have zero accepts;
overflow has zero success commit and zero writes, discards two accepted younger
tokens, and recovers after reset. Lazy valid, zero forwarding payload and no
M149 instance are preserved.

Two claim corrections are required:

- The fourth mode is deterministic, not randomized. The TB has no RNG, and a
  different seed produces the same PASS line and byte-identical assertion
  report.
- Testing 1/2/3 and one varying schedule does not prove arbitrary finite
  response latency; the admitted scope is bounded elasticity.

Production SVA checks response masks and every data lane under stall but omits
the complete response identity tag. M259 adds a review-only overlay covering
all weight and accumulator response tag fields; both stall covers pass with no
assertion failure. That overlay should be promoted into production SVA.

Full96, a complete trace, selected SRAM, DC/STA, energy, measured cycle speedup,
physical/system speedup, paper PPA and headline remain false.
