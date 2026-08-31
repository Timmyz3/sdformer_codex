# M281 independent hammer of M276 BN coefficient protocol/II

M281 independently admits M276 only as an exact-SHA M235 coefficient-engine
protocol and intrinsic-service-interval milestone.  It does not admit full
dynamic BN, a new speedup, mapped PPA, energy, system performance, or a paper
headline.

The independent evidence consists of:

- a dependency-free reparse and RTL-order recomputation of all 220,800 frozen
  rows;
- a fresh Synopsys VCS replay of the complete M276 test, assertions and
  production RTL;
- a separate Synopsys VCS bench for clean 8/9 timing, held-request turnaround,
  result stability, and illegal-zero attacks while idle, mid-compute and with a
  pending result;
- a wrong-expected-RTL-SHA preflight that exits 10 before VCS starts;
- nested SHA manifests and seals.

Scoped result: 220,800 request accepts, 220,800 result accepts, zero integer
output mismatch, first-result latency 8 cycles, intrinsic unstalled accept II 9
cycles, 220,799 held successor requests, five result-stall cycles and zero
assertion failures.  At 333.333 MHz the isolated ceiling is 37.04 million
coefficient sets/s; that value is not activation throughput or a system speedup.

Primary review: `m281_m276_bn_protocol_ii_independent_hammer_review_r1.json`.
Reproducible receipt: `replay_exact/m281_m276_independent_vcs_receipt_r1.json`.
Production RTL remained SHA256
`ec0bf05540433ecfc436eac63b41a4cecf4cc53b46533f2fd4f44c7eb70bd611`.
`docs/359_DATE终局冻结_20260813.md` remained SHA256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
