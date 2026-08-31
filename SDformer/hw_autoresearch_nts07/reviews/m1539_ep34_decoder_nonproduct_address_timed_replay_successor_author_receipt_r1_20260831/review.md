# M1539 ep34 decoder non-product replay source author receipt

Status: **PASS source/test/full identity preflight; no production**.

M1539 accepts exactly `DENSE_TYPED_K8`, `BIT_EQUAL_SERVICE_K1X8`, and
`BIT_TYPED_K8`.  `PRODUCT_CAPTURE_TYPED_K8` is rejected and remains blocked by
M1526.  The common 96-lane/Acc24/3-ns/240-KiB/192-B-per-cycle resource digest
is byte-identical to M1525.  The source explicitly charges source/descriptor
traffic, the common control round trip, nine-tile weight refill and SRAM fill,
96-byte source weight vectors, psum RMW, dense commits, banks, dependencies,
and fixed latency.

CPython 3.6 compilation and the 12-attack author test pass.  Full hashing of
the actual M1521 directory accepted all 122 sealed members and the M1527 and
M1536 authorities.  No production command was exposed or executed.

The remaining P1 is deliberate: a different-author source hammer and a
distinct sealed streaming runner are required before the only production
attempt.  Therefore this receipt admits no transactions, cycles, traffic,
speedup, energy, PPA, Table A row, or paper result.
