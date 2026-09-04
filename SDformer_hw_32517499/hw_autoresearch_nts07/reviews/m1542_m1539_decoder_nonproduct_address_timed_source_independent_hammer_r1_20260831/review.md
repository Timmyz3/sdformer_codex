# M1542 independent hammer of the M1539 decoder replay source

Status: **PASS source hammer; distinct streaming-runner authoring may proceed;
production remains blocked**.

The M1539 source, test, contract, and sealed author receipt are byte-consistent.
Both the current interpreter and CPython 3.6 compile the source and test, and
the 12-attack test rejects all mutations.  Fast authority preflight and the
synthetic address-timed schedule pass independently.  The author receipt seal
also passes; its already sealed 122-member payload hash was therefore not
repeated by this source-only hammer.

The frozen common-resource digest is exactly
`64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10`.
An independent emitted-request audit observes exactly nine 1536-byte weight
tile refills, each followed by an eight-bank SRAM fill of 192 bytes per bank,
and 96-byte weight-vector reads.  Source fetch, descriptor traffic, the common
control round trip, psum read/write, dense output commit, bank calendars,
dependencies, and external service are present in the schedule.

Only `DENSE_TYPED_K8`, `BIT_EQUAL_SERVICE_K1X8`, and `BIT_TYPED_K8` are
admitted.  `PRODUCT_CAPTURE_TYPED_K8` is rejected by configuration validation
and remains blocked by M1526; `production_release()` independently rejects all
launches and the CLI exposes no production mode.  The synthetic cycle and byte
counts are test vectors only and are not a decoder result.

M1542 authorizes only a distinct, sealed, streaming runner to be authored and
hammered.  It authorizes no production execution, transaction result, traffic,
cycle, speedup, energy, RTL, PPA, Table-A row, or paper claim.
