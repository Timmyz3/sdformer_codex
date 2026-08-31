# M861 decoder streaming/event-sweep source-author handoff

M861 is an additive implementation-only successor authorized by the sealed M857 failure audit. It leaves frozen M785/M768 transactions, ports, issue/return/commit equations, wait reasons, hashes, total cycles and cycle-class precedence unchanged.

The production-facing scheduler now consumes requests once, retains no request list and emits no `scheduled_requests` or `compressed_schedule` population. Ordered address/commit hashes and compressed-group count are updated online. Cycle classes are reconstructed from exact waiting/dependency/inflight/active half-open intervals with the frozen priority `active > dependency/inflight > weight > psum > memory > compute`.

The bounded reference path retained detail only for miters. A 512-request deterministic random DAG matched all 11 M768 result fields and complete produced-token readiness. A hand-authored E/D/I/R test exposed every one of the six priority classes. The adversarial pytest suite passed 14/14, including 1RW/1R1W, outstanding limits, same-cycle response-slot reuse, out-of-order/touching intervals and a bounded real M854 first-row prefix.

Controlled scale diagnostics completed 1K/10K/100K synthetic prefixes. The exact M854 first-row identity completed a 100K-request streaming prefix in 4.14 s with no detailed retention; a separate 1K real prefix old-vs-new miter passed. These are scalability diagnostics only, not production cycles or speedup.

The full 38,672,612-request first row and the 4,800-row production population were not run. A fresh independent source hammer must pass before any full-row gate; afterward the full first row, bounded-memory proof, population projection, possible exact-row sharding, true release and final-launch hammer remain mandatory. No VCS, DC, EDA, license, GPU, remote or training action was taken.
