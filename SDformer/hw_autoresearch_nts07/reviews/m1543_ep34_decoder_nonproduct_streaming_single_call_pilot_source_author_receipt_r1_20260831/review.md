# M1543 ep34 decoder streaming single-call pilot source author receipt

Status: **PASS source authoring; independent hammer required; no actual pilot
and no production**.

M1543 binds the independently hammered M1539 request scheduler and fixes the
future calibration scope to canonical ep34 call 0: sample 10, D0, all ten
timesteps, under exactly `DENSE_TYPED_K8`, `BIT_EQUAL_SERVICE_K1X8`, and
`BIT_TYPED_K8`.  `PRODUCT_CAPTURE_TYPED_K8` remains blocked.

The implementation mmap's one bit-plane file and consumes requests one at a
time.  Dependency tokens are retired after each destination, while the bank
and port calendars, outstanding return queues, address/commit digests, cycle
state, and nine-tile weight cache remain live for the complete call.  Thus the
source does not obtain bounded memory by erasing physical conflicts or cache
misses.  A strict 8 GiB peak-RSS gate is checked during scheduling.

Current Python and CPython 3.6 both compile and pass the 13-attack test.  The
synthetic schedule conserves the common commit sequence and keeps the expected
dense, equal-service K1x8, and typed K8 distinctions.  Its cycle and request
counts are test vectors only.

The actual canonical payload was not opened by the tests and no pilot was
executed.  The CLI exposes only description, authority preflight, and a
synthetic self-test.  A different-author source hammer and a different-author
single-call launch hammer must pin this source before the one calibration call
may run.  No 120-call production attempt, transaction result, cycle, traffic,
speedup, energy, RTL, PPA, Table-A row, or paper claim is authorized.
