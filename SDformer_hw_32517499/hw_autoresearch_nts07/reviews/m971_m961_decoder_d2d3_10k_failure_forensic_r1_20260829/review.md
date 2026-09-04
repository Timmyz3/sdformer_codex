# M971 — M961 D2/D3 10K consumed-attempt failure forensic

## Verdict

`FAIL_M961_CONSUMED_NO_RESULT_NO_QUARANTINE__DRIVER_PROJECTION_CONTRACT_BUG__NO_RERUN`

M961 is permanently consumed and must not be rerun, restored, renamed, or
aliased.  Its frozen M946/M896 exact scheduler path is not disproved.  The
failure is in M961's post-row projection contract: it labels compressed input
bytes as source-fetch request counts and consequently asserts that a real 10K
prefix is one compressed source-fetch transaction.

The failure path also exposes a separate evidence-retention defect.  M961
creates `output_stage` only after both rows and both projections succeed.  The
exception therefore occurs before a stage exists; the shell cleanup has
nothing to move.  At audit time the sealed attempt exists, while canonical
result, result stage, quarantine, and persisted traceback are all absent.
There is no quarantine double seal to validate.  This absence is a P0 finding,
not evidence that execution did not start.

M971 ran no decoder prefix, M961 runner, EDA, GPU, remote, or network command.
It changed no M961/M946/M896 source, release, threshold, attempt, payload, or
docs/359.

## Frozen execution identity

- M961 source contract SHA256:
  `966cbb77aee1c03df2ac6dc8deb8ee707ee7560f89d1b9daf368335de86b5420`
- M969 release SHA256:
  `8029a14bbe0c6211c6c21997b5ce93ab34da24271b521f06b670c560b548d691`
- M961 driver SHA256:
  `c997626a0eff58b4824d534335c9bc0627d8408f0f8e14a81e490bfc8895c54a`
- M961 runner SHA256:
  `2ba4d4c8fb5b7ec90943c9ed71a60747a9296f880b8bdb09fc5620b1d41c009a`
- Frozen M946 SHA256:
  `0ffd1ee810f24d1a95b0df33ffe8eae43240920e12a2fccb86c947d2be51b6ac`
- Frozen M896 SHA256:
  `c877f70849eb254bd5b227c79e8120773a9c48aa7405a2e6564b7eb4647aae39`
- M970 release-hammer review SHA256:
  `c87c83efc1bfa48fdafe199102fed247aa237834a01fea8f101fc9ebc167c071`
- docs/359 SHA256 remains
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

The M961 attempt directory recursively verifies.  Its receipt SHA256 is
`2cbee7962a1c59b14062e003d2bfbaba2fd269a26ff91b29fea98d35378526d4`,
status is `CONSUMED_BEFORE_EXECUTION`, and its sealed release/release-hammer
bindings match M969/M970.  This establishes a consumed one-shot namespace; it
does not establish a completed or citable 10K result.

## Exact failing field

The frozen decoder geometry is:

| layer | `Cin×H×W` | source bytes | 192-B source-fetch requests |
|---|---:|---:|---:|
| D2 | `386×60×80` | 231,600 | 1,207 |
| D3 | `194×120×160` | 465,600 | 2,425 |

M785 computes `source_bytes = ceil(Cin×H×W/8)` and passes that byte count to
`_source_read`; `_external_transaction` then computes the request count as
`ceil(byte_count/192)`.  M961 instead freezes 231,600 and 465,600 in a variable
named `SOURCE_FETCH_REQUESTS`.  Those values are bytes, not requests.

Consequences are deterministic without replay:

1. The M950 real 1K D2/D3 checks remain inside the first source-fetch
   transaction, so both legitimately report `compressed_transactions=1`.
2. A 10K D2 prefix consumes all 1,207 source-fetch requests and continues for
   8,793 requests; D3 similarly continues for 7,575 requests.
3. Therefore `exact_miter.compressed_transaction_count == 1` at M961 driver
   line 374 is false for both real 10K rows.  Python short-circuit order makes
   this the first failing field after the exact status and expanded-request
   conservation checks.
4. The later predicate `100000 < SOURCE_FETCH_REQUESTS[layer]` compares
   requests against bytes and is invalid even though its integer comparison
   evaluates true.

The exact exception is raised by `project_100k`, after `rows =
[M946.run_bounded_prefix(... D2 ...), M946.run_bounded_prefix(... D3 ...)]`
has returned.  M946's `exact_schedule` compares all frozen M890/M896 fields
before returning a row.  Thus both in-memory row computations crossed the
M768/M861/M890/M896 exact miter.  The failure does not show a numeric or cycle
mismatch in M946/M896.  The rows were never written or sealed, so their
runtime metrics must not be reconstructed or cited from memory.

## Failure-retention audit

At M971 audit time:

- sealed consumed attempt: present and recursively valid;
- canonical M961 result: absent;
- M961 output-stage directory: absent;
- M961 quarantine directory: absent;
- persisted stdout/stderr or traceback: absent;
- active M961 execution process: absent.

The runner's cleanup only moves `result_stage` when that directory already
exists.  The driver creates it after projection and result construction.
Because projection fails first, neither cleanup nor its double-seal policy can
operate.  The terminal traceback reported for M961 is consistent with the
source line above, but no checksum-bound traceback exists on disk; M971 does
not upgrade session text into result evidence.

## Severity

- **P0-1 — consumed identity without failure artifact:** one-shot attempt is
  burned, but no canonical failure receipt, stage, quarantine, log, or
  traceback survives.
- **P0-2 — wrong-unit released boundary:** byte counts were admitted as request
  counts, making the released `SOURCE_FETCH_ONLY` and one-transaction 10K
  predicates false.
- **P1-1 — missing real-10K predicate coverage:** M950 tested real D2/D3 only
  at 1K and synthetic traffic at 10K; M968/M970 did not exercise the exact
  released real-10K projection predicate.
- **P1-2 — pair-atomic late staging:** both exact rows are held only in memory
  until both projections pass, so one projection fault discards evidence for
  both rows.
- **P2-1 — misleading schema vocabulary:** `d2_source_fetch_requests=231600`
  and `d3_source_fetch_requests=465600` should have been byte fields.
- **P2-2 — self-test mirrors the assumption:** M961's fake row hardcodes one
  compressed transaction and cannot expose the byte/request boundary error.

## Unique legal successor

Only a fresh additive identity is legal.  It must not rerun or relabel M961.
The successor must keep M946/M896 and all numerical/cycle semantics frozen,
while changing only M961's projection/evidence wrapper:

1. derive source-fetch request count from the generated first transaction's
   `count`, and name byte and request fields separately;
2. remove the false `compressed_transaction_count == 1` and
   `SOURCE_FETCH_ONLY` assumptions; report the measured 10K transaction mix
   and first-commit boundary without prewriting their values;
3. create a per-row stage and log before each row starts, seal D2 immediately,
   then seal D3; on any exception write and double-seal exit code, traceback,
   completed-row identities, and partial scope before quarantine;
4. require a new independent source hammer and separate release for exactly
   one new D2/D3 10K attempt; prohibit 100K until that result receives an
   independent result hammer.

No M961 retry, threshold relaxation, M946/M896 edit, automatic 100K, full-row,
decoder-complete, Table-A, system-speedup, or paper claim is legal.

