# M2216 independent M2215 failure/result hammer

Verdict: `PASS_M2216_FAILURE_DIAGNOSIS__NEW_IDENTITY_PARSE_ONLY_SUCCESSOR_SUPPORTED`.

Diagnosis score: 98/100. Remaining frozen-producer defects: P0/P1/P2 = 0/1/2. This score concerns failure diagnosis and bounded recovery, not paper readiness. The P1 is the consumed M2215 parser; its failed status remains in force.

## Finding and recovery

M2215 compiled and ran successfully at the raw-tool level. The sole production postprocessing failure is parser line 40, which searches `Chronologic VCS simulator copyright` in the compilation log. That banner is emitted by `simv`; the compilation log instead has `Chronologic VCS (TM)`, the matching `V-2023.12-SP1_Full64` version, all seven modules completed, and a completed link. The raw simulation has return code 0 and exactly one expected PASS and cover ledger. All remaining old-parser predicates independently pass.

A read-only invocation of the original pinned parser with output `/dev/null` reproduced exactly that line-40 `AssertionError` before any output write. The quarantine and consumed-attempt directory, M2214 review, and all ten M2213 inventory entries were revalidated afterward. No input was changed. The original `parser.log` is empty because the runner captured stdout only; the traceback is reconstructed by this audit, not misrepresented as a quarantined original traceback.

A new parse-only identity can recover the sealed raw evidence. It should pin the exhaustive quarantine/attempt seals and source review, validate the actual compiler and runtime banners in their respective logs, retain all numerical/protocol checks, add nonzero SVA-cover checks, record both parser streams, and produce a separately sealed result referencing the failed raw producer. It must retain `FAILED_OR_INCOMPLETE_DO_NOT_CITE` for M2215. A further VCS invocation is unnecessary for this failure.

## Raw observations, pending successor result admission

| Observation | Ordinary | Post-read | Pre-read |
|---|---:|---:|---:|
| Accepted bank reads | 2304 | 2304 | 576 |
| Raw workload cycles | 3386 | 3386 | 1119 |
| Commits | 24 | 24 | 24 |
| Signed products | 4608 | 4608 | 4608 |
| Golden mismatches | 0 | 0 | 0 |

Post-read and pre-read have the same B4 group-major schedule and LRU4 residency; post-read performs the real transaction even on a resident hit. Their read difference is 1728, exactly the independently observed 1728 accepted post-read-hit bank requests and responses. The corresponding 216 bundle requests, responses and identity accepts agree. All three SVA covers are nonzero: request 552, response 1932, terminal commit 4. The ordinary token-major order is a separate baseline; the causal matched pair is post-read/pre-read.

This is one deterministic four-context, six-group directed workload designed to exercise LRU4 capacity pressure. Its cycles are raw diagnostics. They do not establish population speedup, SRAM energy or mapped performance. A 75% directed request reduction can be admitted through the successor with these boundaries; it does not automatically establish a 75% energy reduction. The additive post-read counters also remain a barrier to a matched-area or matched-power comparison unless equalized or separately stripped.

## `context` warning

All 26 `KUAI` warnings point to the TB's task parameter or loop variable `context`. None points to the DUT or SVA module. The pinned VCS release accepted those identifiers, elaborated and linked the complete design, then executed the four context identities and all six slices with unique tagged commits and a full golden scoreboard. This warning is a portability defect, not evidence of a failed or skipped current simulation. Rename it in the next TB identity when that TB otherwise changes; do not modify the frozen source or rerun this experiment solely for that warning.

The only other compile warning is the already explicit unsupported Linux-version warning. Runtime ASLR save/restore information is not a failure. No compile error, runtime fatal, timeout, assertion failure or incomplete finish was found.

## Boundaries

No new VCS, license, synthesis, timing, power, GPU or Git operation was performed by this review. One CPU parser replay was used for diagnosis. No source, raw log, quarantine, consumed attempt or `docs/359` was modified. There is no RTL result admission in this failure review itself; review the separately identified parse-only receipt before citation. No additional RTL experiment is required to diagnose or recover this parsing failure.
