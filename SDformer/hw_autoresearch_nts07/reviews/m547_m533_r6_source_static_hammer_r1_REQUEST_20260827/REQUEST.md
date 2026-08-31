# M547 / M533 r6 fresh source-static hammer request

This is a **read-only, zero-tool-run review**. Do not execute the runner, VCS, simv, any open-source simulator, DC, Formality, PT/PTPX, experiment, or remote job. Do not create a result directory.

The review must independently establish all of the following before returning 100/100 with P0/P1/P2 = 0:

1. `tb_m528_dead_write_only_1rw_product_capture_r4.sv` differs from r3 only on four lines inside `oracle_pack_row12`: the illegal local identifier `packed` and its three references are mechanically renamed `packed_row`. The module name, oracle behavior, stimuli, functional token, coverage token, P2 token, and attack tokens remain byte-identical otherwise.
2. Core r2, SVA r2, macro adapter, macro binding plan, foundry asset identities, and `docs/359` remain frozen at the hashes in `request.json` and the source contract.
3. The new result identity is absent. The consumed r3 partial remains untouched, unsealed, and permanently `FAILED_UNSEALED_DO_NOT_CITE` under the exact M544 review.
4. The r6 runner is syntactically sound under a read-only `bash -n` check and cannot reach `mkdir` because its fresh source review, candidate hammer, final release, and final-release hammer do not yet exist.
5. Every exit after the exact new result `mkdir`—including copied-precheck failure, collision, resource/monitor, VCS compile, simv, functional token, coverage/attack gate, or success-seal failure—must be forced by the EXIT trap into a double-sealed `FAILED_DO_NOT_CITE` receipt. A genuine success must also be double-sealed and remain functional-VCS-only until a receipt hammer.
6. Failure receipts must durably contain phase, runner exit code, exact child return code when applicable, monitor status, resource and collision presence/hashes, recursive pre-receipt artifact inventory/hashes, and immutable source hashes.
7. The launch-admission candidate is exactly `launch_now=false`; it does not authorize execution. A fresh candidate hammer, separately authored `launch_now=true` release, and fresh final-release hammer are mandatory.

Required PASS schema/status: `m547_m533_r6_source_static_hammer_v1` / `PASS_M547_M533_R6_SOURCE_STATIC_HAMMER`. The review must be double sealed and bind the exact runner, source contract, TB r4, and candidate hashes. Any unresolved failure-terminal or parser-readiness defect is P1 or higher and must fail the review.
