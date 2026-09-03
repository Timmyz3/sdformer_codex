# M1944 — M1943 C2 exact-50-ps K8 pilot runner hammer

## Verdict

**FAIL / DO NOT AUTHOR M1945 RELEASE.** This was a static-only review: zero license queries, zero attempts, and zero EDA runs.

The runner correctly binds the exact M1939 Tcl (`0257e1e...`), M1938 failure review (`25d4fc6...`), and M1940 source review (`9b542017...`). It is K8-only, has one `lmstat` call site and one `dc_shell` call site, publishes the attempt before the license query, blocks same-UID EDA, uses an owned atomic lock, quarantines failures, has no retry path, passes exact 50-ps optimization/reporting settings, and exactly parses the planned M1945/M1946 authority chain. If a future review passes, its status literal must be exactly the one already encoded in the runner.

## Blocking finding

`reports/area_posthold.rpt` is only checked for existence/nonzero size. The runner never extracts its unique `Total cell area:` value and never independently compares that value with the frozen 137,363.9139348 µm² ceiling. The Tcl has a valid first-line area guard, but M1940 explicitly required the successor runner to independently parse `report_area` alongside setup, hold, and DRC reports. Therefore the runner-level fair +5% area contract is incomplete.

The next additive runner should retain the Tcl-side guard and add a strict parser that rejects missing, duplicate, nonnumeric, nonfinite, or nonpositive total-cell-area rows, applies the 137,363.9139348 µm² ceiling, and records the observed area and ratio in the receipt.

## Nonblocking forensic finding

The pre-license attempt marker says `license_queries=1` and `dc_shell_runs=1`, even though neither has occurred yet. Those are authorized maxima, not observed counts. A successor should label them as maxima and record actual launches separately in both success and failure receipts, closing the ambiguity already identified in M1938.

No M1945/M1946 artifact or EDA launch is authorized from this review.
