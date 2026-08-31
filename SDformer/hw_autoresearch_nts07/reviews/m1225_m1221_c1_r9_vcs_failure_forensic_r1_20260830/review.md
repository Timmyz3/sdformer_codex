# M1225 read-only M1221 C1/R9 VCS failure forensic

Verdict: **the consumed M1221 attempt compiled successfully but is not a functional VCS PASS. The terminal normal-phase failure is primarily a TB service-ready/response-retirement ordering defect. Do not retry M1221 and do not modify DUT RTL or SVA.**

## Sealed execution boundary

The attempt identity is exact and says `automatic_retry=false`. The 94-member failure quarantine and its outer seal verify recursively. Compile exit codes are `0 0`; simv/tee also returned `0 0`, but the release runner correctly exited 31 because the required `NORMAL_M935_COMPLETE` phase token was absent. The sealed failure receipt therefore remains `functional_vcs_verified=false`.

## Phase-localized evidence

Directed, reset-pending, sticky-attack, service-attack, and the complete random phase finished. Every random index 0–23 has exactly one ENTER and COMPLETE token. The normal phase entered, and its clean-reset prep gate completed in zero cycles. The terminal fatal occurred at 8,113,500 ps after the normal `weight_req_valid` watchdog exhausted 2,000 cycles; the normal phase never completed.

The log contains 26 unmasked protocol assertion failures: 12 weight-request hold, 13 weight-response hold, and one psum-response hold. These failures already occur in legal random traffic and reappear around the normal serve boundary.

## Root-cause discrimination

“No issuable product row” is contradicted by two facts. The TB loads all 64 rows and gives row 0 mask `16'h0003`, while M935 selects and clears one residual source bit per accepted beat. More decisively, the first normal `serve_normal_beat` reached its response-drive code, which is reachable only after observing `weight_req_valid`. An issuable first product therefore existed.

The unsafe sequence is in the TB: `serve_normal_beat` leaves weight/psum request-ready high, waits for response acceptance, then waits an additional posedge before deasserting response-valid. This permits the next M935 beat to be captured or exposed while the previous service response is still presented, outside the second serve call. The response-hold failures are direct evidence of invalid service choreography. M1162's source would set its sticky boundary fault on a spurious/early held response, but the terminal path did not call the R9 state dump, so the fault bit was not directly logged. It must be described as a likely secondary consequence, not a proven primary DUT defect.

## Minimum additive repair gate

Author a new TB namespace only. Keep M935, M1162, R3 SVA, workloads, attacks, II=2, and the two-source normal row byte-exact. Retire each request-ready only at a race-free edge after the corresponding one-fire/suppression proof. Hold each response valid and payload stable until its exact ready/accept edge, then deassert at the immediately following negedge—remove the extra posedge. Before serving beat two, prove the prior response count advanced once, both response valids are zero, and the wrapper transaction retired. Add the full state dump to the normal request and response watchdogs.

A new static checker must reject restoration of the extra response posedge, missing ready retirement, unstable response service, missing normal timeout dump, and removal of the zero-SVA-failure gate. Only after a fresh source hammer may a fresh disjoint VCS release be authored. M1221 cannot be retried.

No VCS, EDA, GPU, remote action, or RTL mutation was performed by this forensic.
