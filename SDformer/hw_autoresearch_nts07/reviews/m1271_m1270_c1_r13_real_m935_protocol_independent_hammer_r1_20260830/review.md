# M1271 — M1270/R13 real-M935 source independent hammer

Date: 2026-08-30  
Mode: source-only independent hammer; no VCS/simv/EDA/GPU/remote  
Decision: **SOURCE_NO_GO; checker-only repair and fresh independent re-hammer required**  
Score: **82/100**  
P0/P1/P2: **0/4/1**

## Bottom line

The checked-in R13 TB itself is manually clean and its intended workload is
structurally credible.  Public prep loads one 64-row task with only row 0
nonempty (`mask=0x0003`).  Frozen M935 therefore selects source 0 as a first
beat, clears bit 0 after acceptance, and naturally selects source 1 as a
non-first/last beat.  No parent or child `issue_request_*` force, release,
continuous assignment, procedural assignment, alias, bind, or DPI/VPI override
was found in the actual TB.

The service timing is also source-level coherent: requests are enabled on a
negedge and sampled once on the following posedge, then both ready inputs are
removed at the next negedge.  Responses are driven at negedges and held through
the observed accept.  The first beat holds weight alone across two complete
sampled cycles before psum is exposed; the second beat never exposes a psum
response.  Exact request/response deltas and overshoot checks prevent duplicate
request accounting.  Completion requires two issue accepts, one psum commit,
one row completion, one task completion, epoch `0x9001`, II >= 2, and zero
boundary/core/M935/service faults.  Every `$fatal` in the actual TB is routed
through the operand-printing oracle.

However, the source checker is fail-open under four independent classes of
claim-destroying mutation.  Six mutations were executed through
`check_text()` and all six were incorrectly accepted:

| P1 class | Accepted mutation | Why it invalidates the gate |
|---|---|---|
| Runtime token spoof | Comment out the real PASS display; comment out the real PHASE-DONE display | Raw-text `$display` matching counts commented statements as executable tokens |
| Workload disappearance | Comment out `real_m935_completion()`; wrap both real beat calls in `if (1'b0)` | Raw substring checks do not prove that the integrated workload executes |
| Request override blind spot | Add bare `issue_request_first = 1'b0;` in the initial block | Assignment regex covers only `dut[.child].issue_request_*`, not the connected top-level issue signals |
| Operand-printer disappearance | Block-comment the oracle `$display` statement | Oracle-body validation searches raw text, so a fatal can remain with no preceding operand record |

These are P1 because a future mutated TB can receive a static PASS while
removing the only runtime workload/token/diagnostic evidence or directly
driving an issue-request object.  The current clean TB does not cure a
fail-open admission checker.  Therefore M1270 is not SOURCE_GO, separate
release authoring is not authorized, and VCS remains prohibited.

## Required repair boundary

Repair only the checker/tests; keep the R13 TB, contract, frozen RTL/SVA and
`docs/359` byte-exact.  The repaired checker must:

1. locate executable `$display` statements in comment/string-stripped code and
   require exact PHASE/PASS statements in the authoritative initial flow;
2. prove the completion call and two beat calls are executable and not under a
   statically false guard/comment/string/dormant task;
3. reject assignments/forces/releases to both bare connected
   `issue_request_*` signals and parent/child hierarchical forms;
4. prove an executable operand `$display` and `$fflush` dominate the sole
   `$fatal` inside the oracle;
5. add all six mutations above to the test corpus.

A fresh different-author hammer is required after repair.  This review does
not authorize release authoring, VCS, or any functional/timing/performance
claim.

## Seal and identity audit

- Author static checker and tests currently report PASS (16/16), but are
  insufficient for admission for the reasons above.
- The author `SHA256SUMS` and outer seal both verify.
- Frozen M528/M935/M1162/R3-SVA hashes match the contract.
- `docs/359` remains
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
- M1265 remains consumed/quarantined and was not touched.

P2: the current source checker reports counts such as `real_beats=2` from
static literals rather than a control-flow proof.  This becomes harmless once
the P1 executable-flow requirements above are enforced, but the result schema
should be renamed to make the distinction explicit.
