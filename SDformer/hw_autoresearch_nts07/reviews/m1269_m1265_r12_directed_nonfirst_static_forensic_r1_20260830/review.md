# M1269 — R12 `directed_nonfirst` static failure forensic

Date: 2026-08-30  
Mode: independent, read-only, static forensic  
Decision: **CONFIRM_M1268_TB_SEAM_FAILURE; M1162_NONFIRST_PSUM_LOGIC_IS_CORRECT; NO_C1_TRAFFIC_REBIND**  
Score: **99/100**  
P0/P1/P2: **0/0/1**

## Frozen evidence

- R12 TB SHA256: `e13d630f4cf2e2f7e0264dc2325218aee4cc580497be3b37deb1ff7a641ad302`
- M1162 SHA256: `639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595`
- M935 SHA256: `e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8`
- Failed sim log SHA256: `dbc340e87ea4cea4dda4e27f174cb5acb23fde324ea2380b630f127e84870dc0`
- The only runtime evidence is the consumed, quarantined M1265 result. No `simv`, VCS, EDA, GPU, or remote action was run for this forensic.

## What the fatal proves—and does not prove

R12 line 636 is one compound predicate:

```systemverilog
weight_fire_count != w0 + 1 || psum_fire_count != p0 || psum_req_valid
```

The sealed log reports only the shared fatal string at 109501 ps. It prints none of the three operands. Therefore the existing runtime evidence **cannot uniquely distinguish** a missing weight fire, an extra psum fire, or a still-high post-edge psum valid. Any report naming one operand as measured fact would overclaim.

Static ranking is nevertheless possible:

1. `weight_fire_count` is the least likely cause: the same child-seam helper already traversed the preceding weight-first and psum-first directed cases, and both cases reached the non-first task.
2. If the parent observed `issue_request_first=1` on the non-first handshake edge, both requests would fire because both readies are one. After the edge, `psum_request_accepted_q=1` suppresses `psum_req_valid`, so line 636 would most naturally fail through `psum_fire_count != p0` while the final `psum_req_valid` operand is already zero.
3. This is a hypothesis, not a measurement. The compound fatal makes the exact operand irrecoverable from the sealed text log.

The strongest predecessor control is R11: it used the same oracle and byte-identical M1162 but forced the M1162 parent connection. Its unique run emitted `PHASE_M1219R9_DIRECTED_COMPLETE`, and `cp_nonfirst` accumulated three matches before a later unrelated random-phase failure. R12 changes this stimulus boundary to the child output-variable seam. This strongly confirms M1268's classification of the new failure as a seam/stimulus problem, not an M1162 regression.

The zero-match `cp_nonfirst` summary does not resolve the ambiguity. The cover requires post-latch `request_active && !request_first`; the TB calls `$fatal` at the first edge plus 1 ps, before a subsequent assertion sampling edge can observe that state.

## Scheduling audit

The counter monitor uses blocking assignments in `always @(posedge clk_core)`. The task also wakes at that posedge but executes `#1ps` before reading the counters. Advancing simulation time by 1 ps guarantees that all active-region monitor statements, wrapper nonblocking assignments, and same-time combinational settling have completed. Consequently line 636 is not a same-active-region race between the task and the blocking counter monitor.

The child-seam force and both request-ready assignments occur at the preceding negedge, 1.5 ns before the sampled posedge. They are not introduced on the handshake edge.

The `UNIT_DELAY` define affects the foundry parent SRAM model. M1162 request-valid generation is plain RTL combinational logic and has no SRAM-data dependency. In the boundary-only case the M935 request-output seam is forced. Therefore SRAM UNIT_DELAY cannot create a non-first psum request or explain a missed request counter at this check.

The remaining plausible issue is boundary-testbench port-force observation/provenance, especially the value of `issue_request_first` as seen at the wrapper on the handshake edge. It is not evidence of a memory timing failure.

## M1162 first-principles result

M1162 lines 145–147 define:

```systemverilog
psum_read_request_valid = issue_request_valid
    && (request_active_q ? request_first_q : issue_request_first)
    && (!request_active_q || !psum_request_accepted_q);
```

For a legal non-first tuple:

- before latching: `request_active_q=0`, `issue_request_first=0`, so psum valid is zero;
- after latching: `request_active_q=1`, `request_first_q=0`, so psum valid remains zero;
- line 198 sets `psum_request_accepted_q = !issue_request_first || psum_request_fire_w`; for non-first this marks the absent psum requirement as already satisfied, but never emits a psum request.

M935 independently sets `active_ctx_first_q` on row admission and clears it after a non-last accepted source. Its arithmetic uses external `issue_psum_prior` only when `active_ctx_first_q` is true; later sources use resident `psum_acc_q`. Thus the architectural rule is consistent end to end: one psum read on the first source of a row, none on subsequent sources.

## Impact on C1 true traffic

No C1 real-traffic ledger, cycle count, memory-byte count, or 1.73–1.75x component opportunity requires rebinding from this failure. The failure occurs in synthetic boundary-only stimulus, and the governing M1162/M935 formulas prohibit a real non-first psum read.

R12 functional admission remains closed because the run did not reach PASS. This is an evidence-status limitation, not proof of extra architectural psum traffic.

## Minimal next step

Do not retry consumed M1265 and do not edit RTL. If another VCS attempt is justified, author a fresh TB-only successor that prioritizes **real M935 traffic rather than another cross-port force**:

1. move the existing byte-frozen `normal_m935_completion` phase ahead of synthetic seam phases so real M935 loads row 0 with mask `16'h0003` and naturally emits one first plus one non-first beat before any force;
2. make `serve_normal_beat(expect_first=0, beat_index=1)` the authoritative non-first check, preserving its exact expected psum-fire delta of zero and adding separately named diagnostic failures/snapshots rather than a compound fatal;
3. count `cov_nonfirst` from that real second beat and remove the synthetic child-output-force `directed_nonfirst` from admission, or explicitly demote it to non-admitting seam diagnostics;
4. if the child seam is retained for diagnosis, sample child and parent `issue_request_valid/first` immediately after the preceding negedge force, snapshot the pre-NBA handshake operands in the posedge monitor, and report every operand separately after `#1ps`;
5. retain boundary-only labels for all remaining synthetic cases and require a fresh independent source hammer/release before one new namespace.

This is the smallest admissible successor because it proves the architectural non-first behavior at the actual M935→M1162 boundary and avoids treating an output-variable force as real traffic. It does not alter C1 behavior or its traffic model.
