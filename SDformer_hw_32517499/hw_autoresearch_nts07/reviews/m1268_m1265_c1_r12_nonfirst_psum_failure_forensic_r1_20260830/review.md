# M1268 — M1265/R12 non-first psum failure forensic

Date: 2026-08-30  
Mode: independent, read-only forensic; no VCS/simv/EDA rerun  
Decision: **TB-side stimulus/seam failure; not evidence of a real RTL extra-psum request**  
Score: **99/100**  
P0/P1/P2: **0/0/1**

## Scope and frozen evidence

This review reads only the quarantined unique M1265 result, frozen R11/R12 TBs,
M528/M935/M1162 RTL, R3 SVA, and existing sealed schedule/contract evidence. It
does not edit any source, authorize a retry, run `simv`, run VCS/DC/PT, or touch
`docs/359` (observed SHA256 remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`).

The unique M1265 compile/elaboration/link succeeded. The simulation stopped at
109501 ps in `directed_nonfirst`, R12 TB line 638. The exact predicate is a
three-way OR:

```systemverilog
weight_fire_count != w0 + 1 || psum_fire_count != p0 || psum_req_valid
```

The log emits only `non-first beat issued a psum request`; it does **not** emit
the three operand values. Therefore the sealed failure does not establish that
a psum request fired, and cannot distinguish a weight-count failure from a
psum-count failure or post-edge psum-valid observation.

## First-principles control-cone proof

For the intended R12 tuple, `issue_request_valid=1` and
`issue_request_first=0`. M1162 defines:

```systemverilog
psum_read_request_valid = issue_request_valid
    && (request_active_q ? request_first_q : issue_request_first)
    && (!request_active_q || !psum_request_accepted_q);
```

Hence:

1. Before latch, the middle term is the intended `issue_request_first=0`.
2. On the accepting edge, M1162 latches `request_first_q=0` and sets
   `psum_request_accepted_q = !issue_request_first || psum_request_fire_w = 1`.
3. After latch, both `request_first_q=0` and the accepted suppression make the
   psum request invalid.

There is no alternative RTL term that can manufacture a psum read. M528 only
implements the nine parent-scratch macros and is outside this request-control
cone. M935 supplies the `first` metadata, but R12 deliberately forces M935's
*child output variable* rather than the M1162 parent connection.

Thus, if the intended `first=0` actually reaches M1162, an extra psum request
is Boolean-impossible in this RTL. The failure necessarily means that the
intended tuple was not established/observed as assumed, or that another member
of the compound TB predicate failed. It is not proof of an M1162 design bug.

## Why the R12 seam is the failing boundary

R11 and R12 share the same oracle and M1162 RTL. R11 forces the parent
`dut.issue_request_*` connection. Its unique M1250 run reached
`PHASE_M1219R9_DIRECTED_COMPLETE`; `directed_nonfirst` therefore completed, and
`cp_nonfirst` later accumulated three matches before an unrelated random-phase
M935 fault.

R12 changes the synthetic directed path to force
`dut.u_frozen_m935.issue_request_*`, an output-variable seam. M1265 stops on the
first directed transaction that requires a forced `first` value opposite to
the preceding first-beat cases. The failure occurs before R12 can establish a
single `cp_nonfirst` match. This is consistent with child-output force not being
a reliable executable proxy for the parent M1162 input connection under this
compiled topology. It is not consistent with a new M1162 RTL regression,
because M1162/M935/M528/SVA are byte-frozen and the intended Boolean tuple has
no psum-valid path.

The oracle statement “a non-first source beat must not request/read prior psum”
is also architecturally correct: M935 selects `issue_psum_prior` only when
`active_ctx_first_q` is true; later beats accumulate from `psum_acc_q` and only
the final beat commits.

## Claim impact

- M1265 remains **FAILED_OR_INCOMPLETE** and provides no R12 functional PASS.
- The failure does not revoke the source-level C1 schedule opportunity or the
  existing M1162 Boolean contract, but it also cannot elevate either to RTL
  cycle, traffic, energy, PPA, system-speedup, or headline evidence.
- The R11 directed boundary evidence remains valid only within its prior
  boundary; its later random failure remains quarantined.
- No integrated-random M935 claim is created. R12 explicitly declared
  `integrated_random=false`, and it never reached its integrated normal phase.

### Counterfactual upper impact if RTL really read psum on every beat

The sealed schedule has 812,160 tasks and 70,853,184 beats per axis. Exactly
812,160 beats are first beats, leaving 70,041,024 non-first beats. At 96x19 bit
= 228 B per psum payload, a real extra read on every non-first beat would add:

- `70,041,024 x 228 = 15,969,353,472 B` per axis (15.969 GB decimal,
  14.873 GiB);
- total psum-read count would become 87.2404x the first-only count;
- across the three comparison axes, the diagnostic replay volume would add
  47.908 GB, although those axes are alternatives and must not be summed as one
  deployed workload.

Under the existing ideal model (parallel weight/psum services, zero request
stall, one-cycle response), this would not necessarily change the II=2
recurrence because the extra response could arrive beside the weight response.
It would, however, invalidate any first-only psum traffic/energy claim and can
create unmodelled bandwidth stalls in a finite service. This counterfactual is
reported only to bound risk; M1265 does not prove it occurred.

## Disposition

Classify M1265 as a **TB child-output-seam/stimulus failure with an
under-diagnostic compound oracle message**. Do not patch the oracle invariant,
do not cite the failure as RTL extra traffic, and do not retry M1265. Any future
successor would need a fresh namespace and a stimulus method that demonstrates
the exact parent-boundary tuple values at the accepting edge; this review does
not authorize such a successor or launch.

P2 observation: the fatal should report `weight_fire_delta`, `psum_fire_delta`,
`psum_req_valid`, parent `issue_request_first`, and latched `request_first_q`
separately. The missing diagnostic is the only residual uncertainty in this
forensic classification.
