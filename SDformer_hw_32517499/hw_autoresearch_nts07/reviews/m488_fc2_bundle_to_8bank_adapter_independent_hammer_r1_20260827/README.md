# M488 FC2 bundle-to-8bank adapter independent hammer r1

## Verdict

**77/100. GO as a directed protocol primitive; NO-GO as the canonical K8 vs K1x8 fairness closure.**

M488 correctly implements the intended one-entry partial request distributor,
eight slot-indexed response assemblies, out-of-order cross-bank reconstruction,
held response selection under core stall, and sticky fail-closed quarantine for
invalid bank responses. The exact-SHA production run is internally consistent,
and a fresh independent Synopsys VCS compile/run reproduced all directed counts.

However, the same-cycle slot-reuse path has a real combinational dependency
cycle, and neither M342 SOURCE_CAP=8 nor M349 K1x8 is integrated in this
milestone. Therefore M488 narrows the interface-shape gap but does **not** yet
close the external-bank fairness gap used by the paper comparison.

## Score

| Dimension | Score | Finding |
|---|---:|---|
| Exact-SHA contract and seal | 18/20 | Frozen RTL/SVA/TB/filelist/contract/docs359 hashes pass; production `SHA256SUMS` verifies. Runner is recorded but not itself preflight-pinned, and `RUN_COMPLETE.txt` is emitted after the inner checksum list. |
| Request distribution | 13/18 | Accepted banks are removed from the pending mask; partial bank readiness is stable and no duplicate beat appears in the 341-beat scoreboard. Same-cycle reuse creates the P0 combinational cycle. |
| Response reassembly | 18/20 | Epoch/slot/generation/tag and expected-bank checks reject duplicate, wrong-bank and stale responses; 8 slots tolerate cross-bank/cross-slot reordering. Zero-cycle SRAM response is not supported/documented. |
| Stall and slot reuse | 9/16 | The held-slot repair preserves a visible response while stalled and directed reuse passes in VCS, but the reuse equations have two legal Boolean fixed points. |
| Fail-closed behavior | 11/12 | An illegal response atomically suppresses all request/response accepts and makes the fault sticky. A request aimed at a currently live slot is treated as a fatal protocol error, not ordinary ready/valid backpressure; integration must prove the producer contract. |
| Fairness closure | 4/10 | The external pins now have the same scalar-bank shape as M349, but there is no M342+M488 integration, no M349 matched wrapper, no frozen 120-record replay, and no common SRAM latency/energy model. |
| Claim discipline | 4/4 | Contract and receipt correctly say integration/DC/energy/system speedup/headline are false. |
| **Total** | **77/100** | **Primitive GO; publication comparison NO-GO.** |

## Independent replay

Tool: Synopsys VCS V-2023.12-SP1, fresh compile from the frozen filelist.

Observed pass line:

```text
PASS M488 bundle-to-8bank adapter requests=98 bank_beats=341 partial=46 request_stalls=11 response_stalls=21 out_of_order=10 attack=1 cycles=228 headline=false system_speedup=false
```

Coverage report independently reproduced:

- full-eight request: 15 matches;
- partial request distribution: 46;
- pending request stall: 12;
- eight bank responses in one cycle: 1;
- out-of-order bundle response: 8 property matches / 10 scoreboard events;
- core response stall: 21;
- same-cycle slot reuse: 1;
- protocol-error cover: 2 cycles;
- assertion failures: 0.

The independent run uses the same frozen RTL/SVA/TB, but a separate compilation
tree and seed (`488991`). This establishes reproducibility of the directed
simulation, not independence of the stimulus design.

## Static audit

### Correct and sufficiently exercised

1. **Partial request distribution.** When there is no pending request, accepted
   banks may fire fall-through; only unaccepted bits are captured. While a mask
   is pending, accepted bits are cleared (`pending_mask_q & ~bank_req_accept`).
   Per-bank request payload is held stable under stall by SVA. Exact expected
   bank-beat accounting would fail if an accepted bank were duplicated.
2. **Out-of-order response reconstruction.** Each bank response is accepted only
   for a live slot, expected bank, not-yet-arrived bit, and exact epoch,
   generation and tag. Multiple banks may complete one slot in one cycle;
   different slots may complete out of order.
3. **Visible response stability.** On the first stalled core response, the
   selected slot is latched. This prevents a newly completed lower-numbered slot
   from replacing the visible response.
4. **Atomic fail-closed behavior.** `illegal_request` or `illegal_response`
   immediately raises `protocol_error`; all accepts are suppressed in that
   cycle, then `fault_q` keeps the quarantine sticky.

### P0 findings

1. **Break the same-cycle slot-reuse combinational cycle.** Frozen lines 147-148,
   160, 184 and 191-192 form:
   `core_rsp_accept -> req_slot_open -> req_shape_legal -> illegal_request -> protocol_error -> core_rsp_accept`.
   For the retiring slot, both accept/no-fault and reject/fault are valid Boolean
   fixed points. A VCS cover reaching the accept point is not a synthesis proof.
   Split intrinsic request shape legality from slot availability and calculate a
   uniquely acyclic retire-and-reuse enable, or remove the zero-bubble reuse.
2. **Integrate the real endpoints before claiming fairness closure.** Instantiate
   M342 SOURCE_CAP=8 behind repaired M488 and compare against M349 through the
   same eight bank models, latency, request-allow schedule, result backpressure,
   counters and frozen 120-record trace. The current contract explicitly marks
   both integrations false.
3. **Rerun exact VCS after P0.1 with an explicit adversarial hold case.** Stall a
   completed higher slot, complete a lower slot while the stall persists, then
   release; also retire/reallocate the held slot on the same edge. Preserve
   identity/weight/beat conservation checks.

### P1 findings

1. Add an explicit minimum-response-latency contract (`>=1 cycle`) or buffer a
   zero-cycle bank response; a fall-through bank response in the acceptance
   cycle sees the slot as not live and is quarantined.
2. Prove whether presenting a valid request for a live slot is forbidden by the
   M218 producer. If it is normal backpressure, it must drive `ready=0` without
   latching `protocol_error`; if it is forbidden, add an end-to-end producer SVA.
3. Once the integrated frozen replay passes, include M488 in matched K8 synthesis
   and switching activity. Do not compare bare M342 K8 area against M349 K1x8.
4. Use the same scalar SRAM macro/CACTI ports and account for adapter assembly
   storage. Logic-only DC cannot close memory area or energy fairness.

### P2 findings

1. Drive inactive `core_rsp_weight` banks to zero or state explicitly that their
   data are don't-care; slot reuse currently may expose stale inactive-bank data
   behind a zero mask bit.
2. Pin the runner hash in an outer contract and include a separately generated
   outer seal over `SHA256SUMS` plus `RUN_COMPLETE.txt`.
3. Parse reported counts into the receipt instead of hard-coding them after a
   permissive positive-count regex, even though frozen deterministic stimulus
   makes the current values reproduce.

## Fairness question

The adapter closes only one structural sub-gap: K8 can now physically talk to
eight independent scalar bank endpoints carrying epoch/slot/generation/tag,
output block, slice, channel and 128-bit response data, matching the *shape* of
M349's memory pins. It does not yet demonstrate equal external memory behavior:

- K8's bundle is admitted once and drains remaining banks through one pending
  mask; M349's eight services issue independently.
- K8 reconstructs an atomic response before M218 consumes it; M349 consumes
  scalar responses in independent services and joins final accumulators.
- no common integrated SRAM latency/stall schedule or frozen workload has been
  run through both paths.

Those internal scheduling differences may be legitimate architectural effects,
but they must be measured behind identical external banks. Until the integrated
wrapper exists, the M342 K8 vs M349 K1x8 matched area/energy claim is not closed.

## Allowed claims

- M488 is a directed-VCS-validated bundle-to-eight-bank protocol adapter.
- On the frozen directed test it distributed 98 bundles into 341 exactly
  accounted bank beats, tolerated partial readiness and cross-slot response
  reordering, reconstructed atomic responses, and quarantined one stale attack.
- M488 provides the missing *interface primitive* needed to build a matched K8
  versus K1x8 experiment.

## Forbidden claims

- M488 closes K8 versus K1x8 bank-interface fairness.
- M342 K8 and M349 K1x8 have been integrated or replayed behind the same SRAMs.
- same-cycle slot reuse is synthesis-safe or DRC-clean.
- any area, frequency, power, energy, FC2 speedup or system speedup follows from
  M488 directed VCS.
- the prior 5.281x serialized-port sensitivity is restored as a paper speedup.

## Seals and identity

- Production result inner seal SHA256:
  `27fa7baf6c25337b029fdf7470dfaaca914e97dbf3e43c0b135d169eae2905be`
- Independent replay hash-list SHA256:
  `a83418f3ee7c261737b18e07d865949ba5e35f722bfe59cbf39d2b187e570d11`
- Independent `replay2_sim.log` SHA256:
  `9439734cd88d2f8a6dfff5d2358358338ee672ca5bbe8d11bdcbd2b6d35b5f51`
- Independent `replay2_assert.report` SHA256:
  `64d4e3df355d090f2012070753950750fe8a8ebc262a274b7b1a897027b6a4f4`
- Frozen docs/359 SHA256 remains:
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

No production RTL, contract, runner, or docs/359 file was modified by this
review. The first `replay_compile.log` is explicitly invalid due to wrong CWD;
only `replay2_*` is citable.

