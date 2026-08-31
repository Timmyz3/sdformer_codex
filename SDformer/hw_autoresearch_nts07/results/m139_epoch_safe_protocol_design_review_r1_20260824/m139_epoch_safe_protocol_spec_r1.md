# M139 reset-epoch-safe response bridge: pre-implementation specification r1

Date: 2026-08-24  
Status: `PROPOSED_NOT_IMPLEMENTED`  
Scope: repair the M137 reset-epoch stale-response alias without changing its admitted normal-operation II or delivery latency.  
Claim boundary: this document is a design specification and review artifact. It is not RTL, VCS, synthesis, formal, PPA, power, physical-speedup, system-speedup, headline, or paper-ready evidence.

## 1. Problem to close

M137 clears `next_token_q`, pending state, skid state, and the sticky fault on `rst_core`, but the macro interface has no reset/flush acknowledgement or response epoch. The concrete silent-corruption trace is:

1. accept a pre-reset request with token `16'h0000`;
2. reset the bridge before that response returns;
3. accept the first post-reset request, again with token `16'h0000`;
4. receive the delayed pre-reset response with the old payload and token `16'h0000`;
5. observe a token match and accept stale data under the new request metadata without `protocol_error`.

Natural `16'hffff -> 16'h0000` wrap is not the defect: M137 has at most one pending request under the frozen strict one-cycle macro contract, and an independent 65,538-request test crossed natural wrap. Reset creates an immediate identity reuse while an external response can survive.

## 2. Decision

Select a four-phase macro flush request/acknowledgement protocol. The bridge must remain quarantined until it observes a fresh `ack=0 -> ack=1 -> ack=0` sequence associated with its asserted flush request. This makes a stale or stuck acknowledgement fail closed.

### Option comparison

| Option | Required bridge/interface cost | Recovery | Normal II/latency | Safety result | Decision |
|---|---:|---|---|---|---|
| Four-phase macro flush request/ack | 2-bit FSM, one `flush_req` output, one `flush_ack` input; optional status output | Variable; minimum sampled `0 -> 1 -> 0` acknowledgement sequence | Architectural II remains 1; fixed-one-cycle delivery remains 1; no added normal pipeline stage | Observable fail-closed boundary: no service before the macro wrapper certifies that all pre-reset returns are destroyed | **Selected** |
| External epoch seed echoed by macro | At least E epoch state bits, seed-valid handshake, E request identity wires, E response identity wires, E-bit compare, and a persistent non-reset seed owner | Fixed after a fresh seed is accepted | No mandatory cycle change, but wider compare can affect timing | Safe only while seed uniqueness/non-reuse is guaranteed; finite E eventually wraps and the root contract moves outside the bridge | Rejected as higher cost for this one-outstanding bridge |
| Atomic bridge+macro reset contract | No new bridge state or ports | Immediate after reset deassertion | Unchanged | Not independently fail closed: the bridge cannot observe a macro that failed to erase an old return, so the exact same-token replay remains silent if the assumption is violated | Rejected as an assumption-only closure |

An atomic top-level reset proof can still be a useful integration check, but it is not a substitute for the selected observable handshake. An external epoch remains a future option if the implementation later permits multiple reset domains, uncontrolled response sources, or more than one outstanding request.

## 3. Required interface delta

Retain every M137 functional request, macro request/response, consumer response, status, and reset port. Add exactly two required macro-boundary ports:

```systemverilog
output logic macro_flush_req;
input  logic macro_flush_ack;
```

An optional zero-state-cost diagnostic output may expose the state decode:

```systemverilog
output logic recovery_active; // optional integration/status port
```

`recovery_active` is not required for correctness or the macro handshake. If omitted, the same state is observable through `request_ready==0`, `macro_request_valid==0`, and the bound assertions.

The acknowledgement belongs to the 16-bank wrapper, not to the raw foundry SRAM. Its completion guarantee is:

> When the wrapper asserts `macro_flush_ack=1` in response to the active request, every pre-flush request and assembled or partial response has been discarded. After the bridge drops `macro_flush_req`, the wrapper must return `macro_flush_ack=0` and must not emit any response belonging to the flushed epoch.

This contract includes per-bank partial returns, aggregate return-valid state, token-echo state, and any response pipeline between SRAMs and M139.

## 4. Minimal synthesizable control state

Add one 2-bit recovery FSM; do not add epoch bits, payload storage, token width, or a new normal-operation pipeline stage.

```systemverilog
typedef enum logic [1:0] {
  REC_WAIT_ACK_LOW,   // flush_req=1; reject a stale/high ack
  REC_WAIT_ACK_HIGH,  // flush_req=1; wait for flush completion
  REC_WAIT_ACK_DROP,  // flush_req=0; wait for wrapper to return to normal
  REC_RUN             // execute the unchanged M137 normal protocol
} recovery_state_e;
```

Existing M137 state remains: sticky `fault_q`, `next_token_q[15:0]`, one pending identity/metadata record, and one skid payload/metadata record.

### Reset and transition rules

- On every sampled `rst_core=1`, clear M137 pending/skid/fault/token state and set the recovery FSM to `REC_WAIT_ACK_LOW`.
- Drive `macro_flush_req = rst_core || state inside {REC_WAIT_ACK_LOW, REC_WAIT_ACK_HIGH}`.
- In `REC_WAIT_ACK_LOW`, observe at least one clock edge with `macro_flush_ack=0`, then enter `REC_WAIT_ACK_HIGH`. A high acknowledgement inherited from before reset cannot release the bridge.
- In `REC_WAIT_ACK_HIGH`, an edge with `macro_flush_ack=1 && !macro_response_valid` certifies completion and enters `REC_WAIT_ACK_DROP`.
- In `REC_WAIT_ACK_DROP`, keep service blocked and wait for an edge with `macro_flush_ack=0 && !macro_response_valid`, then enter `REC_RUN`.
- If `macro_response_valid && macro_flush_ack` is observed on the claimed completion edge, enter sticky quarantine rather than completing recovery.
- If any `macro_response_valid` is observed in `REC_WAIT_ACK_DROP`, enter sticky quarantine: the wrapper claimed completion but still emitted a return.
- Responses in `REC_WAIT_ACK_LOW` and `REC_WAIT_ACK_HIGH` are pre-completion drain traffic. Drop them without generating consumer output and without changing M137 pending/skid/token state.
- Reasserting reset from any state restarts at `REC_WAIT_ACK_LOW`; a high acknowledgement from the interrupted handshake cannot release the restarted epoch.
- In `REC_RUN`, use the exact M137 normal-operation request, token, one-cycle response, fallthrough, skid, and sticky fault rules.
- A sticky protocol fault blocks all service until a new reset and a new complete flush handshake. It must not be cleared merely by acknowledgement activity.

### Quarantine outputs

While `rst_core || state != REC_RUN || fault_q`:

- `request_ready=0` and `request_accept=0`;
- `macro_request_valid=0` and no request identity/address may advance;
- `response_valid=0` and `response_accept=0`;
- pending/skid/token state may not advance, except for reset clearing;
- `busy=1` is recommended for `state != REC_RUN`; and
- stale response payload and metadata may not reach consumer outputs.

## 5. Cycle behavior and cost boundary

Normal operation is architecturally unchanged:

- sustained initiation interval: **II=1** under the same fixed-one-cycle behavioral macro contract;
- accepted-request to fallthrough delivery: **1 cycle** when the consumer is ready;
- skid depth: **1**;
- token width and service width: unchanged;
- new normal pipeline stages: **0**.

The recovery FSM adds only a run-state decode to normal ready/valid gating. It does not justify a frequency-neutral claim until DC/PT measures the actual cone.

Minimum recovery example, with reset low before edge E0 and an immediately responsive wrapper:

| Sampled edge | `flush_req` before edge | `flush_ack` at edge | State after edge | Request may accept at this edge? |
|---|---:|---:|---|---:|
| E0 | 1 | 0 | `REC_WAIT_ACK_HIGH` | no |
| E1 | 1 | 1 | `REC_WAIT_ACK_DROP` | no |
| E2 | 0 | 0 | `REC_RUN` | no; ready becomes visible after E2 |
| E3 | 0 | 0 | `REC_RUN` | yes |

Therefore the minimum reset-release-to-ready recovery is three sampled transition edges, and the first valid request can be accepted at E3. For a delayed wrapper, add its low/flush/high/drop latency. There is deliberately no safety timeout in the minimum design: a missing or stuck acknowledgement blocks forever instead of guessing that stale responses are gone. A watchdog may be added later for diagnosability, but it must only raise a sticky fault and must never auto-release recovery.

## 6. Required SVA contract

The implementation gate must bind assertions equivalent to all of the following groups.

### Recovery safety

1. `recovery_active |-> !request_ready && !request_accept && !macro_request_valid && !response_valid && !response_accept`.
2. `macro_flush_req` is asserted only in reset/`WAIT_LOW`/`WAIT_HIGH`, and is deasserted in `WAIT_DROP`/`RUN`.
3. `WAIT_LOW -> WAIT_HIGH` requires a sampled low acknowledgement.
4. `WAIT_HIGH -> WAIT_DROP` requires `macro_flush_ack && !macro_response_valid`.
5. `WAIT_DROP -> RUN` requires `!macro_flush_ack && !macro_response_valid`.
6. No path enters `RUN` without the ordered low/high/low acknowledgement history since the most recent reset.
7. Responses in `WAIT_LOW`/`WAIT_HIGH` cannot change token, pending, skid, or consumer-visible response state.
8. A response on the completion edge, or any response in `WAIT_DROP`, raises sticky quarantine and prevents entry to `RUN`.
9. Reset from every FSM/pending/skid phase returns to `WAIT_LOW` and clears all consumer-visible state.
10. The first accepted request of every completed recovery epoch uses token `16'h0000`.

### Normal-operation conservation

11. Retain M137 exact accept equalities and legal-request gating.
12. Every accepted request causes exactly one macro request with the same token and address set.
13. Under the strict fixed-one-cycle macro contract, exactly one correctly tagged response arrives on the following cycle or sticky quarantine is raised.
14. Wrong, missing, unsolicited, or duplicate responses fail closed.
15. Token increments exactly once per request, including `16'hffff -> 16'h0000`.
16. Same-cycle skid pop plus new request conserves both transactions.
17. When stalled, words, start, last, width, tag, and token are all stable.
18. Once raised, `protocol_error` remains high until reset; reset alone does not permit traffic before a new flush handshake.

### Covers

Cover minimum recovery, delayed recovery, initially-high acknowledgement, stale-response drain, reset with pending, reset with skid, consecutive reset, first post-recovery request, normal II1, same-cycle pop+new, stall/release, and natural token wrap.

Acknowledgement progress is an environment liveness assumption, not a DUT safety assertion. If the wrapper never completes the handshake, the correct bounded VCS result is continued quarantine.

## 7. Trust boundary and prohibited overclaims

The selected protocol is fail closed against stale/high/stuck handshakes, drain responses before completion, reset at every local pipeline phase, and old returns appearing before service release. It cannot distinguish a Byzantine wrapper that asserts a valid flush completion and later deliberately replays an old payload with the exact token of a new pending request. The wrapper's flush guarantee must therefore be asserted inside the wrapper and checked at integration.

Do not claim:

- the M137 P1 is closed before M139 RTL and the directed VCS matrix pass;
- frequency, area, power, energy, physical speedup, or system speedup from this specification;
- fixed reset recovery unless a bounded wrapper acknowledgement latency is implemented and measured;
- protection from a wrapper that falsely acknowledges flush completion; or
- that the optional epoch scheme is implemented.

## 8. Implementation admission sequence

1. Implement only the two-bit FSM and two required ports around the exact M137 normal path.
2. Implement a behavioral four-phase flush wrapper and run every row in the frozen directed attack matrix.
3. Bind the recovery and inherited normal-operation assertions; require all negative attacks to terminate in quarantine rather than consumer delivery.
4. Re-run the 128-request M137 functional set and the 65,538-request natural-wrap campaign after recovery.
5. Run exact-SHA VCS, then Formality, then matched logic-only DC. Only the last two can establish equivalence and timing/area impact.
6. Integrate the handshake into the 16-bank wrapper and prove that acknowledgement covers every partial-bank and aggregate-response state.

