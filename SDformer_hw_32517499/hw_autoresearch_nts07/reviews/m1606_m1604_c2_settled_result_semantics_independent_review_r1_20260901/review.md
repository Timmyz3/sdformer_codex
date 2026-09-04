# M1606 — M1604 C2 settled result and handshake-semantics review

Verdict: **M1604 is a consumed failed diagnostic, not an RTL/mapped PASS and
not citable. The cycle-4 stop is a post-accept combinational false protocol
error, not a protocol violation at the accepting edge and not an RTL/netlist
difference.** M1606 ran no VCS, `simv`, DC, SAIF, or PTPX and authorizes none.

## Execution and exact observation

The frozen M1604 identity consumed its only attempt and contains exactly one
VCS compile and one `k8_case0` simulation. Compilation has zero errors. The
simulation is clean for cycles 1--3, then stops at the 22.501 ns settled sample
in cycle 4 with `top_pns=100/100`. This field means protocol error is one while
numeric overflow and stale response are zero in both RTL and mapped DUTs.
`first_difference_cycle=-1`; both eight-bit endpoint-fault vectors and all six
registered fault/stale taps are zero. Thus M1594's 1 ps settle repair did its
job: the prior active-region mapped X is gone and both implementations agree.

## Ready/valid proof

The relevant timeline is:

| Time | Producer / state | Accept and legality | Public result |
|---:|---|---|---|
| 21.000 ns negedge | producer asserts `raw_valid=1`, four legal lanes and `raw_last=1` | token is active and `raw_done_q=0` | clean |
| 22.500 ns just before posedge | `raw_valid=1`, `raw_packet_legal=1` | `raw_accept=1`, `illegal_request=0` | `fault_q=0`, `protocol_error=0` |
| 22.501 ns settled | producer legally still holds `raw_valid=1`; accepted terminal beat has advanced `raw_done_q=1` | the same held beat is now reinterpreted as `!raw_packet_legal`, so `raw_accept=0`, `illegal_request=1` | `fault_q=0`, combinational `protocol_error=1` |
| 24.000 ns scheduled negedge | producer withdraws `raw_valid` | `illegal_request` returns to zero | no registered fault was ever set |

This is normal synchronous ready/valid behavior: the producer is permitted to
withdraw `valid` only after observing the accepting edge. The compactor instead
computes `protocol_error = fault_q || illegal_request` combinationally while its
post-edge state already declares the terminal packet done. The false pulse then
propagates through the frontend and core to the K8 top. All registered fault
taps remain zero because `illegal_request` was zero at the accepting edge.

## Three possible treatments

Pre-edge sampling would observe the legal transaction (`raw_accept=1`, no
error), but it is not a repair: it hides a real public-output pulse and a naive
posedge checker would recreate M1593's active-region gate-level race. Keep the
post-edge 1 ps checker; it is race-free and faithfully reports the implemented
interface.

The minimum semantic repair is to expose only the sticky registered fault:
change the compactor's public assignment from
`protocol_error = fault_q || illegal_request` to
`protocol_error = fault_q`. Preserve the `illegal_request` expression, its
`fault_q` latch, `raw_ready/raw_packet_legal` acceptance gate, M1601 settled
checker, stimulus, and all other behavior. A genuinely illegal request present
at a sampling edge still sets `fault_q` and becomes visible after that edge;
malformed input remains unable to handshake through `raw_ready`.

## Unique next step and boundary

Author one source-only RTL successor containing only that public-output change,
then submit it to an independent static source review. Do not change the
checker to pre-edge sampling. Because the RTL identity changes, any eventual
RTL/mapped comparison requires resynthesis and a new mapped netlist, but M1606
authorizes no VCS, simulation, DC, PTPX, SAIF, or retry. M1604 remains
`FAILED_DIAGNOSTIC__SEMANTIC_FALSE_POSITIVE__NOT_CITABLE`; it creates no timing,
power, PPA, speedup, system, or paper claim. docs/359 remains `dedde7ce...`.
