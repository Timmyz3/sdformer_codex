# M1802 independent source hammer: M1801 C2 registered public fault evidence successor

Verdict: **PASS, 99/100, P0=0, P1=0, P2=0.** No EDA was run.

M1801 is additive. Against M1796, the RTL changes only milestone/module/instance identifiers. The legal data, ready/valid, issue, Acc24 arithmetic, completion, child-core, and eight-bank adapter paths are identical. The full testbench retains the same five exact cycle pairs and workload; it fixes the M1797 PASS-token inconsistency from `protocol_attacks=4` to `protocol_attacks=5`.

The exact owner equations are now hard-bound:

- core: `header_valid || raw_valid || core_busy || core_mem_req_valid || core_mem_rsp_valid || result_valid || token_done_valid`
- adapter: `adapter_busy || core_mem_req_valid || (|mem_rsp_valid)`

The independent run confirmed rejection of both complete-enable-to-zero mutations, every 7+3 individual owner-term deletion, both request/response valid-gate mutations, reset-recovery corruption, the three real K8 illegal accept/sticky gate removals, the five-attack hard-gate corruption, the PASS-token corruption, and all other declared mutations. The total is 42/42 rejected under CPython 3.6 and 42/42 under CPython 3.12.

The public `protocol_error` has only the synchronous reset-to-zero and sticky-set-to-one assignments. There is no combinational public assignment. Idle payload X values under valid/owner-enable zero cannot enter the public fault cone. The full top retains accept-zero plus next-registered-sticky checks for illegal header, illegal raw, and spurious response, followed by reset and legal numeric/tuple/weight/completion traffic.

All M1796/M1797 and M1801 contract/author identities are double sealed; docs/359 remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`. No M1801 attempt or result exists, and this review ran no EDA.

Authorization is deliberately narrow: one source-identity-bound release containing exactly two VCS campaigns may be created—(1) the directed registered-public-fault boundary and (2) the full K8-vs-K1x8 numeric/cycle/attack regression. Both need unique PASS tokens and zero assertion/protocol/numeric/tuple/weight/completion failures. This review does not establish RTL functionality, mapped functionality, PPA, power, energy, performance, system speedup, or a paper-citable result.
