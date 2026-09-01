# M1801 C2 registered public-fault evidence successor — author receipt

Status: `PASS_M1801_SOURCE_AUTHORING__M1797_P1_P2_CLOSED__READY_FOR_DIFFERENT_AUTHOR_M1802_HAMMER__NO_EDA`.

M1801 is an additive evidence successor to M1796. It does not modify M1796 or
the M1797 review and does not change the registered sticky fault repair, data,
ready/valid, issue schedule, Acc24 arithmetic, completion, or memory interface.

## M1797-P1-01 closure

The checker now binds the exact normalized owner equations:

- core: `header_valid || raw_valid || core_busy || core_mem_req_valid || core_mem_rsp_valid || result_valid || token_done_valid`;
- adapter: `adapter_busy || core_mem_req_valid || (|mem_rsp_valid)`.

The mutation campaign rejects both complete equations replaced by constant zero,
deletion of each of the ten owner terms, deletion of the request/response valid
gates, and mutations of the reset-recovery sequence. These checks close the
specific masking hole demonstrated by M1797.

## M1797-P2-01 closure

The full TB executes two header attacks, one K8 raw attack, and two response
attacks. Its functional hard gate requires `protocol_attack_count==5`, and the
sole PASS token now reports `protocol_attacks=5`. Both values are separately
mutation-protected. The K8 illegal-header, illegal-raw, and spurious-response
paths each retain accept-zero and later sticky-fault gates.

## Mechanical result

The fail-closed checker and 42-mutation suite pass under CPython 3.6 and 3.12.
The M1797 fail-closed review is checked as predecessor authority. `docs/359`
remains `dedde7ce...bdfc4`. No VCS, simulator, synthesis, license, attempt,
result, commit, or push was performed.

A different author must run M1802. Only a P0=0/P1=0 review may authorize the
two RTL VCS campaigns; physical K8/K1x8 comparison and energy must then be
rebuilt under the same constraints.
