# Fresh-hammer request: M814 / M533 R18 source package

Independently audit the exact sealed source package named in `request.json`. This is source-only work: do not invoke VCS, simv, a license query, or any EDA; do not create a result or launch release.

The highest-risk checks are semantic, not cosmetic:

1. Derive the matcher parents for masks `0001/0003/000c/0031/004c/0083` and show `[null,0,null,0,2,1]` without trusting TB comments.
2. Check against frozen RTL that the witness can make the macro response and direct forward enqueue in the same cycle. Reject the withdrawn four-row construction.
3. Verify real ping-pong coverage uses `dut.prep_active_q && dut.exec_active_q`; the R17 prep-handshake proxy must be absent.
4. Verify normal coverage is gated before P2, and final PASS remains downstream of P2, held-final, and all six attacks.
5. Recount actual literal `require_regular_sha` calls as 83. Do not include the function definition.
6. Rerun pinned-Python 3.6 static/closure tests, all three negative mutations, and the runner-owned rc86 pre-mkdir dry-run with zero side effects.

On a clean pass, author the runner-required review identity at `reviews/m814_m533_r18_unit_delay_source_static_hammer_r1_20260829/` and double seal it. The review itself must keep VCS launch unauthorized.
