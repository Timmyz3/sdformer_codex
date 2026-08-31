# M285 / M273r2 author repair

This is the M273 author's response to P1-1 and P1-2 from the M276 independent
hammer.  It is author evidence and waits for review by a different agent.

The production RTL now reports only a registered sticky fault.  Current
config/raw frame errors are captured at their handshake edge and no longer
combinationally gate `protocol_error`, `stage1_issue`, `stage2_issue`,
`product_push`, or `result_valid`.  A configured context with zero loaded tiles
cannot release; the release attempt is rejected and enters sticky quarantine.
Consequently, the clean `5*N+19` formula is explicitly limited to `N>=1`.

Exact-SHA Synopsys VCS retained N1/N4 clean cycles 24/39 and the fixed
one-in-eight-ready N40 result 1618.  The run covered 1681 half-cycle windows,
18 legal configuration-beat accepts, 225 legal raw-beat accepts, zero legal
protocol-error pulses, zero intra-half issue/result-valid changes, all seven old
attacks, the new zero-tile attack, and 182 full-FIFO simultaneous pop/push
cycles.  Numeric/order and assertion failures were zero.

No new DC run was started.  This milestone does not claim an area-matched Fixed
comparison, a new speedup, accuracy, PPA, energy, system speedup, or headline.
`docs/359_DATE终局冻结_20260813.md` remains unchanged.
