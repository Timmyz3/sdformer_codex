# M212 + M213 independent hammer review

Score: **94/100**.  P0: **0**.

M212's terminal-descriptor hint is necessary and sufficient under the accepted
ready/valid contract.  It closes only the final partial compact window; the
authoritative registered done fence is still accepted normally.  Independent
current-source Synopsys VCS passes hand-derived one-descriptor (3 cycles),
three-descriptor (4), full-tail (3), zero-tail (3), 736-cycle descriptor hold,
720-cycle group stall, bank-96, and same-edge header-chain cases with no output
identity mismatch or assertion failure.  Fresh-last and queued plus all-zero
raw-last hint cases also pass under descriptor backpressure.

A fresh current-source M210/M212 256-case VCS A/B reproduces 36 one-cycle
improvements, 220 unchanged cases, and zero regressions.  The software
recurrence matches all 256 M212 cases exactly.  A fresh four-worker replay of
all 120 frozen records and 5,580,000 H67 FC2 tokens is byte-identical to M213:
90,388,767 cycles, saving 795,772 versus M210 (1.008803882x incremental).
Stage 0 reaches the old analytic lower bound; stages 1--3 retain 281,490 cycles.

The independently audited 3 ns DC smoke is current-RTL logic-only pre-macro:
20,620.782090 um2, 30,535 leaf cells, 2,773 sequential cells, 79 logic levels,
2.52 ns critical path, +0.0007 ns setup and +0.0000 ns hold.  This is +0.6612%
area versus M210 and has no physical timing margin under ideal clock and
ZeroWireload.  It is not paper PPA.

The review records four M214 risks verbatim in the JSON receipt: authoritative
done acceptance, lone-window/bitmap atomicity, non-causal overcount in the
current `same_cycle_done_loads` counter, and the 0.7 ps timing-margin hazard.
M214 remains an opportunity concept until exact VCS and matched DC pass.

The admitted scope is the isolated sparse FC2 frontend.  Complete FC2, FFN,
physical, system, and headline claims remain false.  `docs/359` was not
modified and remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
