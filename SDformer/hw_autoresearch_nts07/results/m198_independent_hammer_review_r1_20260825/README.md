# M198 independent hammer review

Score: **88/100**. Verdict: conditional pass for the sealed **abstract raw
bitmap scanner replay DSE only**. This is not an admitted physical, complete
FC2, FFN, or system speedup.

The independent checker imports no production analyzer. It verifies every
payload SHA/extent/popcount, explicitly decodes all 120 frozen bitmaps, and
exactly reproduces the aggregate, every stage, and all 12 R/B wall-cycle
points. It also passes 576 named boundary/scalar-vector checks and 12,000
random recurrence attacks covering early full-window close, token-end partial
close, trailing zeros, zero tokens, odd tails, buffer reuse, and the ban on
cross-window/token same-cycle fill.

The clean B2 decomposition matters. R4/B2's 1.270698x versus raw W1/R1/B2 is
1.209831x scanner-width benefit followed by only 1.050311x pair-fusion
increment against same-width W1. R8/B2 rises to 1.303184x, but requires a
768-bit raw ingress for only 2.56% fewer abstract wall cycles than R4/B2.
Extra B3/B4 buffers save at most 0.1576% at R4, so R4/B2 remains the sensible
implementation target.

P0 is a bounded stable R4-to-F2 scanner/compactor and the matched finite
frontend. VCS/SVA must prove residual lanes at mid-cycle early closes, order,
no drop/duplicate, partial and all-zero token flush, stalls, queue fullness,
buffer reuse, token/window/epoch tags, SRAM response quarantine and exact
Acc24/commit. DC must then measure the same throughput point. Until that
exists, M198's `+1/+2` close convention and token serialization are model
rules rather than observed controller timing.

One small P1 optimization is stage-select fusion bypass: stage 0 pair fusion
is 0.994911x versus its same-width W1 point. Bypassing it changes R4/B2 from
90,222,444 to 90,112,890 abstract cycles, just 0.1216%; useful hygiene, not a
new headline.
