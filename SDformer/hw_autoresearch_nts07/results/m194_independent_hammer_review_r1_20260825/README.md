# M194 independent hammer review

Verdict: **82/100, conditional pass as a selector only. Proceed to an M195
integrated frontend, but do not admit the 1.092758696 additive proxy as
integrated throughput/area.**

The sealed evidence is internally reproducible. All pinned VCS inputs/outputs
and both runner hashes pass SHA256 verification. The original run reports
5,004 accepted legal pairs, 40,032 bank checks, 12/12 nonzero covers and no
assertion-failure marker. The DC reports independently confirm 550.998004 um2,
638 cells, 131 sequential cells, 15 logic levels, +1.7766 ns setup and +0.0208
ns hold slack at the declared 28 nm, ideal-clock, zero-wireload, macro-free
3 ns screen.

I also compiled the released RTL with a new VCS testbench that reuses neither
the milestone TB nor its SVA. It accepted 20,005 legal transactions; 20,004
were issued and checked over 160,032 bank outputs, while one additional legal
transaction was deliberately stalled and flushed by reset. Coverage included
6,747 two-window pairs, 48,463 bank fallthrough selections, 109 last and 19,896
not-last cases, 1,962 stall holds, 646 same-cycle replacements, 65 generic
255+255 count-boundary pairs, one reset flush and all four fail-close attack
classes. The independent test passed.

The block is not sufficient to implement M195. It selects from counts and heads
that some other block must maintain. It has no descriptor buffers, accepted
head decrement, all-bank empty/release logic, odd-tail/token-boundary control,
SRAM response tags, stale-response epoch, Acc24 return alignment or final
commit. Those are the dominant correctness and area risks. In particular,
`issue_pair_last` is request-time metadata; it cannot safely release a window
when SRAM returns have latency.

The arithmetic behind the proxy is correct:

`1.1089684997184623 * 37144.673821 / (37144.673821 + 550.998004)
= 1.0927586963044136`.

But that is only a selector-added, replay-only upper bound. The missing
integrated control has a remaining break-even budget of 3,496.601375 um2, and
finite fill/drain plus memory stalls are not in the replay factor. Admission
requires a matched W1/W2 integrated VCS A/B and matched 3 ns DC.

One contract detail should be repaired in a successor overlay: the DC contract
sets `claim_boundary.selector_logic_only_dc=false` despite sealing precisely a
selector logic-only DC result. The milestone SVA should also gain semantic
priority/fallthrough/last assertions and four predicate-specific attack covers.

`docs/359_DATE终局冻结_20260813.md` was not modified and remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
