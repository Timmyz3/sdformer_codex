# M210 + M211 independent hammer review

Score: **93/100**.  M210 functionally closes M207's legal bank-48
truncation deadlock, and M211's exact frozen-payload arithmetic reproduces with
zero integer mismatch.  The admitted result is an isolated FC2 sparse-frontend
opportunity count, not a matched hardware speedup or complete FC2/FFN/system
claim.

Independent Synopsys VCS attacks cover: the historical M207 deadlock, M210
packet-bank 48 and window-bank 96 bounds, 737-cycle descriptor hold, stage-0
handoff under a held group, terminal header chaining, 256 partial/zero/full
tails, and two directed protocol attacks.  No assertion failure or unexpected
protocol error was observed.

The exact M211 result is 91,184,539 cycles, saving 1,694,275 cycles from the
nontruncating M209 control model.  Stage 0 is 21,990,740 cycles.  Its remaining
572,258-cycle gap is exactly 199,969 three-descriptor tokens whose partial
second window is not preclosed plus 372,289 one-descriptor tokens with no
predecessor interval in which to hide close/load.

The exact-SHA sealed r2 3 ns DC run is logic-only pre-macro evidence:
20,485.332086
um2, 30,537 leaf cells, 2,773 sequential cells, 92 levels, 2.54 ns critical
path, +0.0008 ns setup slack and +0.0000 ns hold slack.  Ideal clock,
ZeroWireload, no macros, no physical margin, no paper PPA claim.

`docs/359_DATE终局冻结_20260813.md` was not modified and remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
