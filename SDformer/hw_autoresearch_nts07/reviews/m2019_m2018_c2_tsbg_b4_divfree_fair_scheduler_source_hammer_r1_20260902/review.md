# M2019 | M2018 C2/TSBG-B4 divfree fair-scheduler source hammer

Date: 2026-09-02

## Verdict

**PASS, 98/100; P0=0, P1=0, P2=0.** M2018 is suitable for the next,
separately reviewed VCS-source-authoring step. This review does **not** authorize
a VCS compile, `simv`, DC, PT, Formality, GPU work, a license query, or an EDA
attempt/result/release.

The source-level fairness repair is real: both schedule modes elaborate static
wiring into one 192-row live map and then use the same 12-by-16 hierarchical
priority selector, one M803 adapter, one LRU4 weight store, one typed-signed
bridge, the same four Acc24 contexts, and the same commit machinery. The
synthesizable source has no `/`, `%`, `scan_linear_q`, `find_linear`, or old
`active_q/sign_q` cube. This removes the predecessor's mode-dependent runtime
quotient/remainder and 4-to-1-by-768-bit active-cube selection from the physical
comparison.

## Identity and retained boundary

- M2018 RTL SHA-256:
  `96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21`.
- M1995 predecessor SHA-256:
  `2c1a8a7644b359a153decdc3106a8718992d37d54809007b61e184121fcc14fd`.
- M1880 and frozen `docs/359` identities remain exact; `docs/359` is still
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
- The complete parameter-and-port header after the additive module name is
  byte-exact to M1995.
- The instantiated M803 adapter call, commit view, and protocol/debug exposure
  are byte-exact to M1995. Fetch request/response identity, nine-bit signed
  weight widening (including INT8 -128), Acc24 overflow detection, LRU4
  replacement, commit, terminal, done, stale, fault, and reset/recovery paths
  remain present.
- The frozen M1999 result remains only the predecessor's directed behavioral
  VCS evidence. M2018 does not inherit that PASS without its own future run.

## Independent scheduler proof

The independent hammer modeled the old forward scan and the new clear-and-pick
selector for both modes at G1, G12, and G48. It tested empty and full maps plus
128 deterministic random sparse maps per `(mode, geometry)`, 780 comparisons in
total. Every selected `(context, group)` sequence matched:

- mode 0: `(context, group)` order;
- mode 1: `(group, context)` order.

Every selection was at most one-hot, every live row was selected exactly once,
and no selected row repeated. For G12 mode 0, positions 12--47 of each 48-row
context stripe are static zero padding. The hierarchical encoder passes over
that padding combinationally, so it preserves order and adds no `ST_FIND`
cycle. G1 and G48 bounds are likewise exact.

Clearing `row_live_q` is cycle-equivalent to advancing M1995's monotonically
increasing scan pointer because load order forbids a duplicate group inside a
context and execution begins only after all four contexts terminate. The
selected active/sign row is captured into `current_active_row_q` and
`current_sign_row_q` on the same edge as the one-hot live-bit clear. Subsequent
fetch, response, bridge, and commit stalls use only that captured payload, so
backpressure cannot replay or retarget the row.

## Driver, index, and syntax audit

- `row_live_q` has one procedural owner, the single `always_ff`. Its five source
  assignment sites are reset, load, the two elaboration-constant mode mappings
  for selection clear, and done cleanup. Reset/load/find/done are mutually
  exclusive state cases apart from the enclosing reset branch.
- `find_onehot` has one combinational owner. The local-first and block-first
  `seen` chains guarantee zero-or-one hot for all 192 input patterns. Mode 0
  indices are `context*48+group`; mode 1 indices are `group*4+context`; their
  legal maxima are 191. No absent G12/G1 row can assert live.
- `3'(map_ctx)` and `6'(map_group)` are legal IEEE SystemVerilog sized-value
  casts. Both operands are generate-time constants bounded to 0--3 and 0--47,
  respectively, so there is no truncation or runtime index. Exact VCS/DC parser
  acceptance is deliberately **not** claimed by a source-only review and must
  be checked in the future exact-SHA VCS attempt.
- The 12-by-16 selector is common to both elaborated modes. M2018 also contains
  a common one-hot-gated 192-to-1 payload-selection network for the active/sign
  row and metadata. Therefore the accurate claim is that M2018 removes the old
  **mode-dependent** DIV/REM and dynamic active cube and makes the remaining
  selector common; it must not be described as having "zero mux" or zero
  scheduler cost.

## Contract and boundary

The M2018 source contract is internally exact and double sealed. Its execution
ledger is all zero; every current authorization and every performance/PPA/paper
claim is false. The independent hammer passed under Python 3.6 and current
Python and did not launch any EDA or query a license.

This PASS authorizes only authoring an exact-SHA, fail-closed VCS source package
covering M2018 at G1/G12/G48 in both modes. That future package still needs a
different-author review and a separate execution release. DC/PT/FM remain
unauthorized, and no M1866 cycle ratio, same-area result, component speedup,
system speedup, energy, or headline is admitted here.

