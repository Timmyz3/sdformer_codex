# M1274 — C2 mapped-gate SAIF/PTPX chain read-only evidence audit

## Verdict

`STOP_CURRENT_POWER_CLAIM__RETAIN_M903_LOGIC_ONLY__NO_FRESH_ONE_SHOT_RELEASE`

The M979--M1045 chain successfully closed source identity, clean VCS startup,
license checkout, and the UCLI `power` protocol.  It never completed one
production mapped-gate case and never emitted a production C2 SAIF.  No PT or
PTPX result exists for this C2 chain.  Consequently C2 power and energy remain
false and must not enter the DATE table.

The last real *capability* success in the requested range is M1045's independent
tiny UCLI-power probe: compile and simulation returned zero and emitted a
2,106-byte, 24 ns SAIF with the frozen top and DUT hierarchy.  The last real
*production-path* progress is the subsequent M1046 K1 mapped compile/link plus
header acceptance; case 0 then reached the watchdog with zero result/token and
zero production SAIF.  These two meanings must not be conflated.

Downstream evidence has superseded the repair plan that was current at M1045.
The final sealed disposition is M1155: stop observation expansion on the old
frozen netlist and retain only M903's logic-only component evidence.  No fresh
one-shot release is currently authorized.

## Intended claim and evidence grain

The intended measurement is a **C2 component logic-energy comparison** across
K1, K8 and equal-bandwidth K1x8, aggregated over five frozen directed cases per
axis.  It is not a full-network, trace-weighted, macro-inclusive, post-layout or
energy/frame measurement.  Even a future successful run would require the
M974 boundary: TT 0.9 V/25 C, 333.33 MHz ideal clock, ZeroWireload, no SPEF,
case energy computed as `P_total[mW] * duration[ns]`, and all five cases summed.

## Chain reconstruction

| Milestone | Furthest proven point | Failure / boundary | Citable power? |
|---|---|---|---|
| M979/M1001 | Three-axis x five-case source and rekey chain statically closed | No EDA, SAIF, PT or PTPX | No |
| M1003/M1004 | First release sealed | Wrong M1001 contract SHA; STOP before EDA | No |
| M1013/M1018 | Entered `COMPILE_k1` | Clean environment omitted `VCS_HOME`; no `simv`, zero cases, zero SAIF | No |
| M1022/M1029 | Full64 VCS frontend started | Clean environment removed both license routes; no `simv`, zero cases, zero SAIF | No |
| M1030--M1033/M1043 | Tiny license checkout passed; production K1 compile/link created `simv` | First case failed before simulation with `UCLI-117`; production compile omitted `-debug_access+r` | No |
| M1044/M1045 | Independent tiny `-debug_access+r -lca` UCLI power probe passed; 2,106 B, 24 ns SAIF | Tiny DUT only; M1045 explicitly did not run production | No |
| M1046/M1050 | Tiny production preflight passed; K1 mapped compile/link and header acceptance succeeded | Case 0 watchdog; gate-level uninitialized-state X propagation; zero completed cases and zero production SAIF | No |

The failure progression is not a sequence of contradictory diagnoses.  Each
successor reached one stage further: startup path, license checkout, UCLI
capability, then mapped functional execution.  The evidence is complete at
each failure boundary and incomplete beyond it.

## Current terminal state after the requested range

The additive M1058 reset-hygiene RTL passed RTL VCS but its fresh M1080 mapped
run reproduced the same post-header watchdog.  Observation successors then
localized the first stable failure further:

- M1151's 128-cycle atomic bitmap first sees functional `protocol_error` and
  derived `fault` become X at cycle 3; request/result shadow counters become X
  only later.  The observation shadow is therefore not the first root.
- M1154/M1155 require thirteen stable semantic taps to distinguish the original
  scalar-memory endpoint from a valid-qualified endpoint and the internal
  consistency/protocol cones.  The frozen netlist retains five fault-Q taps but
  has optimized away eight required paired-accept/consistency/protocol taps.
  Binding anonymous `n*` nets was correctly rejected as non-reproducible.

Therefore the old frozen netlist is permanently closed for further diagnostic
or production activity.  M1146 is consumed and cannot be retried; M1154's
namespace is fresh but is a fail-closed source namespace, not an execution
release.

## What is citable today

The only admitted C2 physical/performance row remains M903:

- TSMC 28 nm standard-cell, 3.000 ns ideal-clock, ZeroWireload, logic-only
  pre-macro DC setup/area;
- K1/K8/K1x8 areas `124620.173180 / 131086.241193 / 585479.153645 um^2`;
- directed component K8 versus equal-bandwidth K1x8: `1913` versus `1945`
  cycles, `1.016728x` cycle ratio, `4.541078x` throughput/mm2, and `77.6104%`
  lower logic cell area.

Those numbers are component/directed and logic-only.  Hold is diagnostic, macro
count is zero, and power, energy, PPA, system speedup and headline claims are
false.  The M1045 tiny-SAIF result is only a tool-capability receipt; it is not
C2 design activity and cannot be converted to power.

## Data-quality findings

1. **Critical — production activity completeness is zero.**  Across the
   production M1013/M1022/M1033/M1046 directories there are zero production
   SAIF files and zero completed gate cases.  The only C2-chain SAIF is the tiny
   M1046 preflight file.  Any C2 power number sourced from this chain would be
   fabricated or mis-grained.
2. **Critical — functional integrity is not established at mapped gate.**  A
   mapped binary can compile and link yet fail four-state functional execution;
   that blocks SAIF admission before PT/PTPX.
3. **High — tiny preflight and production are different grains.**  Both use the
   UCLI power protocol, but only the tiny DUT completed.  Its 24 ns duration and
   hierarchy cannot substitute for a C2 axis/case window.
4. **High — old authorization is stale.**  M1045's one M1046 authorization and
   M1150's later one M1146 authorization were both consumed.  Their hashes are
   audit history, not present execution pins.
5. **Low — Synopsys installation is presently discoverable.**  Without exposing
   route values, both license-route variables are nonempty; exact VCS and
   `vcsMsgReport` binaries exist; `dc_shell` is the expected symlink to
   `snps_shell`.  Tool startup is no longer the dominant blocker.

## Minimum legal next step

There is **no executable one-shot release now**, so the next legal step is
source authoring, not VCS/PTPX execution:

1. Create a new RTL namespace that explicitly preserves all thirteen semantic
   taps before synthesis: four paired request/response accepts,
   `consistency_fault_now/q`, core/adapter protocol errors, and five retained
   component fault-Q signals.  Observation outputs must have no fan-out into
   functional flow control.
2. Implement an executable dual-DUT/atomic-first-X TB.  Both DUTs receive the
   same stimulus; one retains the original memory endpoint and one qualifies
   ready/accept with request-valid and known payload/slot.  No force, initreg,
   delayed checker, X coercion or anonymous-net binding is allowed.
3. Different-author source hammer, then exactly one fresh RTL-VCS -> DC ->
   mapped-VCS diagnostic namespace.  If the qualified endpoint does not remove
   or uniquely localize the first X, terminate this campaign and keep M903 only.
4. Only after five K1 cases pass without X and match `259/737/3153/7569/14`
   may a new three-axis x five-case SAIF release be authored.  PT/PTPX remains a
   separate downstream admission after activity scope, duration, TX and cone
   coverage pass.

For the DATE schedule, this is a substantial functional-closure campaign, not
a one-command power run.  It should not block use of the already admitted M903
area-efficiency result, and no legacy selected-slice power number may be
relabelled as C2 power.

## Exact evidence identities

- M903 admitted result review JSON:
  `reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829/review.json`,
  SHA256 `89785b3a06fc5981cb1e652bce18c4ab3853809ccf6dee7d1b96a65bd018b10a`.
- M872 canonical DC outer-seal file:
  `dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829/SHA256SUMS.seal.sha256`,
  SHA256 `0c9da50fc21c97b66f192779e10a50de2319ddc77da51e236ed6ee786aafcd5e`.
- M974 measurement-plan review MD:
  `reviews/m974_m903_m872_c2_three_axis_pt_saif_ptpx_first_principles_r1_20260829/review.md`,
  SHA256 `8345689cd70ce37f03503efaaf2aa612772090434bc5b81fa8cb0e7d7b829ec7`.
- M1045 release-hammer review JSON:
  `reviews/m1045_m1044_m1043_m1046_c2_saif_release_hammer_r1_20260829/review.json`,
  SHA256 `f2007801bc1cfa8d064e8332c221fcfd96ff105b03dcbedcc4ef207950c46ed5`;
  outer-seal-file SHA256 `7c1dcdb02f1c259e3150b56ba995b397e0f65917b779f4f85b0a756b66c6011c`.
- M1046 tiny preflight SAIF:
  `results/m1046_m1001_c2_ucli_power_preflight.2027456.sealed/tiny.saif`,
  SHA256 `f08876128ccc78ee1c15001659227d8e47c9f9cee03b2061b0e0fd73eafe78f8`;
  its directory outer-seal-file SHA256 is
  `f9bac1e8638e3b82e4aed19f7fec8405b292d077aee04197c6c60453a508bdb7`.
- M1046 production failure quarantine:
  `results/m1046_m1001_c2_three_axis_mapped_gate_saif_r5_20260829.failed_or_incomplete.2027456.quarantine`,
  outer-seal-file SHA256
  `cb6f6b69e2cb51d60556f5bcb8a7748865f72ee2bdbe2f178925a624d9e9d705`.
- M1050 failure audit JSON SHA256:
  `de6802c2dd139f63c90036aed08c35107649c900c4a736e39390fbbd463bcd8b`;
  outer-seal-file SHA256
  `bc239844a71b5c017002ea1f6a756143d3c58b5ebf39d6a5499c76228da188bb`.
- M1151 terminal functional-X audit JSON SHA256:
  `08f9041acc9671f76f5c94a87c5ceba4797c8bfe9f8cdae41bbe9647ea7d3411`;
  outer-seal-file SHA256
  `72bf8c7500a45961aefada1cb3b720bfc0b357eb7e08257379015fb6c1288c5f`.
- M1155 final STOP review JSON SHA256:
  `c7f057e133cf0ce99563dc672ada5c41b594dea44a640ab28f9589718f388716`;
  outer-seal-file SHA256
  `f27a738dc55a06de9d9cb906c395b9ec94dcfd7b0fd0ba84527bec29700e039d`.

Current tool identities, observed without recording license-route values:

- VCS: `0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287`;
- `vcsMsgReport`: `b34e06a92b05856532f868d32c0c81f1708506096856ad9a97bd27e2bd60215b`;
- DC launcher raw link: `dc_shell -> snps_shell`;
- resolved `snps_shell` payload:
  `23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2`.

`docs/359_DATE终局冻结_20260813.md` was not modified and remains SHA256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
