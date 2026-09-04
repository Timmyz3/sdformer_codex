# M100 M99 phase-slack logic-only DC independent hammer

Date: 2026-08-24

Status: `GO_SCOPED_STANDALONE_PARTIAL_PASS_NO_GO_FULL_CONTRACT_FMAX_PPA_SYSTEM`

This review is read-only with respect to the contract, RTL, scripts, M97/M100
run directories, and `docs/359`.  It writes only this independent review
directory.

## Verdict first

M100 is a strong, auditable **standalone logic-island partial pass**.  Under the
frozen, identical DC flow and external-bank port cut, M99 reduces mapped
standard-cell area from 34,758.486144 to 13,830.642035 um^2.  Equivalently,
M97/M100 is **2.513150586650x**, M100 retains **39.7906916248%** of M97 cell
area, and the reduction is **60.2093083752%**.  M100 also removes the specified
unrolled 32-bit/4-bit audit-resource families and meets the fixed 3.000 ns
constraint.

The frozen 35% area gate is nevertheless missed correctly: 13,830.642035 is
1,665.1718846 um^2 above the 12,165.4701504 um^2 cap.  Therefore the result is
not a full contract pass.  It is also not Fmax, complete lookup timing,
macro-inclusive PPA, power/energy, or system speedup evidence.

Score: **92/100 for the scoped M100 milestone**.  DATE/paper claim readiness of
this artifact alone is materially lower because SRAM, routed timing, power and
cycle-to-time composition are intentionally absent.

Severity: `P0=0 / P1=4 / P2=5`.

## Evidence integrity

Running the M100 `evidence_manifest.sha256` from the repository root verifies
all **20/20** listed files, with zero failure and no duplicate digest.  The
manifest uses repository-root-relative paths; running it from inside the run
directory produces expected path-not-found errors and is not a hash failure.

The current contract, launcher, Tcl, filelist, SDC, M82 RTL, M99 RTL, M97
receipt and M97 evidence-manifest hashes all match the launch-time identities
recorded in `admission.txt`.  The M97/M100 Tcl files differ only in the top
name, and the filelists differ only in M85 versus M99; the M82 source and SDC
are common.

DC V-2023.12-SP3 returned backend code zero, emitted nonempty mapped
Verilog/SDC/DDC, contains no line beginning `Error:`, and ends normally.  The
historical launch/backend-awaiting markers coexist with the final partial-pass
receipt and marker.  The canonical terminal state is the receipt plus
`RUN_PARTIAL_PASS_NOT_FULLY_ADMITTED.txt`; no `RUN_COMPLETE` is expected because
the full gate set failed.

## Independent numeric reconstruction

### M100 area and cell counts

The area and QoR reports reconcile exactly:

| Quantity | Reconstructed value |
|---|---:|
| combinational cells | 13,498 |
| sequential cells | 3,131 |
| leaf cells | 16,629 |
| hierarchy cells | 1 |
| area-report cells | 16,630 |
| combinational area | 7,518.545933 um^2 |
| noncombinational area | 6,312.096102 um^2 |
| total cell area | **13,830.642035 um^2** |
| macro/black-box count and area | 0 / 0.000000 um^2 |
| net/physical total area | undefined under ZeroWireload |

`7,518.545933 + 6,312.096102 = 13,830.642035`, and
`13,498 + 3,131 = 16,629`; adding the one hierarchical cell gives the area
report's 16,630.

### Timing and constraints

- Period 3.000 ns, setup uncertainty 0.200 ns, hold uncertainty 0.050 ns.
- Setup worst path is `lookup_pattern[3]` to
  `m82_stream/output_values_q_reg[395]`; arrival is 2.7714 ns and slack is
  **+0.0054 ns MET**.
- QoR reports 2.52 ns critical path length and 68 logic levels.  The path
  length excludes the 0.250 ns input delay, so it need not equal the full data
  arrival.
- Hold worst path is an M82 output-register self path; slack is
  **+0.0001 ns MET**.
- Setup/hold TNS and violating-path counts are zero; transition,
  capacitance and fanout violation counts are zero.
- Three unconnected M82 observability ports remain: `beat_width[3]`,
  `beat_accept`, and `collecting`.  They are not unresolved production
  references.

The pass is real at the fixed target but has only 5.4 ps setup and 0.1 ps hold
reported margin.  With ideal clocks, ZeroWireload, zero macros and a cut SRAM
path, it is not a robust Fmax or physical timing point.

### M97-to-M100 arithmetic

All headline arithmetic in the receipt and partial-pass marker is reproducible
from the two total cell-area reports:

```text
34758.486144 / 13830.642035 = 2.513150586649537
1 - 13830.642035 / 34758.486144 = 0.6020930837522266
M100 area fraction = 0.3979069162477734
```

Thus **2.51315058665x** is an area ratio, not an acceleration factor, and
**60.2093083752%** is the same-flow logic-only cell-area reduction.

The remaining receipt ratios also recompute: combinational cell count
31,711/13,498 = 2.3493110090x, leaf cells 34,819/16,629 =
2.0938721511x, data arrival 3.9579/2.7714 = 1.4281229703x, and M100 has
23 more sequential cells than M97.  The arrival ratio compares two different
critical endpoints and is not an Fmax ratio.

The local hierarchy numbers have a harmless rounding subtlety.  Exact
postcompile hierarchical-reference areas are 10,136.070043 um^2 for M97's M82
and 10,134.936043 um^2 for M100's M82.  The receipt subtracts the six-decimal
displayed hierarchy values 10,136.070 and 10,134.936, obtaining local areas
24,622.416144 and 3,695.706035 um^2 and local ratio 6.6624390335x.  Using the
exact reference areas gives 24,622.416101 and 3,695.705992 um^2, ratio
6.6624391110x and reduction 84.9904819641%.  This 0.000043 um^2 rounding does
not affect any total-area claim.  The two M82 mapped areas also differ by
1.133957 um^2 (0.0112%) because boundary optimization occurs in different top
contexts; they should be called common RTL/flow, not an exactly equal mapped
area constant.

### Audit-resource counts

The resource receipt is reproducible when its terms are read correctly as
**precompile synthetic-operator occurrence counts**, not mapped arithmetic
cell counts:

- M97 `references_precompile.rpt` has 126 + 378 = **504** signed 32-bit
  add occurrences, **254** signed 32-bit greater-than occurrences, and
  **128** signed 4-bit greater-than occurrences.
- M100 has no 32-bit add family, no 32-bit comparison family, and no 4-bit
  comparison family in the corresponding precompile references: **0/0/0**.
- M100 postcompile resources do contain one 4-bit `DW01_add`; it is an adder,
  not the contract's 4-bit comparison family, so it does not violate the gate.
- M100 has zero unresolved production references and zero macros.

The reported resource gate therefore passes and supports removal of M85's
unrolled audit cone.  It does not mean that M100 has no adders or comparators.

## The 35% frozen gate

The failure decision is arithmetically exact:

```text
cap = 34758.486144 * 0.35 = 12165.4701504 um^2
observed - cap = 1665.1718846 um^2
observed / M97 = 39.7906916248% > 35%
excess / cap = 13.6876903565%
```

It is also structurally coherent, but very aggressive.  M100's M82 occupies
10,134.936043 um^2, or 73.2789% of M100 and 29.1582% of M97.  Keeping that M82
fixed leaves only 2,030.5341074 um^2 for the wrapper at the 35% cap.  The
current wrapper is approximately:

| Local M99 wrapper component | Area |
|---|---:|
| combinational | 2,449.8180 um^2 |
| noncombinational, 618 FFs | 1,245.8880 um^2 |
| total | 3,695.7060 um^2 |

The wrapper therefore needs another **45.0569%** reduction.  If all 618 FFs
and their 1,245.8880 um^2 remain, wrapper combinational area must fall to at
most 784.6461074 um^2, a **67.9712%** combinational reduction.  Because the
cap remains above common M82 plus the local sequential area, it is not
mathematically impossible; however there is no present lower-bound proof that
the required remaining mapper/audit combinational logic can fit.  The gate is
reasonable as a stretch target, not as evidence that the achieved 60.2%
reduction lacks value.

## Claim boundary

### GO as a scoped standalone-module result

The following is defensible when all qualifiers travel with it:

> In an exact same-flow TSMC28 DC logic-island comparison with ideal clock,
> ZeroWireload, zero macros and the documented external-bank port cut, the M99
> one-entry-per-cycle metadata compiler removes M85's 504/254/128 unrolled
> audit-resource signature, reduces mapped standard-cell area by 60.2093%
> (M97/M100 area ratio 2.51315x), and meets the fixed 3.000 ns constraint with
> +0.0054 ns setup and +0.0001 ns hold slack.  The prior phase-metadata poison
> bottleneck migrates to the retained lookup/M82 datapath.

This is a meaningful architecture/area and bottleneck-removal claim.  It is an
audited partial-pass result, not `contract_all_gates_pass=true`.

### NO-GO claims

- **Full M100 contract admission:** the frozen 35% area gate failed.
- **2.513x acceleration or performance:** 2.513x is an area ratio only.
- **Fmax or an M97/M100 Fmax ratio:** one fixed target, especially with M97
  failing that target, does not locate either design's minimum passing period.
- **Complete PWP lookup timing:** address outputs and bank-word inputs terminate
  on opposite sides of an excluded eight-bank SRAM.
- **Paper/physical PPA:** no SRAM macro area/timing, routed interconnect, CTS,
  extracted parasitics, activity power or energy is present.
- **M88/module/system acceleration:** M100 has not yet been composed into an
  absolute-time M88 schedule; no full-network FPS, energy or system speedup is
  admitted.

## M101 recommendation

**Prioritize a matched M97/M100 Fmax sweep and then compose it with M88.  Do
not make another metadata-area-only rewrite the primary M101 task.**

The reason is Amdahl-like: the local M99 wrapper is only 26.72% of M100 after
M100, while common M82 is 73.28%.  Even deleting the entire wrapper leaves
10,134.936043 um^2.  Passing the 35% stretch gate requires 1,665 um^2 more, but
that would improve an already strong area result without producing a
performance claim.  The present critical path has already migrated out of the
metadata audit into lookup/M82, so further audit serialization/compression is
unlikely to improve clock rate unless it changes the retained lookup path.

M101 should:

1. Recompile both M97 and M100 independently at a frozen period grid and then
   binary-search the fail/pass bracket.  Keep DC version, DBs, corners, SDC,
   port cut, M82, compile passes and hold repair identical.  Record the last
   failing and first passing period, setup/hold WNS/TNS, area, critical
   endpoint and logic levels.  A single re-constrained netlist is insufficient;
   optimization must be rerun per period.
2. Call the result **logic-island Fmax under this port cut**, not physical
   Fmax.  M97 remains an area-valid but 3 ns timing-invalid point until this
   matched sweep creates its own passing bracket.
3. Feed the two bracketed clocks into an M88 absolute-time model.  Keep DRAM
   bandwidth fixed in GB/s, or model a separately clocked DRAM interface; do
   not retain a fixed 32 B/core-cycle while changing core frequency, because
   that silently scales physical bandwidth with Fmax.
4. Preserve M88's 128 parser cycles and one commit.  Admit zero new exposed
   M99 cycles only for inactive-slot concurrent preparation.  Use the
   independent conservative phase-preparation margin of at least 12,634
   cycles; the production/direction value 12,645 and the conservative audit
   bound differ slightly but both preserve zero modeled midstream refill
   stalls.
5. Report both cycle count and absolute time.  If cycles are unchanged and
   M100's bracketed Fmax is higher, the product yields a scoped module
   throughput result.  If the Fmax gain is weak, retarget the now-visible
   68-level `lookup_pattern -> M82 output_values_q` cone with a registered
   descriptor/predecode or M82 pipeline change and charge any added latency/II
   in M88.

Further metadata area work is secondary and justified only if the 35% frozen
gate itself is a required admission objective.  If pursued, optimize the
2,449.818 um^2 local combinational cone first; replacing the 592-bit metadata
state with a macro changes the comparison scope and requires a matched
macro-inclusive A/B rather than being counted as free compression.

## Findings and disposition

### P0

None.

### P1

1. `M100-H-P1-01`: the strict 35% area gate is missed; preserve full-contract
   NO-GO and the partial-pass marker.
2. `M100-H-P1-02`: +0.0054/+0.0001 ns at one ideal-clock ZeroWireload point
   admits fixed-3ns logic-island timing only, not Fmax or robust margin.
3. `M100-H-P1-03`: M97 is a same-flow cell-area denominator, not a timing
   baseline; neither 2.513x area nor 1.428x critical-arrival ratio may be
   promoted to acceleration.
4. `M100-H-P1-04`: the SRAM path, macro PPA/power and M88 absolute-time
   composition are absent; complete-module, system and paper claims remain
   NO-GO.

### P2

1. `M100-H-P2-01`: local hierarchy ratios use rounded six-decimal M82 areas;
   use exact total areas for the main claim and label local ratios approximate.
2. `M100-H-P2-02`: common M82 RTL maps 1.133957 um^2 differently between top
   contexts; do not imply identical hierarchical mapped area.
3. `M100-H-P2-03`: resource values are synthetic-reference occurrences, not
   physical arithmetic-unit counts; one 4-bit adder remains and is outside the
   4-bit-comparator gate.
4. `M100-H-P2-04`: the evidence manifest must be checked from repository root,
   and historical awaiting markers must not override the terminal receipt.
5. `M100-H-P2-05`: M101 should use the conservative independent M88 overlap
   margin of 12,634 cycles when claiming zero modeled refill stalls rather than
   laundering either overlap assumption into an implemented controller claim.

## Final GO / NO-GO matrix

| Decision surface | Verdict |
|---|---|
| manifest integrity, 20/20 | GO |
| backend completion and mapped artifacts | GO |
| fixed-3ns logic-island setup/hold measurement | GO |
| resource-family removal | GO |
| qualified same-flow area ratio/reduction | GO |
| scoped standalone M99 architecture result | GO, partial-pass wording required |
| full frozen M100 contract | NO-GO |
| Fmax or frequency acceleration | NO-GO pending matched sweep |
| complete lookup/SRAM PPA, power or energy | NO-GO |
| M88/module absolute-time speedup | NO-GO pending composition |
| full-network/system/DATE headline | NO-GO |
| M101 matched Fmax sweep plus fixed-bandwidth M88 composition | GO, highest priority |

