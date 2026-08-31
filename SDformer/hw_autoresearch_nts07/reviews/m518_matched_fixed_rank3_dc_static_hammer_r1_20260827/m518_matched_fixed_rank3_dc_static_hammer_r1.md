# M518 matched Fixed-T10 versus rank3 logic-only DC static hammer

Date: 2026-08-27  
Reviewer role: independent source-only hammer; not the author and not the launch authority  
Overall assessment: **NEEDS REVISION — do not sign or run the r1 DC campaign**  
Score: **94/100**; P0/P1/P2 = **0/1/2**

No DC, VCS, Formality, PT, PTPX, or open-source EDA command was run. The author
contract, Tcl, shell runner, filelist, SDC, RTL and `docs/359` were not modified.

## Blocking finding

### P1-1 — the precompile `OPT-150` gate does not observe the precompile build transcript

The contract and review request require both `TIM-209` and `OPT-150` to hard-exit
before `ungroup` or any compile command. The Tcl at lines 51–59 writes
`check_design_precompile.rpt`, redirects `check_timing` into
`check_timing_precompile.rpt`, but computes **both** diagnostic counts from only
the latter file. Consequently `precompile_opt150_count` cannot see an
`OPT-150` emitted earlier by `link`, `uniquify`, `set_min_library`, operating
condition/SDC application, or wire-load setup. The shell later scans the full
`dc.log` (runner lines 345–346), so a final result containing `OPT-150` would be
quarantined, but only after the Tcl may already have executed `ungroup` and
`compile_ultra`. That is fail-closed for publication, but it does not satisfy
the explicitly contracted **precompile** stop boundary.

This is not hypothetical tool-routing trivia: existing DC transcripts in this
repository show `OPT-150` in `dc.log` while the corresponding
`check_timing_precompile.rpt`-derived counter is zero. The established repair
pattern is to redirect the precompile build commands to a report, combine that
text with both precompile check reports, count `TIM-209`/`OPT-150` over the
combined text, and retain M518's explicit `exit 36` before line 88. Any repair
changes the Tcl, runner and contract identities and therefore requires a new
author package and a fresh independent static review.

**Admission consequence:** root must not create
`contracts/m518_matched_fixed_rank3_logic_only_dc_launch_admission_r1_20260827.json`
for the reviewed identities. No r1 DC attempt is authorized.

## Verified checks

### Identity and prerequisite seals

All author-declared source, tool and library identities matched the live files.
Both Fixed and rank3 VCS result/review manifests and outer seals verified
recursively. The live DC launcher is V-2023.12-SP3 at the frozen SHA. The author
request member manifest and outer seal also verified. `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

The canonical DC result and one-shot attempt sentinel were absent at review
time. No launch-admission file existed.

### Matched comparison boundary

An independent parser recovered exactly 50 synthesis-visible ports from each
top after excluding only `M518_VCS_V06_HARNESS`. Direction, symbolic width and
name tuples are identical and ordered identically. Both points analyze the
same two-file corpus, select only the top name, and otherwise share the same
Tcl, SDC, slow/fast databases, 3.000 ns clock, 0.200/0.050 ns setup/hold
uncertainty, 0.250 ns IO delays, 0.100 ns input transition, 0.010 pF output
load, max fanout 24, flattening and mapping sequence.

The different five-beat Fixed configuration and six-beat rank3 configuration
are algorithm-owned state/payload costs, not hidden interface differences;
the sealed clean-cycle measurements start at the first accepted configuration
beat and therefore charge each point its real framing cost.

### Cycle and area-efficiency denominators

The sealed VCS anchors independently reconcile to Fixed N1/N4 = 29/80 and
rank3 N1/N4 = 24/39. Thus the raw throughput ratios are 29/24 = 1.208333x and
80/39 = 2.051282x. The proposed primary N4 throughput/mm2 ratio is correctly
derived as

`((4/39)/A_rank3) / ((4/80)/A_fixed) = 80*A_fixed/(39*A_rank3)`.

Both areas are read only from the two new outputs of this same campaign. No
historical M289 area appears in the runner or formula. N1 and N4 remain
separate, with no steady-state extrapolation.

### Postcompile fail-closed gates

Subject to P1-1, each point separately requires: zero DC return code; a unique
PASS Tcl terminal and no explicit-failure marker; setup and hold MET with no
VIOLATED path; exactly five clean constraint sections (max/min delay,
capacitance, transition and fanout); `check_design=1`; `check_timing=1` with an
explicit unconstrained-endpoint check; no inferred latch, multiple driver,
unresolved reference or black box; zero macros; nonempty mapped Verilog, SDC,
DDC and SVF; and exactly 50 postcompile ports. Canonical publication occurs
only after both points pass and their port counts agree.

### Resource, attempt and quarantine order

Caller-pinned runner/admission identities, canonical/attempt/tool collision
checks, all frozen identities, prerequisite seals and the source port equality
check precede resource sampling. Three samples each require commit headroom >=
67,108,864 KiB, MemAvailable >= 134,217,728 KiB, SwapFree >= 33,554,432 KiB,
and clean cgroup OOM counters. Only after those checks does the runner create a
work directory and atomically publish the sealed one-shot sentinel. Any later
failure or signal keeps the attempt consumed and publishes a sealed quarantine;
the canonical directory is published only after a sealed two-point result.

## P2 observations

1. The attempt marker text says `CONSUMED_AT_FIRST_DC_LAUNCH`, but the marker is
   created at runner lines 265–274, before the first actual `dc_shell` launch at
   lines 316–318. Early consumption is conservative and cannot create a false
   result, but the provenance label should say prelaunch consumption or be
   moved immediately before the launch.
2. Runtime resource sampling latches a failure but does not terminate the live
   DC child immediately; the point is rejected only after the child exits.
   This preserves result integrity but weakens the resource-protection intent.

## Claim boundary

Even after repair and a clean run, this campaign can admit only a matched,
flattened, zero-macro, logic-only DC/STA comparison of two standalone ATLIF
tops. It does not admit macro-inclusive PPA, power, energy, trained-rank3
accuracy, workload/system speedup, paper-ready PPA or a paper headline. A new
independent receipt review remains mandatory.

