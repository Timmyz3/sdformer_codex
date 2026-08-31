# M518 matched Fixed-T10 versus rank3 logic-only DC static hammer r2

Date: 2026-08-27  
Reviewer role: independent source-only hammer; not the author and not the launch authority  
Overall assessment: **READY FOR ONE SEPARATELY ADMITTED R2 DC ATTEMPT**  
Score: **98/100**; P0/P1/P2 = **0/0/2**

No DC, VCS, Formality, PT, PTPX or open-source EDA command was run. The author
package, RTL and `docs/359` were not modified.

## Verdict

The r1 P1 is closed. Root may create one new, double-sealed r2 launch admission
at
`contracts/m518_matched_fixed_rank3_logic_only_dc_launch_admission_r2_20260827.json`.
The admission must bind this review, the sealed r1 no-launch verdict, and every
r2 source/tool/library/VCS identity. This review does not itself run or admit
DC results.

The r1 runner/contract remain permanently unauthorized. Their launch-admission
path was absent during this review and must remain absent.

## r1 P1 closure

The r2 Tcl redirects a single complete precompile construction block into
`precompile_build.rpt`. The block textually contains, in order: common two-file
`analyze -define SYNTHESIS`, selected-top `elaborate`, current-design validation
and selection, `link`, `uniquify`, slow-to-fast `set_min_library`, operating
condition, the common SDC, ZeroWireload and `set_fix_hold`.

Separate redirects then produce `check_design_precompile.rpt` and
`check_timing_precompile.rpt`. The Tcl reads all three files and counts both
`TIM-209` and `OPT-150` over their concatenation. A nonzero count writes exact
failure provenance and executes the sole `exit 36`. The only `ungroup` and all
four compile commands occur later, inside the zero-count `else` branch. Static
counts are: one diagnostic gate, one `exit 36`, one `ungroup`, and four compile
commands.

The runner independently requires all three nonempty reports, the exact
three-source provenance line, zero diagnostic tokens in each individual
source, zero diagnostic tokens in the complete DC log, and the PASS-only Tcl
terminal. Thus an omitted source, forged aggregate or diagnostic outside the
aggregate cannot publish a canonical result.

## Regression audit

### Exact identity and predecessor exclusion

All r2 contract/runner/Tcl/filelist/SDC/RTL/tool/library hashes matched. The r2
runner also verifies the sealed r1 verdict and outer seal, and requires r1
status `NEEDS_REVISION__R1_LAUNCH_NOT_AUTHORIZED`, r1 launch authority false
and r1 DC false before resource sampling. The r1 and r2 launch admissions,
r2 canonical result and r2 attempt sentinel were all absent at review time.

All Fixed and rank3 VCS result/review member manifests and outer seals verified
recursively. `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

### Fair matched boundary

An independent parser recovered exactly 50 synthesis-visible ports for each
top after excluding only `M518_VCS_V06_HARNESS`; all ordered
direction-width-name tuples match. Both points analyze the same two-file
corpus and use the same Tcl, SDC, slow/fast DBs, 3.000 ns clock,
0.200/0.050 ns setup/hold uncertainty, 0.250 ns input/output delay, 0.100 ns
input transition, 0.010 pF output load, max fanout 24 and flatten/mapping
sequence.

The sealed cycle anchors remain Fixed N1/N4 = 29/80 and rank3 N1/N4 = 24/39.
The raw rank3/Fixed throughput ratios are therefore 1.208333x and 2.051282x.
The N4 throughput/mm2 ratio is correctly defined as
`80*A_fixed/(39*A_rank3)`, using only the two new same-run areas. No old M289
area is read or used as a denominator.

### Fail-closed execution and claim boundary

Caller-pinned runner/admission hashes, collision checks, exact identities,
all predecessor/VCS seals and the 50-port source comparison precede three
resource samples and attempt consumption. Each sample requires at least
64 GiB commit headroom, 128 GiB MemAvailable, 32 GiB SwapFree and clean cgroup
OOM counters. Failure after consumption seals a quarantine; canonical
publication requires clean Fixed and rank3 points plus equal 50-port mapped
counts.

Per point, setup and hold must be MET, all five constraint classes must be
clean, check-design/check-timing must return one, the unconstrained-endpoint
audit must be present and clean, and latch/multidriver/unresolved/black-box
checks must all be zero. The result must contain nonempty mapped Verilog, SDC,
DDC and SVF and report zero macros.

Even a clean result is only a flattened zero-macro logic-only standalone ATLIF
DC/STA comparison pending a separate receipt review. It is not macro-inclusive
PPA, power, energy, trained-rank3 accuracy, workload/system speedup,
paper-ready PPA or a headline.

## Nonblocking P2 observations

1. The attempt marker still says `CONSUMED_AT_FIRST_DC_LAUNCH` although it is
   atomically published before the first `dc_shell` process. This is
   conservative one-shot behavior but imprecise provenance wording.
2. Runtime resource failure is latched and rejects the result after the DC
   child exits, rather than terminating the child immediately. Result
   integrity remains fail-closed; resource protection is delayed.

