# M1468 different-author blind hammer: M1467 C2 debug-access successor

Verdict: **PASS, release-authoring only**. M1467 may advance to a fresh M1469
release author. It is not authorized to launch VCS, simv, PrimeTime or PTPX.

The consumed M1432 campaign is pinned at `SIM_k8_0`, with one VCS compile, one
simv run, zero SAIF and zero PTPX outputs. The exact root cause is the missing
VCS `-debug_access+r` option required by the frozen gate-level UCLI power
command (`UCLI-117`), not an RTL or protocol failure. This hammer did not read
or enumerate M1432's unsealed private build.

M1467 adds exactly one `-debug_access+r` option to the shared compile prefix.
Consequently both the K8 and equal-bandwidth K1x8 compile commands receive the
repair. The same five workloads, mapped netlists/SDCs, library, UCLI/SAIF scope,
PTPX script/corner and expected cycles remain exact-pinned through M1432.

The independent campaign reran all 13 source tests and passed 30/30 static and
authority checks. It rejected 119/119 mutations with zero false negatives: 32
source/execution mutations and 87 exact-contract mutations. Attacks covered
debug-flag deletion/duplication/replacement, axis/case/counter loss, workload,
filelist, top, UCLI, cycle/event, netlist, SDC, library, PTPX script/corner and
SAIF-scope drift, collision/lock/attempt weakening, replace semantics, partial
axis publication, retry and all-SAIF-before-PTPX order.

No EDA tool, license query or production attempt was run. No power, energy,
cycle, performance, system-speedup, PPA or headline claim is promoted.

Next gate: a different author may create M1469, bound to this sealed review.
M1472 must then independently authorize at most one M1467 campaign with no
automatic retry.
