# M2126 independent source hammer: M2125 TSBG RTL-SAIF window diagnostic

## Verdict

**PASS — exactly one M2127 diagnostic campaign is authorized.** Score: **100/100**; P0/P1/P2 = **0/0/0**.

This authorization is deliberately narrow: one license query, one shared VCS compile, two serial `simv` runs, and two fresh DUT-only SAIF files. DC, PT/PTPX, ICC2, reuse, retry, caller-selected identities, mapped activity, power, energy, speedup, and paper claims remain unauthorized.

## Independent findings

1. The consumed M2119 quarantine and M2120 failure review are exhaustive double-sealed. The failed ordinary SAIF independently reproduces the frozen fingerprint: duration 60,877.5 ns, 93,971 activity records, and 58,277 records with nonzero TX. The execution counts are 1 license query, 1 compile, 1 simulation, 0 accepted SAIF, 0 DC, and 0 PTPX; automatic retry is false. M2120 permits M2125 source authoring only and does not authorize direct execution.
2. M2125 is additive. All 21 M2117 inventory members and all 15 M2125 inventory members are byte-exact. The M2018 RTL, M2051 testbench and two fixtures, M2119 quarantine, M2120 review, and protected docs/359 are unchanged.
3. The runner has one production entry and a fixed source-owned identity. Review, source hashes, predecessor seals, freshness, and collision checks occur before the attempt and license query. The final count equality enforces exactly 1/1/2/2 license/compile/simv/SAIF and zero DC/PTPX, with serial axes and no retry or reuse.
4. AST inspection confirms exactly one `+vcs+initreg+random` in the shared compile command and exactly one `+vcs+initreg+0` plus `+WORKLOAD_SLOT=42` in the single two-axis runtime command. Each axis contributes exactly one fixed selector. No UNIT_DELAY, SDF, force/release, assertion suppression, or other X-coercion option exists.
5. The wrapper observes `full_execute_start_cycle`, takes one explicit negedge, settles 0.01 ns, validates identity/knownness, and stops. It similarly observes the selected completion, takes one explicit negedge, settles, validates the exact ledger, and stops. Each UCLI script selects only its corresponding `.implementation`, enables after the first `run` stop, disables after the second, writes exactly one SAIF, and quits.
6. The parser binds the exact ordinary/TSBG ledgers, requires duration = cycles x 3 ns, requires exactly 93,971 records per axis, checks every TX field and the global sum are zero, checks every record's T0+T1+TX conservation, requires at least 20 toggled records, and requires activity on eight public critical cones.

## Mutation hammer

The following controlled changes were independently rejected:

- ordinary runtime cycle 20,292 -> 20,291;
- scalar weight reads 14,304 -> 14,303;
- runtime duration 60,876.0 -> 60,877.5 ns;
- injected fatal runtime token;
- one nonzero SAIF TX record;
- one T0+T1+TX conservation failure;
- record count 93,971 -> 93,970;
- zero activity on `commit_accept`;
- old M2119 SAIF duration 60,877.5 ns;
- missing M2126 review, before any persistent state or subprocess.

The positive synthetic SAIF contains exactly 93,971 records, TX=0 globally and per record, exact conservation, and nonzero activity on all critical cones.

## Authorization and boundary

M2127 may be launched once with the exact frozen runner and review SHA identities. A failure consumes the authorization and must be quarantined; it cannot be retried. Even a passing M2127 still requires an exhaustive M2128 result hammer and remains a VCS-only RTL activity diagnostic. It cannot be cited as power, energy, mapped-netlist activity, component/system speedup, silicon initialization behavior, or paper-ready PPA. Any later DC/SAIF-map/PTPX campaign requires a new source namespace and another independent source hammer.

No EDA executable, license query, or GPU was invoked by M2126.
