# M1109 independent audit of M1091r3 mapped-observation failure

Verdict: **M1091r3 consumed its sole attempt and failed closed. Do not retry that namespace. The first failure is an X-reconvergent mapped implementation of a synchronous-reset debug counter, not a 128-cycle-window, memory-model, retained-payload, or UNIT_DELAY failure.**

## Evidence trust and frozen state

The audit read the quarantine and consumed-attempt directories without invoking DC, VCS, simv, or the M1091r3 engine. Both original seals recompute exactly:

- attempt: 1 manifest member, manifest `fe23dada...`, outer-seal-file `615b231d...`;
- quarantine: 112 manifest members, manifest `35b94e8e...`, outer-seal-file `80594858...`;
- the quarantine's sole symlink, `mapped_vcs/csrc/_3721051_archive_1.so`, resolves to a regular member inside the same sealed directory and its followed bytes match the manifest;
- `attempt.json` is `d5e0ef7e...`; it records exactly one DC attempt, one mapped case, no activity file, and no random initialization;
- `failure.json` is `266da458...`; status is `FAILED_DIAGNOSTIC_DO_NOT_CITE` in phase `FRESH_MAPPED_VCS_CASE0_SHORT_128`;
- `docs/359` remains `dedde7ce...`.

The mapped simulation executable returned 0 even after `$fatal`, but the required PASS token is absent. The engine checks both return code and token and therefore correctly quarantined the run. A return-code-only consumer would misclassify this case.

## First concrete failure and observable boundary

At 16.5 ns, window cycle 0, the first unknown is:

`obs_service_group_count = 0000x0x000000x0000x0000x0000x00x`

The unknown bit indices are `[27, 25, 18, 13, 8, 3, 0]`. The testbench checks 22 observation signals in a fixed sequence and fails on the first unknown. Consequently, exactly the 11 earlier checks are known at this edge: header/raw accept, busy, protocol/numeric/stale/fault, memory request/response accept, FIFO count, and outstanding count. The ten checks after group count were not executed and cannot be classified from this log. No stage line was emitted.

The 3 ns clock has five reset-active positive edges. Reset is released on the 15 ns negative edge, 1.5 ns before the header-accept/check edge at 16.5 ns. Thus the failure is not evidence that the observation window was too short.

## RTL-to-mapped cone

The M1090r3 wrapper introduces no group counter: it directly connects `debug_group_accept_count` to `debug_group_count`, then assigns that signal to `obs_service_group_count`. M1058 RTL assigns the debug output from `group_accept_count_q`; the large synchronous state block clears this register under `rst_core`, clears it again on `header_accept`, and increments it on `group_accept`.

The mapped netlist retains all 32 bits as physical `DFQD1/DFQD2` flops with only D, CP, and Q pins. Therefore the counter was **not** constant-propagated away. For bit 0, the mapped cone contains:

- `n166977 = ~rst_core`, then `n187524 = ~n166977`;
- `n146196 = ~(n187524 | n110613)`;
- `D0 = ~(Q0 & n146126) & (n146196 | Q0)` through `MAOI22`.

While reset is asserted, `n146196=0`; with an uninitialized `Q0=X`, the factored D expression remains X for either Boolean value of `n146126`. Every reset clock therefore recaptures X instead of the RTL's literal zero. This is Boolean-equivalent synthesis under two-state reasoning but not X-equivalent gate simulation: a synchronous reset mux was factored into a reconvergent old-Q cone.

The cell model confirms `DFQD1/2` are non-resettable flops and their specify path is `(0,0)`. The VCS command has no `UNIT_DELAY`, SDF annotation, or initreg option. UNIT_DELAY and release-to-edge propagation are therefore excluded as the cause.

## Excluded alternatives

- **Observation wrapper counter or counter deletion:** excluded. The wrapper is direct fanout and all 32 mapped counter flops remain.
- **M1058 retained payload reset:** excluded for this first failure. The first eleven control observations are known, raw payload has not yet been presented, and failure is localized to the debug counter output.
- **Memory model:** excluded for this first failure. No raw/memory transaction has begun, and both memory accept observations were already checked as known.
- **128-cycle window:** excluded. Failure occurs at cycle 0 before the intended workload begins. Extending the window or delaying the checker would mask evidence, not repair reset hygiene.

The ten observations after the first `$fatal` remain undetermined. Their mapped flops existing is not proof of their runtime values.

## DC numbers are failed-flow diagnostics only

DC itself returned 0 and emitted its terminal marker. The frozen logic-only, pre-macro reports contain:

| Diagnostic | Value |
|---|---:|
| Cell area | 125,766.647183 µm² |
| Leaf / sequential cells | 155,228 / 31,480 |
| Macros | 0 |
| Setup slack at 3.0 ns | +0.0044 ns, MET |
| Worst hold slack | -0.0190 ns, VIOLATED |
| Hold violating paths | 29,442 |

Because mapped functional observation failed, these values are not paper PPA, not an admitted component row, and not a successful mapped-gate result. The three `HA1D0` too-few-port compile warnings are present but do not intersect the demonstrated first-X counter cone; this audit does not assign them as root cause.

## Findings and only allowed repair

**P0 — mapped observation reset hygiene fails.** M1091r3 cannot admit mapped functionality, activity, power, or paper evidence. The old attempt must remain frozen.

**P1 — diagnostic label drift.** The r3 engine's exception printer says `M1091r2 failure`. The attempt/failure JSON, paths, status checks, and control flow use r3, so this is a stderr-label defect only; it does not alter receipt identity or branch behavior. Fix it only in an additive successor.

The unique minimal repair is a **new namespace** that separates accounting from the functional state block:

1. move the affected service and adapter debug accounting registers into observation-only shadow-counter blocks with explicit asynchronous `posedge rst_core` reset, using read-only taps of the exact existing handshake/accounting events;
2. ensure those taps and counters have no fan-in from observation outputs and no fan-out into any functional ready/valid, payload, accumulator, or memory path;
3. retain already-known live-state observations as direct fanout unless a fresh all-signal census proves another X;
4. strengthen the successor checker to sample and report a 22-bit unknown bitmap atomically before failing, so one first-X does not hide the ten later observations;
5. statically require resettable mapped cells (or an equivalently explicit reset pin) for every shadow-counter bit. Do not use initreg, delayed checking, a longer warm-up, false paths, or `set_case_analysis` to hide X.

Only after a different-author source hammer verifies the isolation and reset-cell contract may one fresh DC/mapped-VCS attempt be authorized. No repair is implemented by this audit.

## Claim boundary

Legal now: the sealed M1091r3 attempt failed closed at the first mapped unknown, and the first-X cone is localized to synchronous-reset debug accounting.

Illegal now: retrying M1091r3; citing its area/timing as admitted PPA; claiming mapped functionality, activity, power, system speedup, Table-A admission, or paper readiness; classifying the ten unexecuted observations as clean.
