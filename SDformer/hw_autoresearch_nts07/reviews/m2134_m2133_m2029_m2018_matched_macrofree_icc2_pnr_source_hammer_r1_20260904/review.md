# M2134 independent source hammer: M2133 matched macro-free ICC2 P&R

## Verdict

**PASS — exactly one M2135 production attempt is authorized.** Score: **99/100**; P0/P1/P2 = **0/0/0**.

M2133 closes the deterministic producer/consumer failure found by M2130 without weakening SPEF admission. The installed ICC2 V-2023.12-SP3 command reference states that `write_parasitics -output` is a filename prefix, producing `<prefix>.<parasitic_tech>_<temperature>[...].spef`, and that the command separately emits `XX.spef_scenario`. M2133 therefore writes the selected `tt_power` corner to the dedicated prefix `raw_parasitics/routed`. The exact-SHA canonicalizer independently enumerates every direct `.spef` entry, requires exactly one regular, nonempty, nonsymlink file matching `routed.n28_1p9m_6x1z1u_typ_25[.0...].spef`, excludes `.spef_scenario`, rejects a pre-existing canonical target, and uses a same-filesystem `os.replace` to publish literal `output/routed.spef`. The runner requires its PASS token and receipt before the strict pair parser. The parser rejects `.gz`, scenario-only, missing, empty, and symlink substitutions and binds the receipt to TT/25 C.

The M2130 P0 is therefore repaired. No new blocking source defect was found.

## Independent attacks and regression checks

The independent checker completed **107/107** gates. It did not invoke ICC2, `lmutil`/`lmstat`, VCS, DC, PT, Formality, another EDA executable, a license query, or a GPU workload.

- SPEF producer semantics: the installed, hash-pinned ICC2 manpage confirms prefix-plus-tech/temperature naming, single-corner `-corner`, and the separate scenario sidecar.
- Canonicalization mutations rejected: no SPEF, multiple SPEFs, scenario-only, wrong technology, wrong temperature, empty file, symlink, pre-existing `output/routed.spef`, and raw gzip-only.
- Parser mutations rejected: missing canonical SPEF, gzip substitution, scenario substitution, canonical symlink, wrong receipt corner, wrong receipt temperature, wrong raw name, open-net count 999, DRC count 777, direct-unbound count 1, missing design-library report, `VIOLATED -999`, live-query area mismatch, routed DEF pin mismatch, and routing-policy mismatch.
- The supplied M2133 tests independently re-ran **19/19 PASS** (7 canonicalizer plus 12 parser tests). Python compilation, Bash syntax, and JSON parsing passed. No standalone `tclsh` exists on this host, so this review does not claim an independent Tcl interpreter run; it instead inspected the complete frozen Tcl source and its inherited, previously reviewed structure without launching ICC2.

## Inherited M2129 gates

- Immediate post-link direct `is_unbound`, `is_unmapped`, `is_black_box`, and black-box `ref_name` censuses remain before NXTGRD and are separately required to be zero; mismatch collections are not a substitute.
- TT, SS, FF, and physical master coverage each remain 94/94 before the parasitic-tech read; the core-site, M1-M9, and VIA1-VIA8 gates remain present.
- The complete report inventory remains mandatory. Setup/hold facts are cross-checked to anchored six-significant-digit timing reports, and routed area/cell counts are cross-checked to the dedicated live-query report.
- Numeric `check_routes` open-net and DRC summaries remain required and zero. Routed DEF die plus all 4551 placed pin name/layer/location/orientation tuples remain part of the matched physical identity.
- Both axes share the normalized physical SDC, die/core, deterministic pins, M2:M8 routing, CTS/hold whitelists, scenarios, NXTGRD, effort, and external 288-KiB SRAM boundary. Six physical-policy artifacts are byte-compared before parsing.
- ICC2 and the actual `lmutil lmstat` executable remain kind/path/executable/SHA pinned. The Milkyway inventory remains exhaustive at 1051 regular files, including 1044 FRAM and 2 CEL files.
- The one-shot runner requires an exhaustive M2134 double seal, pins runner/Tcl/parser/canonicalizer/contract hashes, checks result/attempt/work/lock absence and same-UID/memory gates, consumes the attempt before its single license query, runs two axes sequentially, permits no retry, quarantines any incomplete work, seals the raw result, and atomically publishes it.

## Non-severity editorial observation

One inherited descriptive phrase in the contract still says `routed.spef or routed.spef.gz`. It is superseded within the same contract by the operative strict regex `^routed[.]spef$`; the M2133 canonicalizer, runner, parser, and independent gzip mutations all enforce the literal uncompressed name. This stale historical wording cannot widen execution or result admission and is not a P0/P1/P2 finding. It should be cleaned in a later documentation-only identity, not by mutating M2133.

## Authorization and claim boundary

M2134 authorizes **only M2135**, with the exact budget:

- `license_queries = 1`
- `icc2_shell_runs = 2` (ordinary LRU4, then TSBG-B4)
- `all_other_eda_runs = 0`
- `automatic_retry = false`

This authorization is source readiness, not a physical result. M2135 must remain raw and non-citable until an independent M2136 result hammer validates both routed axes. Even after result admission, the allowed claim is a matched post-route, macro-free logic-island feasibility comparison with an identical external/common 288-KiB SRAM model. It is not macro-inclusive, SRAM-integrated, whole-accelerator, whole-network, max/min-RC signoff, EMIR/LVS, tapeout, silicon, or paper-ready PPA.

Protected `docs/359` remains exactly `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
