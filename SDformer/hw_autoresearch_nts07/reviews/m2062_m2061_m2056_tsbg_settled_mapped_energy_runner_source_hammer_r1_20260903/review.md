# M2062 independent source hammer of the M2061 mapped-energy successor

Date: 2026-09-03 (Asia/Shanghai)

## Verdict

**PASS, 100/100 at source scope; P0=0, P1=0, P2=4.** Exactly one new-identity
M2061 execution is authorized. This is not an M2058 retry. The execution budget
is one `lmstat` preflight, two VCS compiles, two simulations, two SAIF files,
and two PTPX runs, all P1-serial, with zero automatic retry and a mandatory
independent result hammer before any numeric claim.

No EDA executable, simulator, compiler, GPU process, or license query was run
by this review.

## M2058 boundary

The sealed M2058 attempt token still says `M2058_ATTEMPT_CONSUMED_NO_RETRY`.
Its independent failure review and attempt token both verify, no canonical
M2058 success result exists, and the failure review explicitly forbids retry.
M2061 uses fresh attempt, success, failure, private-build, work, and stage
namespaces and reuses no M2058 simulation, SAIF, or PTPX output.

The M2058 failure was caused by an ambiguous grouped X/Z check before its first
SAIF stop, followed by a failure-publication problem because VCS build symlinks
were passed to a seal that rejected symlinks. M2061 addresses both mechanisms:

- qualifiers, faults, busy, and counters are checked unconditionally;
- load/request/response/bridge/commit payloads are checked only under their
  owning valid, with bridge bank payload additionally gated by bank-valid;
- every check uses a named, per-signal `require_known` diagnostic; and
- failure quarantine copies only regular evidentiary files while separately
  fingerprinting every regular file by path/SHA/bytes and every symlink by
  path/target before sealing and no-replace publication.

A synthetic regular-file-plus-symlink tree confirmed that the symlink remains
in the complete fingerprint but is absent from copied quarantine evidence.

## Frozen workload and timing window

Both axes preserve the pre-registered ep34 global slot42 identity: M2047 anchor
slot0, sample0, layer28 FC1, token0, G48, 149 rows, 1278 issues, 29472 products,
and 24 commits. The 383 descriptor-load cycles remain outside the SAIF window.

The first stop is at the first execute negedge plus 10 ps. The second stop is
at the selected completion negedge plus 10 ps. Thus both endpoints have the
same phase and delta margin, and elapsed activity time must be exactly:

- ordinary LRU4: `20292 * 3 ns = 60876 ns`;
- TSBG-B4: `7569 * 3 ns = 22707 ns`.

The parser and PTPX Tcl independently reject any other duration. After the
second stop, a third UCLI `run` must reach exactly one full-field
`PASS_M2051_EP34_TSBG_FULL40_CYCLE`; the parser now additionally enforces the
order `BEGIN < END < final PASS`.

## Netlist, SDC, SAIF, and PTPX identity

Each filelist contains exactly one original M2029 mapped netlist: schedule mode
0 for ordinary LRU4 and mode 1 for TSBG-B4. The associated original M2029 SDC
is exact-hash pinned. The PTPX script derives design, SAIF strip scope, and
cycle denominator solely from the two-value axis selector. It loads SSG and TT
libraries before the SDC, then explicitly selects TT 0.9 V/25 C and
ZeroWireload. It requires zero black boxes, zero memory macros, exact 100%
net/leaf annotation, successful `check_power`, four unique nonnegative power
fields, positive total power, and a matching subtotal.

This remains zero-delay functional mapped activity with no SDF, ideal clock,
no CTS, no macro, and external weight SRAM excluded. It is not a gate-delay
repair or paper-ready total energy result.

## Parser hammer

The frozen parser passes its static source check with 33 dependencies. A
42-check pure-Python hammer passed positive cases for both axes and rejected:

- swapped stop markers, duplicate final PASS, and an altered M2051 ledger;
- an injected mapped X/Z diagnostic;
- incorrect SAIF duration and nonzero SAIF `TX`;
- a wrong compile top;
- a wrong PTPX strip scope; and
- a power subtotal mismatch.

The hammer also verifies source syntax, exact two-stop structure, settled
negedge checks, valid-gated sidebands, unconditional control/counter checks,
SSG+TT-before-SDC ordering, exact annotation gates, new namespaces, zero retry,
and symlink-aware failure quarantine.

## Source fixes made before freezing

Three source-only hardenings were applied and are included in the final parser
hash: the empty `SOURCE_SHA256` inventory was frozen; contract rows now require
exact schema/cardinality/no duplicate path; and runtime log parsing now
requires begin/end/final-PASS order. These change neither workload nor RTL.

The eight-file M2061 source set, the 33-dependency inventory, the source
contract, and this review are exact-hash sealed. The caller must use the exact
authority pins supplied with the handoff; any source or review drift fails
before attempt creation.

## Claim boundaries

1. Source review only; no M2061 functional, power, or energy result yet.
2. Single pre-registered component workload, not the 1,920-workload M2057
   distribution and not full FC/network/system energy.
3. Real ep34 activity masks but deterministic directed INT8 weights.
4. Standard-cell averaged prelayout power only; external weight SRAM remains
   symbolic and clock tree/macro energy is excluded.

## Authorization

Exactly one invocation of the exact runner/parser/contract/review identities is
authorized. The invocation consumes the M2061 attempt before its single license
preflight and cannot retry. A successful candidate remains non-citable until a
separate independent result hammer verifies its double seal, log ordering,
SAIF duration/activity, annotation, PTPX reports, and claim boundary.
