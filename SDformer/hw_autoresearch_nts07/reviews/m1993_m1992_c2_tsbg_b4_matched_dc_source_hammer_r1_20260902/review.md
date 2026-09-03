# M1993 independent source hammer for M1992 matched TSBG DC

## Verdict

**PASS, 99/100, P0/P1/P2 = 0/0/0.**  This review authorizes exactly two
`dc_shell` invocations by the exact runner SHA
`4d7db586133997c0722a941c42c4d84a37b03eda1e688c035f814565fec3ad5f`:
one `SCHEDULE_MODE=0` and one `SCHEDULE_MODE=1`.  It authorizes no other EDA
action.  The earlier runner identities `0ab65e...e6120`,
`521d9b...18af`, and `302bac...b707` are superseded and are not
authorized.

The two elaborations use the same M1880 top, public ports, exact two-file RTL
list, M519-R8 Tcl, 3 ns SDC, slow/min libraries, compile body, and production
default `BUNDLE=4, SOURCE_GROUPS=48`.  The only elaboration parameter supplied
by the runner is `SCHEDULE_MODE=0/1`.  In the reviewed RTL the schedule
parameter affects only the token-major versus group-major scan order; both
axes retain the same B4 descriptors, LRU4 cache, four Acc24 contexts, M803
adapter, and commit path.

## Fail-closed and parser checks

- The exact M1990 directed VCS result and M1866 CPU premodel review are SHA
  pinned and double-seal checked.  M1866's review SHA is copied into the raw DC
  receipt, so its 2.5338x CPU opportunity cannot be silently rebound to this
  physical ablation.
- The exact runner, filelist, Tcl, SDC, RTL, adapter, upstream reviews, and
  frozen docs/359 identities are checked before attempt consumption.  The
  source review itself is caller-SHA pinned and double-seal checked.
- The attempt is consumed before the license preflight.  Any interrupted or
  failed run is sealed under `FAILED_OR_INCOMPLETE_DO_NOT_CITE`; a successful
  work directory is sealed before atomic publication.
- Each axis has an exact six-hour wall limit, followed by `TERM` and a bounded
  60-second grace period before `KILL`.  Timeout exits enter the same sealed
  failure quarantine and cannot be retried.
- The DC log admits exactly one fixed-hash Synopsys GUI bootstrap error block.
  Any other `Error`, `Fatal`, TIM-209, or OPT-150 event is rejected.  Required
  area/QoR/setup/hold-diagnostic/netlist artifacts and zero max-capacitance,
  max-transition, and max-fanout violations are gated.
- The setup/hold report parser takes the minimum slack among all printed
  paths.  An adversarial report containing `-0.25 ns` followed by `+0.10 ns`
  correctly resolves to `-0.25 ns`; the superseded last-match parser would
  have falsely admitted it.
- Eighteen independent mutations spanning axis equality, G12 substitution,
  top/filelist drift, seal or M1866 identity removal, authorization widening,
  wall-timeout removal/loosening, WNS parsing, area-gate loosening, and
  claim-boundary inflation were all rejected without EDA or license access.

## Claim boundary

This is a logic-only, pre-macro, ideal-clock, ZeroWireload **schedule
ablation**.  Because both axes intentionally own the candidate B4/cache/Acc24
state, it is not conventional-baseline PPA and does not price a smaller
ordinary implementation.  SRAM-like arrays synthesize as standard cells.
Hold is diagnostic only.  The layer-private weight cache has no implemented
cross-layer flush/rebind; weight-domain changes require reset or external
rebind.  G48 is a static production elaboration point, not dynamically
verified by M1990.  No exact RTL cycle ratio, same-area result, power, energy,
system speedup, or paper-ready PPA is authorized.  Any raw result remains
pending a fresh independent result review.

No EDA tool or license query was launched by this review.
