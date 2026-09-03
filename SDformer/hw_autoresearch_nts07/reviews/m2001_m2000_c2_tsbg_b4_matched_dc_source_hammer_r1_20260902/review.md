# M2001 independent source hammer

## Verdict

PASS (100/100; P0/P1/P2 = 0/0/0). The exact M2000 runner is authorized for exactly two logic-only DC runs: `ordinary_lru4` with `SCHEDULE_MODE=0` and `tsbg_b4` with `SCHEDULE_MODE=1`. No other EDA run is authorized.

## What was checked

- Exact runner, two-entry filelist, M1995 RTL, M803 adapter, Tcl, SDC, M1999, M1866 and docs/359 identities.
- M1999, M1866, M1995 failure review and M1997 source review double seals.
- The M1992 reserved-keyword parse failure is isolated; the admitted successor is the separately identified M1995 source and its M1999 directed-VCS chain.
- Both DC axes share the same top, public ports, sources, Tcl, SDC, slow/min libraries and synthesis body. Their only elaboration difference is `SCHEDULE_MODE=0/1`; `SOURCE_GROUPS=48`, `BUNDLE=4` and `CACHE_ROWS=4` remain production defaults.
- Six-hour per-axis timeout, one consumed attempt, no retry, failure quarantine, seal-before-publish, exact bootstrap whitelist, TIM-209/OPT-150, design-rule artifacts and minimum-slack parsing.
- 28 independent mutations were attempted and 28 were rejected.

## Claim boundary

This authorization can produce a matched, logic-only, pre-macro physical schedule ablation. It cannot establish exact RTL cycle speedup, same area, hold closure, power, energy, production-G48 dynamic verification, conventional-baseline PPA, cross-layer cache rebinding, paper-ready PPA or system speedup. Any raw M2000 result remains pending a fresh independent result hammer.

No EDA or license query was launched by this review. docs/359 was not modified.
