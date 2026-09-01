# M1667 — M1665 C1 canonical recovery result hammer

Verdict: **PASS as a recovered canonical DC candidate; still not a paper PPA
result.** The M1665 recovery is byte-preserving and copy-only. Formality,
independent PrimeTime and power remain mandatory.

## What was checked

- The source quarantine has exactly 39 manifest members. Those 39 files plus
  its `SHA256SUMS` and outer seal are byte-for-byte identical under
  `M1665/original_quarantine`.
- M1665 seals exactly 42 target-manifest members. Its target manifest and outer
  seal pass; the complete 46-file physical topology is regular and contains no
  symlinks. The two additional nested files are the preserved source manifest
  and source outer seal, which also verify.
- M1649/M1650/M1651/M1655/M1659/M1660/M1664 identities and all applicable
  file/review double seals are exact. The recovery attempt is one-shot,
  copy-only and sealed.
- CPython 3.6.8 and 3.10.16 independently return the same semantic result.
  Fourteen in-memory mutation classes per interpreter were exercised; all
  28/28 attacks were rejected without touching evidence.

## DC and physical facts re-derived from sealed bytes

- `dc.rc=0`; the only Error/Fatal is line 32, before `Current time`, caused by
  the absent `HOME` variable while Design Vision sourced its optional
  `dv.tcl`. There are zero Error/Fatal lines after flow start and the log has a
  normal shutdown.
- Setup WNS/TNS: `+0.002221110 ns / 0`; hold WNS/TNS:
  `+0.000999451 ns / 0`; zero violating paths.
- Cell area: `152,898.625984 µm²`, or `+3.838623%` over the sealed
  `147,246.392090 µm²` baseline.
- Nine `TS1N28HPCPHVTB128X128M4S` macros, zero DRC violating nets, and exact
  non-empty DDC/SVF/SDC/mapped-Verilog artifacts.

## Why this is not an EDA rerun

The reviewed M1659 recovery source has no executable EDA command; M1664 grants
zero EDA runs; the sealed attempt says one copy-only recovery with no retry;
the implementation bytes and their nested seals are identical; and both the
provenance and terminal receipt say `dc_rerun=false`. Thus M1665 is a canonical
copy of the completed M1649 DC artifacts, not a second synthesis run. This
statement is scoped to the recovery action and says nothing about unrelated
host processes.

## Remaining boundary

This review admits the DC setup/hold/area/macro/DRC candidate for the next
verification stage only. Direct/transitive RTL plus gate-to-gate Formality,
independent max/min PrimeTime and power are still absent. Consequently
`paper_citable`, `paper_ppa_ready`, power, energy, cycle speedup, system
speedup and headline claims remain false.
