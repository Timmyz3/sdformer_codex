# M1736 author source handoff

M1736 is a zero-EDA, read-only canonicalization of two immutable predecessor artifacts: the sealed M1722 Formality proof and the completed M1733 PrimeTime report set. It does not rerun Formality, PrimeTime, DC, a license query, or a network call.

The admitted Formality evidence reports `Verification SUCCEEDED`, `verify_return=1`, 16,549 passing compare points, zero failing/aborted/unverified/unmatched points, and nine SRAM macros on each side. The admitted PrimeTime reports show setup WNS/TNS `+0.027871 ns / 0`, hold WNS/TNS `+0.001827 ns / 0`, and zero setup or hold violating paths at 3 ns with setup/hold uncertainty `0.2/0.05 ns` and nine macros.

The M1733 campaign remains sealed `FAILED_OR_INCOMPLETE_DO_NOT_CITE_AS_CAMPAIGN`. Its scanner rejected exactly two startup-environment diagnostics emitted before the main Tcl: `PT-063` for an unset Library Compiler executable and a missing `::env(HOME)` startup-script error, with associated `CMD-013`/`CMD-081` messages. The frozen raw log then enters the exact main Tcl, writes the marker and all reports, executes `quit`, prints `Diagnostics summary: 2 errors, 5 warnings, 30 informationals`, and reaches the normal shell epilogue. M1736 allows no other line-start `Error:` diagnostic and reports zero unaccounted errors.

Timing coverage is disclosed rather than called complete: setup and hold each contain 13,860 endpoints, 13,851 met, zero violated and nine untested; minimum pulse width contains 78,506 endpoints, 50,526 met, zero violated and 27,980 untested/no-clock. The result remains prelayout, ideal-clock, `ZeroWireload`, without parasitics; `paper_ppa_ready=false`.

Python 3.6 and 3.12 each pass 12/12 source tests. A different-author M1737 mutation-heavy review and a separately sealed M1738 release remain mandatory before the one allowed zero-EDA canonicalization. No attempt/result/release was created while authoring.
