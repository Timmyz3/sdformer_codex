# M1471 independent result hammer of the M1470 C3 hold-closed candidate

Verdict: **FAIL (one P0 numeric mismatch; M1470 is not admitted).**

The evidence itself is mostly sound. I recursively reverified the M1454 DC, M1456 PrimeTime, and M1457 Formality manifests and outer seals, recomputed the TCL/netlist/SDC/DDC hashes, and independently reparsed the raw reports. PrimeTime reports setup WNS `+0.000299 ns`, fast-min hold WNS `+0.030474 ns`, zero setup/hold violations, and coverage `127700 = 64394 met + 0 violated + 63306 untested`; the untested reasons are exactly `63042 constant_disabled + 264 no_paths`. Formality is gate-to-gate only and reports `11180` passing points with zero failing, unmatched, unverified, or aborted points.

The blocking mismatch is in the M1454 DC hold number. The sealed `timing_hold_fast_dc.rpt` contains 100 paths; its first and minimum slack is `+0.034585 ns`. M1470 records `+0.044297 ns`, which appears on later paths but is not WNS. The setup WNS `+0.000300 ns`, cell area `63756.125879 um^2`, zero macros, and single incremental compile reparse correctly.

The comparison arithmetic also reparses: M917 area is `62433.503388 um^2`, so the new logic area overhead is `2.118449901458219%` (M1470's rounded `2.1184%` is correct). The sealed M1288 failure quarantine reports old hold WNS `-0.022628 ns`, yielding `+0.053102 ns` improvement to the new independent PT result. That old result remains a `FAILED_OR_INCOMPLETE_DO_NOT_CITE` diagnostic and cannot independently become a paper result.

No EDA rerun is needed to repair this receipt. The required next step is an additive corrected candidate using DC hold WNS `+0.034585 ns`, followed by a fresh different-author result hammer. All boundaries remain strict: prelayout, logic-only, ideal clock, ZeroWireload, no SPEF, zero macros, no physical interconnect, no power/energy/speedup/system/headline claim, and Formality is not a direct RTL-to-M1454 proof.
