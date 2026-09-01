# M1675 independent M1674 source hammer

Verdict: **PASS, 97/100, P0=0, P1=0, P2=2.** This review authorizes only the authoring of one future exact-SHA M1676 release. It does not authorize EDA now and makes no Formality, PrimeTime, power, speedup or paper-ready claim.

The proof construction is sound at source level: frozen RTL is first compared with the original admitted M993 netlist using only M993's original SVF. A different `fm_shell` process then compares M993 gates with M1665 gates without an SVF. Therefore M1665's incremental SVF is identity evidence only and is not misrepresented as a direct RTL proof.

The future attempt is exact-closed and sequential: two Formality processes, then one independent PrimeTime process. Caller-pinned runner/release hashes, M1675/M1676 authority, exact predecessor/source/tool/library identities, same-UID EDA exclusion, memory/commit/disk resources and licenses all precede attempt creation. Attempt consumption precedes the first EDA invocation. Failures after work creation are sealed as `FAILED_OR_INCOMPLETE_DO_NOT_CITE`; retry is forbidden.

PrimeTime reads the exact M1665 mapped Verilog and SDC in a fresh process. The frozen point is 28 nm, 3.000 ns, setup/hold uncertainty 0.200/0.050 ns, nine SRAM macros, slow/max plus fast/min libraries, ideal clock, ZeroWireload and no SPEF. No false path, multicycle, min/max-delay exception, disabled arc, case analysis, ECO or power action is added.

Mechanical evidence: source unit tests pass 7/7 under CPython 3.6.8 and 3.12.13; the independent hammer agrees semantically under both; 33/33 authority mutations are rejected; 17 exact identities, six directory double seals and the contract double seal pass. M1676, the M1674 attempt and the M1674 result are absent.

Two postrun obligations remain. The result hammer must reject unexpected Formality black boxes while permitting only the expected SRAM abstraction, and must independently reparse PT coverage, global timing, constraints and actual max/min paths. Until that different-author result review passes, `paper_citable=false`.
