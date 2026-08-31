# M135r3 independent hammer review r1

Outcome: **84/100, P0=0, P1=2, P2=4**. GO for the narrow M135r2 RTL functional seal and M135r3 flattened logic-only DC result; NO-GO for macro-complete PPA or any physical/kernel/system speedup claim.

## Independently reproduced

- Exact frozen sources were recompiled with Synopsys VCS V-2023.12-SP1. The production PASS reproduced exactly: 96 vectors, 96 outputs, 212 accepted beats, 8,832 numeric lane checks and start intervals 2/2/2/3.
- The four production SVA covers reproduced at 198/1/1/8 matches, with zero assertion failure.
- All 11 sealed VCS inputs, four sealed VCS outputs and all 20 M135r3 DC evidence entries pass SHA verification.
- M135r3 DC is internally consistent: area 12,732.048032 um2, 16,194 cells, 2,768 sequential cells, setup +0.5341 ns, reported hold +0.0000 ns, zero macros, and clean postcompile check_design/check_timing.
- The r1 and r2 DC runs really failed closed at exit 35 with 19 and 17 LINT-28 warnings. Their area/timing numbers remain diagnostic and noncitable.

## Hammer finding

Four independently directed, metadata-decidable protocol errors are rejected at the traffic interface but still expose a full bank request in the same cycle:

`protocol_error=1, beat_ready=0, beat_accept=0, output_valid=0, bank_use_mask=ffff, bank_conflict_free=1, bank_row_addresses!=0`.

The cases are idle continuation, unsupported width, premature last and restart while collecting. Malformed escape and illegal base controls correctly keep the bank outputs quiet. The fix is to place a state-aware metadata legality precheck before the mapper/read-enable boundary and add bank-side quarantine assertions.

## Performance boundary

The 2/2/2/3 result is an accepted-beat service interval under an always-ready, zero-latency 512-bit bank-data port. It is not request-to-output macro latency and not a speedup. Sixteen SRAM macros, response valid/tag/latency, stale/skew detection, bank wiring, routed clocks, power and a matched baseline are absent.

Flattening removes 448.308002 um2 (3.401335%) and 817 cells (4.802775%) relative to the failed same-RTL hierarchical r2 diagnostic, with unchanged sequential cells. That is useful synthesis-debug evidence only; it is not a citable architectural gain because r2 failed postcompile check_design.

## Next actions

1. Gate metadata-decidable invalid beats before bank request generation and rerun the independent attacks.
2. Add a minimal tagged SRAM request/response latency shim and behavioral macro model; no global scheduler is needed.
3. Seal RTL-to-flattened-netlist Formality.
4. Run macro-inclusive, matched-baseline cycles/PPA/power before introducing any acceleration ratio.
5. Add a supersession overlay pinning the r2 failed marker and report hashes.

Detailed machine-readable findings are in `m135r3_independent_hammer_review_r1.json`; exact evidence reconciliation is in `m135r3_independent_evidence_audit_r1.json`.
