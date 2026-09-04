# M448R3 M431/M438 prelayout standard-cell PTPX

Status: **PASS, pending independent hammer.** Scope is the M416 balanced selected slice only.

- Corner: TT 0.9 V, 25 C; ideal 3.0 ns clock (333.333333 MHz), ZeroWireload, no SPEF, 0 macro.
- Input slew: 100 ps primary on 1,666 nonclock inputs including reset_n; 50/200 ps sensitivity only. All three check_power gates pass with 0 ramp/missing-table/missing-function findings.
- Activity: 64 phases / 192,000 rows / 2,096,003 cycles / 6,288,008.5 ns; 22,800/22,800 exact, 21,827/22,800 nonzero (95.732456%), TX=0.
- Primary power: internal 5.5926948 mW; net switching 0.62686992 mW; leakage 0.03424307 mW; total 6.253808 mW.
- Primary energy per measured cycle: 18.761423 pJ/cycle.
- Sensitivity total: 50 ps 6.253736 mW (0.999988x); 200 ps 6.254251 mW (1.000071x).

reset_n slew is not reset signoff. Clock-network group contains register clock-pin internal power but no CTS. SRAM, macros, extracted interconnect, four-Conv and system energy are excluded. R1/R2 remain failed; this is not paper-PPA or speedup evidence.
