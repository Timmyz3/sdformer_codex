# Results — lifting_halfstep_rne (overnight 2026-09-11)

## Honesty
Rough suitability probe only. **Not** TCAS-II PPA. Isolated under `rtl_probes/`. Separate from Codex Stage B.

## Simulation (iverilog 12.0)
- Command: `./run_sim.sh`
- Result: **TB PASS: lifting_halfstep_rne RNE checkpoint**
- Note: iverilog emits "constant selects in always_*" warnings on part-selects; functionally PASS.

## Synthesis (yosys 0.52, generic techmap + abc)
Top: `lifting_halfstep_rne` (W_IN=16, W_OUT=8 defaults)

| Metric | Value |
|--------|-------|
| Cells (total) | **76** |
| Flip-flops | 9 (`$_DFFE_PN0P_`×8 + `$_DFF_PN0_`×1) |
| AND / NAND / OR / NOR | 11 / 8 / 16 / 5 |
| XOR / XNOR | 2 / 17 |
| MUX / ANDNOT | 2 / 6 |
| Wires / wire bits | 75 / 112 |

No stdcell library / no timing — area proxy = cell count only.
