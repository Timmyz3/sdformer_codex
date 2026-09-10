# OSS EDA flow (Synopsys replacement map) — Grok Bot / TCAS-II

**Tree:** `/workspace/sdformer_c1c2star_grokbot/` only.  
**Do not** touch `hw_autoresearch_nts07`. Local git only; no push.

## Toolchain map

| Synopsys / commercial | OSS stand-in | Status on this box |
|---|---|---|
| VCS | **iverilog + vvp** (primary); Verilator optional | iverilog 12.0 ✓ |
| DVE / Verdi | **gtkwave** (VCD/FST) | gtkwave ✓ |
| Design Compiler | **yosys** (+ internal **abc** pass) | Yosys 0.52 ✓ |
| PrimeTime (timing) | *none without liberty / OpenSTA* | **gap** — document only |
| PrimePower / power | **activity × gate heuristic** (`power_proxy.py`) until `.lib` | proxy only, **NOT silicon** |
| ICC2 / P&R | OpenROAD (not installed here) | **gap** |

## Scripts

| Script | Purpose |
|---|---|
| `sim_op_stw.sh` | Card A OP-STW: iverilog → vvp → `out/waves/c1s_op_stw.vcd` |
| `sim_hbg_rp.sh` | Card B HBG-RP: same → `out/waves/c2s_hbg_rp.vcd` |
| `sim_motion_ttb.sh` | Motion-TTB packer → `out/waves/c2s_motion_ttb.vcd` |
| `sim_sth_gate.sh` | STH-Gate → `out/waves/c2s_sth_gate.vcd` |
| `synth_yosys.ys` | Yosys recipe (generic techmap + abc if available) |
| `run_synth.sh` | Synth A/B/ECP/MW/SMAM/Motion-TTB/STH-Gate → `out/synth/*.stat.txt` |
| `power_proxy.py` | VCD toggle × cell count → rough dynamic energy **proxy** |
| `run_all.sh` | Sim + synth all cards → `out/SUMMARY_TCASII_OSS.md` |

## Yosys note (OP-STW)

`c1s_op_stw_predictor.sv` uses **unpacked array ports**, which Yosys 0.52 cannot parse.
Synth uses the functionally equivalent flattened wrapper:

`flows/oss/wrappers/c1s_op_stw_synth.sv` (`c1s_op_stw_predictor_synth`).

Functional self-check TB still targets the unpacked RTL.

## Outputs

```
out/
  waves/     *.vcd
  sim/       *.log
  synth/     *.stat.txt, *.ys.log
  SUMMARY_TCASII_OSS.md
```

## Limitations (first letter)

- No liberty → no STA / no silicon power / no area in µm² — only generic cell counts.
- No OpenROAD → no floorplan / CTS / route.
- Power numbers from `power_proxy.py` are **heuristic proxies**, labeled as such in SUMMARY.

## Yosys note (Motion-TTB)

`c2s_motion_ttb_packer.sv` uses unpacked array ports; synth uses
`flows/oss/wrappers/c2s_motion_ttb_synth.sv` (`c2s_motion_ttb_packer_synth`).


## Yosys note (STH-Gate)

`c2s_sth_gate.sv` uses unpacked array ports; synth uses
`flows/oss/wrappers/c2s_sth_gate_synth.sv` (`c2s_sth_gate_synth`).

## Yosys note (OGEC)

`c1s_ogec_gate.sv` ports are packed; synth uses
`flows/oss/wrappers/c1s_ogec_synth.sv` (`c1s_ogec_gate_synth`, N_TILE=8)
which instantiates the RTL.

## Yosys note (front_pipe)

Hierarchical RTL uses unpacked array ports; synth uses
`flows/oss/wrappers/c1s_front_pipe_synth.sv` @ N_TILE=8 instantiating
OP-STW / ECP / MW flat wrappers + `tile_active` glue.
