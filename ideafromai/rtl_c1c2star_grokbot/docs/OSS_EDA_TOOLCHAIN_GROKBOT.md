# OSS EDA toolchain (Grok Bot) — short map

| Synopsys | OSS stand-in | Box status |
|---|---|---|
| VCS | iverilog + vvp (Verilator optional) | ✓ iverilog 12 |
| DVE/Verdi | gtkwave (VCD) | ✓ |
| Design Compiler | yosys (+ abc pass) | ✓ Yosys 0.52 |
| PrimeTime | OpenSTA (needs `.lib`) | **missing liberty** |
| Power | activity×gate heuristic (`flows/oss/power_proxy.py`) | proxy only |
| ICC2 | OpenROAD | **not installed** |

Run: `flows/oss/run_all.sh` → `out/SUMMARY_TCASII_OSS.md`.  
OP-STW synth uses flattened wrapper (Yosys cannot parse unpacked array ports).
