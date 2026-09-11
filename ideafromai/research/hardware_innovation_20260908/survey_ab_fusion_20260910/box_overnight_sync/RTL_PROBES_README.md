# rtl_probes — fusion RTL rough suitability (BOX only)

**Isolated scaffold.** Do **not** touch sdformer / ismd / nts07 production trees.
**Separate from Codex Stage B** (schedule_compare_same_port). Stage B must not be diverted here.

## Goal
Tiny Verilog micro-probes inspired by shared-port credit / group accept-continue / lifting half-step RNE.
Use `iverilog` for TB and `yosys` for coarse synth area/cells.

## Honesty
This is **rough suitability** only — not TCAS-II PPA, not production QoR, not a chip claim.
**NOT** Stage B service%.

## Layout
```
rtl_probes/
  README.md
  OVERNIGHT_RTL_SUMMARY.md   # -> /workspace/rtl_microprobes/OVERNIGHT_RTL_SUMMARY.md
  lifting_halfstep_rne/      # half-step RNE checkpoint
  mp1_same_port_credit/      # -> microprobes MP1
  mp2_group_accept/          # -> microprobes MP2
```

Canonical sources live in `/workspace/rtl_microprobes/`.

## Tools
See `../TOOLCHAIN.md` (iverilog 12.0, yosys 0.52, nextpnr-generic 0.7).

## Status (2026-09-11)
- `lifting_halfstep_rne/` — TB PASS; ~76 cells
- `mp1_same_port_credit/` — TB PASS; synth 179 cells, no latch
- `mp2_group_accept/` — TB PASS; synth 921 (W8) / 985 (W16) cells, no latch
