# TOOLCHAIN — BOX overnight 2026-09-11

Recorded on the shared Linux box under `/workspace`.

| Tool | Status | Version |
|------|--------|---------|
| Icarus Verilog (`iverilog`) | installed | 12.0 (stable) |
| Yosys | installed | 0.52 (git sha1 fee39a3284c90249e1d9684cf6944ffbbcbb8f90) |
| nextpnr-generic | installed (optional) | 0.7-1+b2 |

## Install notes
- Packages via `apt`: `iverilog`, `yosys`, `nextpnr-generic`.
- nextpnr-ice40 not required for this overnight; generic nextpnr pulled as the easy optional.

## Purpose
- Support **separate** fusion RTL rough probes under `/workspace/overnight_20260911/rtl_probes/`.
- **Not** part of Codex Stage B; do not divert Stage B into full-chip RTL.

## Honesty
These tools enable sim + coarse synth cell/area reports only. Not a tapeout flow; not TCAS-II PPA.
