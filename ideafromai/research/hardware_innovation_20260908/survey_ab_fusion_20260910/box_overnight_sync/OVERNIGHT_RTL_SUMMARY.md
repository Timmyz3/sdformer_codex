# Overnight RTL Microprobes Summary — 2026-09-11

**Box tools:** iverilog 12.0, yosys 0.52  
**Root:** `/workspace/rtl_microprobes/`  
**Mirror:** `/workspace/overnight_20260911/rtl_probes/` (summaries linked)

## Honesty (read first)
These are **tiny fusion-inspired RTL suitability probes**. They are:

- **NOT** Codex Stage B `service%` / schedule_compare_same_port results
- **NOT** TCAS-II PPA or any chip-level QoR claim
- **NOT** production RTL; no ismd / nts07 trees were touched

Cell counts are generic `techmap` proxies only (no liberty, no timing, no place-and-route).

## Scorecard

| Probe | `make sim` | `make synth` | Cells | Key cycle / behavior counts |
|-------|------------|--------------|------:|-------------------------------|
| **MP1** `mp1_same_port_credit` | **PASS** | **PASS** (no latch) | **179** (MODE=2) | C0 svc a/b=56/8 stall=8; C1=32/32 stall=8; C2=19/24 stall=29 |
| **MP2** `mp2_group_accept` | **PASS** | **PASS** (no latch) | **921** (W=8) / **985** (W=16) | Accept path 1 cyc; B0 recompute mean 6.0; B2 accept×4 + recompute mean 7.0 |

## MP1 — same_port_credit_probe
- Modes: C0 fixed A>B; C1 RR; C2 credit backpressure (stall when `credit==0`, restore then RR).
- TB: identical traces; mutex never both serviced; C2 stalls ≥ C1.
- Details: `mp1_same_port_credit/RESULTS.md`

## MP2 — group_accept_continue_probe
- Modes: B0 never accept; B1 always; B2 `pred_accept` gated.
- Mutex: `use_pred` / `issue_recompute` exclusive.
- WIDTH=8 and 16 synth both clean; wider bus → +64 cells.
- Details: `mp2_group_accept/RESULTS.md`

## How to re-run
```bash
cd /workspace/rtl_microprobes/mp1_same_port_credit && make sim && make synth
cd /workspace/rtl_microprobes/mp2_group_accept && make sim && make synth
```

## Related (pre-existing, separate)
- `overnight_20260911/rtl_probes/lifting_halfstep_rne/` — half-step RNE checkpoint probe (already PASS; ~76 cells).
