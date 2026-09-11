# Results — MP2 group_accept_continue_probe

## Honesty
Rough RTL microprobe only. **NOT** Stage B `service%`, **NOT** TCAS-II PPA, **NOT** production QoR.
Isolated under `/workspace/rtl_microprobes/`. Does not touch ismd / nts07 / Stage B trees.

## Simulation (iverilog 12.0) — `make sim`
**TB PASS: group_accept_continue_probe (B0/B1/B2)** (WIDTH=8)

8 predicted events (4× `pred_accept=1` with recompute delay 3; 4× `pred_accept=0` with delay 5), identical inputs to all modes:

| Mode | Description | retire_accept | retire_recompute | cyc_accept | cyc_recompute |
|------|-------------|---------------|------------------|------------|---------------|
| B0 | never accept pred | 0 | 8 | 0 | 48 |
| B1 | always accept | 8 | 0 | 8 | 0 |
| B2 | `pred_accept` controls | 4 | 4 | 4 | 28 |

Key cycle counts:
- Accept path: **1 cycle/retire** (B1: 8/8; B2: 4/4).
- Recompute path mean: B0 **6.0** cyc/retire (48/8); B2 **7.0** (28/4) — longer delays on reject batch.
- Mutex: `use_pred` ∧ `issue_recompute` never both 1 (asserted every cycle).

## Synthesis (yosys 0.52) — `make synth`
Flow: `read_verilog; chparam MODE=2 WIDTH=…; hierarchy -check; proc; opt; techmap; opt; stat`

| WIDTH | Cells | DFFE | DFF | AND | OR | XOR | MUX | Latch |
|------:|------:|-----:|----:|----:|---:|----:|----:|-------|
| 8 | **921** | 136 | 36 | 420 | 74 | 226 | 20 | No |
| 16 | **985** | 144 | 36 | 436 | 90 | 234 | 36 | No |

Δ(W16−W8) ≈ **+64 cells** (gate_acc / true_gate_bus width retained as output).

No stdcell library / no timing — area proxy = generic cell count only.
