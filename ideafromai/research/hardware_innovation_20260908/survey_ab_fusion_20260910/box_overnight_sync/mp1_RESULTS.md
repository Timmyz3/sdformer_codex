# Results — MP1 same_port_credit_probe

## Honesty
Rough RTL microprobe only. **NOT** Stage B `service%`, **NOT** TCAS-II PPA, **NOT** production QoR.
Isolated under `/workspace/rtl_microprobes/`. Does not touch ismd / nts07 / Stage B trees.

## Simulation (iverilog 12.0) — `make sim`
**TB PASS: same_port_credit_probe (C0/C1/C2)**

Identical req/grant traces driven into three MODE instances:

| Mode | Description | serviced_a | serviced_b | stall_count |
|------|-------------|------------|------------|-------------|
| C0 | fixed priority A then B | 56 | 8 | 8 |
| C1 | round-robin | 32 | 32 | 8 |
| C2 | credit backpressure + RR (INIT=4, MAX=8) | 19 | 24 | 29 |

- Mutex: never both `serviced_a` and `serviced_b` in the same cycle (asserted every cycle).
- C0 strongly prefers A under dual demand; C1 balanced; C2 stalls more under credit starvation (`stall_count` 29 ≥ C1's 8).

## Synthesis (yosys 0.52) — `make synth`
Flow: `read_verilog; chparam MODE=2; hierarchy -check; proc; opt; techmap; opt; stat`

Top: `same_port_credit_probe` (MODE=2 so credit path is live)

| Metric | Value |
|--------|-------|
| Cells (total) | **179** |
| Flip-flops | 39 (`$_DFFE_PN0P_`×36 + `$_DFFE_PN1P_`×1 + `$_DFF_PN0_`×2) |
| AND / OR / NOT / XOR / MUX | 64 / 11 / 8 / 37 / 20 |
| Latch inferred | **No** (`No latch inferred` for comb `do_a`/`do_b`) |

No stdcell library / no timing — area proxy = generic cell count only.
