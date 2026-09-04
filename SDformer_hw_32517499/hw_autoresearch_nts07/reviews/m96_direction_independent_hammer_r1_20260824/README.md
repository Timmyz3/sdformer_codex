# M96 direction independent hammer

The unique recommendation is **M96 fixed-group reversible bank-skew weight packing**. It attacks the only large remaining module-local bottleneck—finite-bank source issue—without adding a data port, SRAM capacity, vector cache, or full-system scheduling mechanism.

## Exact M89 K6-C16 ledger

Across the frozen 40 records and 10 samples:

| Component | Cycles | Integrated share | Perfect-elimination Amdahl bound |
|---|---:|---:|---:|
| Finite-bank fused source | 69,964,176 | 91.245% | 11.422×, deliberately unphysical |
| Command/state wait | 2,624,272 | 3.422% | 1.0354× |
| Response/context wait | 2,011,048 | 2.623% | 1.0269× |
| Parent wait | 1,947,448 | 2.540% | 1.0261× |
| Weight-DMA wait | 122,880 | 0.160% | 1.0016× |
| Other calendar residual | 7,496 | 0.010% | negligible |

Integrated cycles are 76,677,320 and source cycles are 69,964,176. The source engine processes 562,451,704 logical updates as 416,232,640 unique weight issues across 10,436,792 K6 groups. Existing fusion therefore already removes 25.997% of repeated issue work.

The loose global eight-bank work-conservation lower bound is `ceil(416,232,640 / 8) = 52,029,080` source cycles. This exposes at most 17,935,096 cycles of bank imbalance plus per-group granularity, corresponding to an optimistic integrated bound of 58,742,224 cycles or 1.3053×. This is an upper bound, not an expected result: M96 must also compute the tighter sum of `ceil(group_union_popcount / 8)`.

## Where the cycles live

The two `conv1` operators are 98.35–98.64% source-bound and together consume 46,726,328 source cycles. The two `conv2` operators contain 832,288 of all 832,696 zero-source groups and 6,001,352 non-source cycles—89.397% of all non-source overhead.

That split suggests two eventual module designs:

- `conv1`: source-bank packing and conflict reduction.
- `conv2`: zero-union alias/materialization or accumulator-port optimization.

Only the first has enough clean, non-system upside to become M96.

## Why the other directions are not M96

- Wider descriptor packing has already been screened. M93 width2 saves 62,880 command-wait cycles but only 27,624 integrated cycles (1.00036×), uses 128-byte packets at 58.37% lane utilization, and regresses two samples.
- A second/parity-banked final-accumulator path has a 2,011,048-cycle absolute ceiling (1.0269×). It is a valid later ablation, but smaller than source banking.
- Parent heuristics and forwarding are scheduler-coupled: M90 regresses 664,160 cycles; M91 misses its gate by 20,270; M92 gains only 79,632 cycles and regresses source by 58,184.
- Critical-first and sparse-first seeds regress by 157,288 and 204,176 cycles. Oldest-order locality matters.
- M55's 1.338× zero/dual source-work ratio is explicitly not a cycle or speedup result.

## M96 frozen probe

Freeze M89 parent choice, DAG, admission order, K6 group identity and union masks. Repack every existing eight-entry weight row with one reversible bank hash:

- H0: `bank = b`
- H1: `bank = b XOR row[2:0]`
- H2: `bank = (b + row[2:0]) mod 8`
- H3: `bank = (b + 3*row[2:0]) mod 8`

Choose one two-bit mode per operator across all ten samples. No per-group or per-sample adaptation is allowed. The weights are permuted offline inside the same rows; total rows, bytes, banks and read ports stay unchanged. Runtime cost is eight configuration bits for four operators plus a 3-bit XOR/add bank decoder; the winning mapping can later be simplified or hardwired.

The existing simulator is sufficient: seal the ordered union masks once, replay `bank_issue_cycles` under H0–H3, and feed only the changed source completion times through the existing calendars.

Promotion requires all of:

- source cycles ≤69,614,355 and integrated cycles ≤76,293,933 (at least 0.5% improvement);
- every sample non-regressing in source and integrated cycles;
- p95 integrated cycles <7,843,680;
- exact group/union identity and weight/DMA byte conservation;
- zero extra SRAM ports, weight capacity, and vector storage.

Until those gates pass, M96 remains a transaction-model layout screen—not RTL, PPA, energy, full-network, or system speedup evidence.

## Reproduction

```bash
python3 hw_autoresearch_nts07/reviews/m96_direction_independent_hammer_r1_20260824/audit_m96_direction.py \
  --hw-root hw_autoresearch_nts07 \
  --output hw_autoresearch_nts07/reviews/m96_direction_independent_hammer_r1_20260824/m96_direction_audit.json \
  --log hw_autoresearch_nts07/reviews/m96_direction_independent_hammer_r1_20260824/m96_direction_audit.log
```
