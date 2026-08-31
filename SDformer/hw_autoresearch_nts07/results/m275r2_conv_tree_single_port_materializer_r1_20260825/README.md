# M275r2 conservative Conv-tree materializer

Status: `PASS_CONSERVATIVE_SINGLE_PORT_SINGLE_ACCUMULATOR_ZERO_EXPOSED`.

This is the author-side repair for the M279 port-contract rejection of M275. It
reconstructs all 40 frozen M248 payloads and 17,280 M251 phases, binds the M251r2
signed-INT8 correction `[-2048, 2032]`, rebuilds every M267 Prim tree, and then
executes a conservative event schedule with one synchronous read-or-write PWP
port and one 96-lane signed12 accumulator on the next bank.

## Corrected result

| Item | M275r2 result |
|---|---:|
| Stored fixed-PWP complete modeled Conv cycles | 352,335,120 |
| Tree materialized complete modeled Conv cycles | 352,332,590 |
| Reduction | 2,530 cycles |
| Tree-only ratio vs stored PWP | 1.0000071807x |
| Complete modeled PWP Conv vs bit-sparse | 1.5405684725x |
| Complete modeled PWP Conv vs dense | 18.8332234608x |
| Cold preparation / maximum preparation | 707 / 827 cycles |
| Minimum transition slack / exposed cycles | 333 / 0 cycles |
| Nonresident parent edges per catalog | 5,041 |
| Minimum modeled capacity | 61,776 bytes |

The 1.54x and 18.83x ratios belong to the complete modeled PWP Conv schedule;
the tree materializer is not their source. Its isolated benefit is only
1.0000071807x versus stored fixed PWP. No system speedup or headline is admitted.

## Explicit event ledger

The result counts current 96-byte weight reads, current 144-byte selected-child
PWP reads, next-bank 32-byte weight/metadata fills, next 96-byte generator weight
reads, next 144-byte parent PWP reads, next 144-byte child PWP writes, and 4-byte
descriptor reads. For each catalog phase, the analyzer constructs port occupancy
from offset zero through `387 + 8 * (flips + nonresident parents)` and rejects:

- fill/read overlap on the next weight port;
- read/write overlap on the next PWP port;
- parent entry use before its child-producing write;
- descriptor or weight use before valid;
- role switch before weight, metadata and PWP valid or before generator idle.

All 17,270 in-sample role switches, including 30 operator boundaries, fit in the
already charged two-cycle phase tail. Ten sample starts remain cold, and ten final
drains do not create a nonexistent next preparation or switch.

## Capacity and evidence boundary

The 61,776-byte lower bound contains two 12,288-byte weight banks, two 18,432-byte
PWP banks, two 96-byte metadata banks, and one 144-byte signed12 accumulator. It
does not include control flops, queues, macro overhead, timing or energy. On-chip
PWP capacity is not eliminated.

There is no RTL, VCS, SRAM macro, DC, PT or energy evidence in this milestone.
The result is an exact phase/event-ledger replay for the isolated modeled Conv
module, not complete Conv RTL and not system performance.

## Reproduction

```bash
python3 system_simulator/scripts/analyze_m275r2_conv_tree_single_port_materializer.py \
  --contract contracts/m275r2_conv_tree_single_port_materializer_contract_r1_20260825.json \
  --output-dir results/m275r2_conv_tree_single_port_materializer_r1_20260825
```

The analyzer refuses to overwrite an existing output directory. Reproduce into a
fresh directory or move the sealed result aside first.

