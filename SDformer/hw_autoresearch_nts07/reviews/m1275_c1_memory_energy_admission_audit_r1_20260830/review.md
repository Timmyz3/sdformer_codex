# M1275｜C1 memory/energy admission audit

## Verdict

**Split verdict.** Existing evidence supports one citable parent-scratch
component ablation and one separately labelled capacity-equivalent static SRAM
model. It does **not** yet support a candidate-vs-baseline C1 energy ratio, a
105-macro dynamic-energy row, throughput/mm², total C1 energy, or system
energy.

The latest `1.7591725402x` raw-CPU same-ledger result and the M623 parent
component result cover the same frozen ep35 population and have exactly the
same candidate parent access vector. However, M623's total energy uses the
older M528 cycle schedule, and its `38.2283%` comparison is against the M504
**all-write** ablation, not against the strongest-zero or same-coordinate-bit
baseline used by `1.759x`. They may appear as two explicitly separated rows in
a paper component table; they may not be fused into a single energy-efficiency
claim.

No EDA, GPU, remote job or production analyzer was run. All checks were
read-only. `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## 1. Evidence that is citable now

### 1.1 M623 nine-macro parent-scratch component ablation

M617's canonical result and consumed coordinate, and M623's independent
`99/100, P0/P1/P2=0/0/0` result hammer, pass their inner and outer seals.
M595's earlier `40.5634%` result was correctly rejected because it paired M504
cycles with M473 read traffic; that number is permanently non-citable.

The admitted M623 row is:

| Schedule | Reads / S10 | Writes / S10 | Dynamic | Leakage | Nine-macro component total |
|---|---:|---:|---:|---:|---:|
| M504 all-write 1RW | 131,926,088 | 218,444,544 | 3.2280012413 mJ | 0.0738875876 mJ | 3.3018888289 mJ |
| M528 dead-write-only 1RW | 131,926,088 | 79,581,608 | 1.9691027740 mJ | 0.0705298262 mJ | 2.0396326003 mJ |

The admitted saving is **1.2622562287 mJ per frozen sampled inference**, or
**38.2283079%**, for nine generated `128x128-bit 1RW` parent macros only. The
same internal ablation gives `456,016,645 / 435,293,339 = 1.04760768x` cycles.

Publication label must retain all of: `generated-macro datasheet component
model`, `nine parent-scratch macros`, `ten frozen sampled inferences`, `one
sequence`, `H67 ep35`, and `four bottleneck Conv3x3`. A sampled inference is
not established as a camera frame. This row excludes logic, other SRAM,
interconnect, clock tree, DRAM and full-network energy; it is not integrated
PPA or silicon.

### 1.2 M1125C 105-macro identical static common charge

M1125C and its seals admit only a capacity-equivalent external static model:

| Quantity | Identical charge per axis |
|---|---:|
| Required / rounded capacity | 214,912 / 215,040 B |
| Native macro equivalents | 105 x `128x128-bit 1RW` |
| Padding | 128 B |
| Area `[model]` | 919,627.85775 um² |
| Slow-corner leakage power `[model]` | 6.30109935 mW |

This is exact capacity rounding using the pinned private macro manifest, not a
physical 105-instance organization. The proposed 60-macro psum depth packing,
24-macro weight half-slot packing and 12-macro residual have no sealed complete
port schedule. The residual is conservative common padding, not live storage.
M962/M1000-style tops that already include nine parent macros cannot be added
directly to this charge without subtracting the exact parent contribution or
resynthesizing a zero-storage logic boundary.

Area and leakage **power** are common coefficients. Leakage energy is
axis-dependent (`6.30109935 mW * T_axis`) and is not presently admitted as a
matched energy result because the three-axis executable memory schedule is
open.

## 2. Compatibility with the latest `1.759x` ledger

M1114 independently admits the M1102 raw-CPU result:

| Axis | Cycles / S10 |
|---|---:|
| Candidate | 434,242,823 |
| Strongest-zero | 763,908,050 |
| Same-coordinate bit | 763,908,050 |

This is `1.7591725402x` on 10 samples, 812,160 tasks and four frozen H67
bottleneck Conv operators. It is raw CPU same-ledger opportunity, not RTL or
mapped speedup.

The candidate parent vector is exactly compatible with M623:

- reads: `131,926,088`;
- writes: `79,581,608`;
- RAW forwards: `13,717,024 = 1,714,628 x 8`;
- the H67 ep35 population, ten samples, four Conv operators and one sequence
  are identical.

But the latest candidate cycle count is `1,050,516` cycles (`0.241335%`) below
M623's M528 schedule. Therefore M623's admitted dynamic energy remains
access-compatible, while its leakage and total are not literally computed on
the final `1.759x` cycle row. A read-only diagnostic using the frozen nine-macro
coefficient would be `0.0703596129 mJ` leakage and `2.0394623870 mJ` parent
component total per sampled inference. Those two updated values are **not
admitted** until a small exact-SHA adapter/result hammer binds M1102 and M623.

More importantly, strongest-zero and same-coordinate-bit have zero parent
accesses in M1102. The `38.2283%` saving compares dead-write-only with an
all-write version of the same candidate mechanism; it is not candidate energy
saving versus either `1.759x` baseline. A legal paper table can therefore use:

1. `1.759x` as a raw-CPU component cycle-opportunity row; and
2. `38.2283%` as a separately scoped dead-write suppression ablation row.

It cannot report `1.759x / 38.2% energy reduction` as one candidate-vs-baseline
pair, and it cannot derive TOPS/W or energy efficiency from the two rows.

## 3. DRAM sensitivity boundary

The frozen M528 traffic aggregate charges `9,069,207,552` weight-DRAM bytes
over S10, identically for its candidate and zero rows. Applying the CICC'26
`3.7 pJ/bit` coefficient gives the arithmetic diagnostic:

`9,069,207,552 B * 8 * 3.7 pJ/bit / 10 = 26.84485435392 mJ`

per frozen sampled inference. This may be shown only as a clearly labelled
**CICC-coefficient DRAM sensitivity `[model]`** after a small result seal. It is
not currently a canonical energy result, is not address-timed, has no DRAM
command/refresh/row-buffer model, and is not DRAMsim3 output. It must not be
called measured DRAM energy or energy/frame.

The sensitivity is common traffic in the frozen M528 comparison and therefore
does not itself create a speedup or DRAM-energy advantage. It also demonstrates
why the parent-only number cannot stand in for C1 total energy: the admitted
2.0396-mJ parent component is about 7.60% of this diagnostic common DRAM term,
and the 1.2623-mJ dead-write saving is about 4.70% of it, before logic and other
SRAM are included.

## 4. Why a complete paper energy table is still STOP

M1126C correctly stops before opening the 51.84M-row canonical population.
M1127C independently confirms that parent and psum events are reconstructable,
but the frozen weight service lacks native READ/WRITE, half-slot/local address,
byte enable, macro activation multiplicity and exact-one DRAM-beat-to-on-chip
store provenance. Consequently:

- no complete three-axis native SRAM read/write totals exist;
- psum paired-depth and weight half-slot 1RW conflict freedom is unproven;
- the 105-macro dynamic-energy equation cannot be evaluated;
- matched candidate/zero/bit logic SAIF/PTPX energy is absent;
- exact-once storage double counting is not closed;
- no total C1 energy, average power, throughput/mm² or energy-efficiency ratio
  is admissible.

The M1126C five-transaction oracle validates schema and arbitration mechanics
only. It is not H67 traffic or energy evidence.

## 5. Minimum next steps

The shortest route to a paper-ready energy table is additive and does not
require a new architecture:

1. **Bind the existing parent component to M1102.** Author a tiny exact-SHA
   adapter over M1102 parent counts/cycles and M623 coefficients, then obtain a
   different-author result hammer. This can legitimately place the parent
   component next to `1.759x`, while retaining the component-only label.
2. **Repair the weight provenance gap.** From the same M1102 semantic iterator,
   freeze one addressed weight-service ledger containing op, half-slot/local
   address, byte enable, native activation count, task/local ordinal and source
   row SHA. Hammer it before full export.
3. **Run the successor three-axis exporter.** Emit parent/psum/weight
   cycle-address transactions, prove exact conservation and charge every 1RW
   conflict as a stall. Any psum depth-share or weight half-slot conflict must
   re-bank/enlarge the 105-macro point.
4. **Close matched logic energy.** Use zero-storage candidate/strongest-zero/bit
   tops with one library/SDC/debug policy, produce matched SAIF/PTPX, then add
   the external dynamic/leakage model exactly once.
5. **Keep DRAM separate.** Seal the 3.7-pJ/bit value only as sensitivity and use
   an address-timed DRAMsim3-equivalent row for the main memory-inclusive table.

Until steps 2--4 close, the strongest legal paper energy presentation is a
component-ablation table plus a static capacity-model table, not a C1 total
energy comparison.

## Seal checks

Inner and outer seal verification passed read-only for M617 result, its
permanently consumed attempt, M623 result hammer, M1125C audit, M1127C source
hammer and M1114 result hammer. No canonical member was changed.
