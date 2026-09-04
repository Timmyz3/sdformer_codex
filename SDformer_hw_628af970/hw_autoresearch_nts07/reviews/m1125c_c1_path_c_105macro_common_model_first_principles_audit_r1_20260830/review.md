# M1125C — C1 Path-C 105-macro common-model first-principles audit

## Verdict

**Split verdict:**

- **GO** for an identical, capacity-equivalent external **static `[model]`**
  charge of 105 native `128x128-bit 1RW` macro equivalents per axis;
- **STOP** for per-axis full external dynamic energy, executable 1RW mapping,
  physically integrated area/timing, throughput/mm², or total C1 energy with
  the evidence currently sealed.

The 105-macro point is exact capacity rounding, not a physical implementation:

```text
214,912 B required
ceil(214,912 / 2,048) = 105 native macro equivalents
105 × 2,048 B = 215,040 B
padding = 128 B
```

Using the pinned foundry manifest area and the independently hammered M623
slow-corner leakage coefficient gives the following common static model:

| quantity | identical charge per axis |
|---|---:|
| native macro equivalents | 105 |
| physical capacity `[model]` | 215,040 B |
| padding `[model]` | 128 B |
| area `[model]` | 919,627.85775 µm² |
| leakage power at `ssg0p9v125c` `[model]` | 6.30109935 mW |

The area and leakage **power coefficient** are common. Leakage energy is not
identical when axis execution times differ; it remains
`6.30109935 mW × T_axis`.

## Port, width and depth audit

The native cell is `TS1N28HPCPHVTB128X128M4S`: depth 128, width 128 bits,
one synchronous 1RW port, area 8,758.36055 µm². Its slow-corner cycle/access
figures are 0.616/0.4679 ns. Those cell figures do not prove a 3-ns integrated
wrapper, arbitration network or 105-instance top.

The proposed 93-macro known-geometry subtotal is arithmetically consistent but
not yet an executable geometry:

- parent: 9 macros in parallel for one 1,152-bit word;
- psum: 15 macros in parallel provide 1,920 bits for an 1,824-bit row. The
  60-macro count assumes two logical 64-row banks can share each 128-row 1RW
  depth group; no cycle-addressed conflict proof is sealed;
- weight: 24 macros in parallel provide one 3,072-bit word. The 24-macro count
  assumes two half-slots are never demanded concurrently; no such trace is
  sealed.

Thus `93 × 2,048 = 190,464 B` is a proposed capacity organization, not a proven
93-macro port mapping. The remaining `24,448 B` rounds to 12 macro equivalents,
but has no frozen logical width, bank/depth allocation, port schedule, native
macro activation multiplicity or access trace.

## Double-count audit

A 105-macro common charge includes all nine parent macro equivalents. Any
matched logic boundary used with it must therefore contain **zero parent SRAM
macros**. The M962/M1000 component top contains nine parent macros and cannot be
combined directly with this charge. The legal remedies remain: resynthesize a
zero-storage logic top, or subtract the independently sealed exact parent area,
leakage and dynamic contribution before adding the common model.

The 12-macro residual is also not a physical residual implementation. Its
24,448-B derivation contains macro proxies for descriptor, source-mask and
liveness state already flattened into the measured logic, an analytical
16-KiB FIFO/control reserve that is not instantiated, plus active/psum-valid
proxies without a complete mapping. Charging all 12 externally while retaining
those internal states is a deliberately conservative common denominator, but
it is not a no-double-count physical total. Therefore the 919,627.85775-µm²
number may be called only an **identical capacity-equivalent external area
charge `[model]`**, never integrated or total area.

## Dynamic-energy evidence audit

M623 establishes a bounded coefficient for the same native macro model:

- 10.50786 pJ per activated native-macro read;
- 10.07307 pJ per activated native-macro write;
- 0.06001047 mW leakage per native macro.

It also seals complete aggregate parent accesses for one candidate schedule:
131,926,088 reads and 79,581,608 dead-write-only writes across ten sampled
inferences. That supports the already admitted parent-scratch component result,
not a 105-macro or three-axis energy result.

M1102 seals aggregate cycles for candidate, strongest-zero and
same-coordinate-bit, and candidate parent read/write/forward counts. It does
not retain a per-cycle/per-address transaction stream. It reports no complete
three-axis read/write counts for psum, weight store, psum-valid, descriptor,
mask, liveness, active bitmap or FIFO/control reserve. M528's source-SRAM and
weight-DRAM byte totals are traffic aggregates, not native-macro activation
traces; psum accesses are not fully exposed. Therefore the equation

`E_dyn_axis = Σ(native_reads × 10.50786 pJ + native_writes × 10.07307 pJ)`

is valid as a future model, but its three axis-specific totals cannot currently
be evaluated. Inventing identical dynamic energy from identical capacity is
forbidden.

## Shortest executable next step

Create one additive, source-only M1126C extractor contract over the frozen
M1102 semantic replay. For all 812,160 tasks and all three axes it must emit a
canonical cycle/address transaction stream with:

`axis, sample, task, cycle, storage_class, R/W, logical_address, logical_width,
byte_enable, native_group, native_macro_activation_count`.

The storage classes must cover parent, psum, psum-valid, weight half-slot,
descriptor, source-mask, liveness, active bitmap and actual FIFO/control state;
the analytical reserve must either be replaced by actual state or remain a
zero-dynamic common padding term. A different-author hammer must then prove:

1. exact conservation against M1102 cycles/work and all sealed M528/M623 parent
   counts;
2. zero same-cycle conflicts for every proposed shared 1RW depth group;
3. exact per-axis native read/write counts and coefficient application;
4. exact-once charging of every internal or external storage row.

If any psum paired-bank or weight half-slot conflict exists, the 105-macro point
is STOP for executable modeling and must be re-banked or enlarged before any
energy/cycle result. No EDA is needed for this next gate.

## Publication-safe boundary

Safe now:

> We apply an identical 215,040-B capacity-equivalent external SRAM charge
> (105 128×128-bit 1RW macro equivalents, including 128 B padding) to all three
> C1 axes. At the pinned 28-nm slow-corner model this corresponds to a common
> 919,627.85775-µm² area and 6.30109935-mW leakage-power charge `[model]`.

The sentence must be followed by: the organization is not physically
integrated; 1RW conflict freedom and complete per-axis dynamic access energy
are pending. The model number cannot be paired with the raw M1102 ratio as
measured throughput/mm², and cannot be called total C1 or system PPA.

No RTL, source, EDA result, GPU/remote job or canonical result was modified.
`docs/359` remains at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
