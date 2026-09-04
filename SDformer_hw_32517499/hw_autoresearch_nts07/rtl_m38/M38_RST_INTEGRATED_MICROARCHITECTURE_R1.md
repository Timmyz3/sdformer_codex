# M38-RST integrated microarchitecture contract, revision 1

Status: milestone-1 specification only. No integrated RTL, executable cycle,
DC/STA/Formality, area, power, energy, trained accuracy, or system-speedup claim
is admitted by this document.

## 1. Scope and arithmetic identity

M38-RST is a T10-only rank-3 reconstruction path added beside the sole M31
96-lane signed-INT8 multiplier pool. The pool continues to perform the five
stage-1 reduction phases. M38 reconstructs two temporal rows per phase using
30 layer-resident ternary left-factor codes. T2 remains the existing dense
24-lane M31 mode and never enters M38.

Each two-bit left code is frozen as:

| Code | Coefficient | Meaning |
|---|---:|---|
| `00` | 0 | zero product |
| `01` | +1 | sign-extended q8 input |
| `10` | -1 | widened negation |
| `11` | illegal | configuration is rejected fail-closed |

Negation occurs after widening: `-(-128)=+128`, so each ternary product is
signed 9 bit. Three products span `[-384,384]` and require signed 10 bit. The
rank sum is sign-extended into the same explicit signed-Q26 pre-saturation
contract used by M31, then clamped to signed Q24 and compared with
`saturated_value >= threshold`. Equality therefore emits an event.

The rank-column scale absorption that makes the left factor ternary is an
algorithm/export requirement. It is not implemented as a runtime per-rank
scale unit. Any checkpoint requiring a runtime non-dyadic scale, a second
multiplier, or a dense T10 fallback is outside M38-RST admission.

## 2. Integrated phase schedule

The conditional no-backpressure schedule is:

```text
cycle 0..4    sole pool stage1(A)
cycle 5..9    sole pool stage1(B) || M38 stage2(A)
cycle 10..14  sole pool stage1(C) || M38 stage2(B)
```

For `N` consecutive resident-context T10 tiles, the theoretical arithmetic
schedule is `5 + 5*N` cycles versus M31's serialized `10*N`. The conditional
steady II is five and the conditional T10 kernel throughput ratio is 2.0.
These equations exclude parameter loading, result backpressure, memory,
operator boundaries, T2 work, attention, and all other full-network work.
They are not Local or Motion system speedups.

## 3. Single intermediate elastic slot

The integrated target has one 384-bit q8 intermediate slot, not the two input
banks of the standalone M37 block. M31's existing 48x24-bit stage-1
accumulators provide the second holding location while a completed tile waits.

Required state is:

- `slot_valid`, `slot_tag`, and `slot_data[383:0]`;
- `stage1_active`, `stage1_complete_pending`, and the source input-bank/tag;
- independent stage-1 and reconstruction phase counters;
- an active context generation captured with both data paths.

The slot supports simultaneous retirement and replacement:

```text
slot_pop  = reconstruction_issue && reconstruction_phase == 4
slot_push = stage1_commit && (!slot_valid || slot_pop)
```

On simultaneous pop/push, phase 4 consumes the old slot value and the clock
edge installs the new tile and tag. When `slot_valid && !slot_pop`, a finishing
stage 1 must retain its final accumulated state as `stage1_complete_pending`;
it must not overwrite the slot or begin another arithmetic tile. Once slot
credit returns, RNE/saturation materializes that pending tile into the slot.

No stage-1 or M38 state may be indexed by live input pins after its handshake.
Tag and context generation travel with every slot and product/output stage.

## 4. Result credit and shared FIFO

M38 writes the existing shared 16-entry, 48-bit M31 result FIFO. It does not
instantiate a private FIFO. A T10 reconstruction may start only when at least
five entries are free, counting a same-cycle FIFO pop as credit. Five entries
are atomically reserved for the tile, so reconstruction cannot stop between
beats 0 and 4 under ordinary sink backpressure.

The FIFO must support full simultaneous pop and push without changing logical
occupancy or corrupting the old read value. `done` pulses when beat 4 is
committed into the shared FIFO, not when beat 4 is consumed. FIFO ownership,
tag order, and beat order remain valid across a stage-1 completed-pending
condition.

## 5. Configuration integrity and bandwidth

The resident T10 arithmetic payload is:

- 30 signed-INT8 right-factor entries: 240 bit;
- 30 ternary left codes: 60 bit;
- 10 signed-Q24 biases: 240 bit;
- one signed-Q24 threshold: 24 bit;
- one stage-1 requant shift: 5 bit;
- total arithmetic payload: 569 bit.

A production context additionally carries a 16-bit monotonically managed
generation and CRC-32C/Castagnoli, for 617 logical bits before physical
SRAM/ECC rounding. CRC-32C uses normal polynomial `0x1EDC6F41` (reflected
`0x82F63B78`), initial value and final XOR `0xFFFFFFFF`, and reflected input and
output. The protected logical payload order is right factor, left codes, bias,
threshold, requant shift, then generation, serialized least-significant bit
first and zero-padded to the next byte. Loading may use a 64/96-bit phased
configuration stream; the wide logical payload is not evidence of a free
single-cycle physical port.

Right factor, ternary codes, bias, threshold, and requant shift become visible
atomically only after all fragments and CRC pass. Data is accepted only when
its requested generation matches the active context. Configuration release or
replacement is legal only after input banks, stage-1 active/pending state,
intermediate slot, product/output state, and result FIFO are fully drained.

Parameter transactions must later be represented in the address-timed system
model. Per-tile external parameter reload is forbidden. The expected M29 scope
is 45 constrained T10 modules; the expected 60 T2 modules remain dense.

## 6. Fair hardware comparison

The next integrated comparison must use four tops under identical libraries,
constraints, hierarchy policy, input banks, scheduler, single intermediate
slot, FIFO depth, output path, generation/CRC policy, and activity stimulus:

1. serialized M31 shared96;
2. M31 plus a direct second signed-INT8 mul96 pool;
3. normalized integrated M37 CSD4;
4. integrated M38-RST.

Standalone block area must not be compared with the 96-lane pool hierarchy.
Total integrated delta and arithmetic/config/control/storage breakdowns are
both required. Algorithmic-equivalent products and CSD/ternary terms must not
be relabeled as identical physical GOPS.

## 7. Required VCS evidence

- all 768 q8-by-legal-ternary scalar pairs;
- explicit illegal `11` rejection and live-code perturbation after load;
- direct product, rank sum, saturation, threshold equality/just-below, and
  output-bit miters, including `-(-128)=+128`;
- at least 1,000 consecutive no-stall tiles with every steady accept interval
  equal to five and stage1/reconstruction simultaneous-issue coverage;
- phase-4 slot pop/push, stage1 completed-pending, FIFO-full, FIFO-full
  simultaneous pop/push, and long sink-stall coverage;
- T10 to nondegenerate T2 to T10 context changes with complete drain;
- generation mismatch, incomplete fragment, bad CRC, busy release, and illegal
  ternary configuration rejection;
- no data multiplication operator outside the sole M31 pool.

## 8. Required Synopsys admission

At both 3.000 ns and 2.000 ns, DC/STA must report setup and hold MET under the
same target library and constraints as the direct-second-pool control. Resource
audit must find exactly the original sole 96-lane multiplier hierarchy and no
M38 multiplier cone. Formality must finish with a nonzero compare-point set,
all passing, zero failing, wrapper exit zero, and sealed input/output evidence.

Nominal 2.000 ns closure alone is not a robust-500-MHz claim. Such a claim also
requires an explicit uncertainty/derate or tighter constraint, macro-aware
timing, and non-ideal interconnect evidence.

Area and energy thresholds are pre-registered but unmeasured: M38 integrated
area delta should be at most 60% of the area-matched direct-second-pool delta
for a strong GO, and same-trace PTPX T10 tile energy should be at most 65%.
Neither threshold is a result until the corresponding Synopsys run exists.

## 9. Accuracy and line scope

H67/Motion may use the existing M29 rank-3 training interface but needs an
integer QAT/export path that emits only legal ternary codes and requires no T10
dense fallback. Local5 must have an independent ep44 module census,
constrained checkpoint, export, and valid825 result; H67 descriptors cannot be
reused. An accuracy pass is provisionally `delta AEE <= 0.02` against the exact
same checkpoint/evaluator's frozen deployed baseline, plus zero Python-integer
to RTL mismatches.

M38 accelerates only the shared T10 ATLIF operator population. It does not
accelerate T2 fallback or the Local/Motion attention front ends. Full-network
cycles, FPS, energy, and system speedup remain false until address-timed traces,
memory, contention, and trained deployment evidence are closed.
