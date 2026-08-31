# M1101 C1 short-work first-principles semantic review

## Verdict

`GO_UNIQUE_ADDITIVE_DOMAIN_REPAIR__STOP_15_BANK_REINTERPRETATION`

The premise that frozen M1056 has a "15-bank event geometry" is false. M1056
schedules **8 logical psum banks** packed into **4 physical 1RW groups**. The
number 15 in M1064 is `ceil(1824/128)`: fifteen SRAM macros used in parallel to
form one 1,824-bit-wide word in each physical group. It is a capacity/width
factor, not fifteen independently timed bank accesses.

No source was changed, no full-row cycle replay was run, and no EDA/GPU/remote
job was started in this review.

## Physical meaning of `work_cycles`

M1016 defines one task over eight resident output-block banks. Its common
receipt charges two psum accesses for every one of those eight banks. For the
candidate, a per-output-block parent/product trace length is multiplied by 8;
for strongest-zero and same-coordinate-bit, input popcount is multiplied by 8.
M528 independently preserves the same unit: every per-block issue or 1RW
recurrence count is multiplied by `BLOCK_BANKS=8` before entering the pipeline.

Therefore frozen `work_cycles` is the total serialized arithmetic/parent work
for all eight output blocks. It is neither a count of physical SRAM macros nor
a freely chosen interval. Canonical positive values lie on the lattice
`8 * positive_integer`; zero denotes no arithmetic issue.

M1072 rederives this field from the exact frozen row bytes through M1016 and
passes it unchanged into the M1056 task plan. A value in 1..7 cannot be a valid
canonical M1016/M528 work value. A value in 8..14 can only be 8 in this frozen
coordinate.

## Why negative dependencies appear

For a task-local work value `w`, M1056 uses:

```
s = max(1, floor(w / 8))
read(b)  = start + b*s
write(b) = min(start + w, read(b) + s - 1)
delay(b) = write(b) - read(b)
```

for logical banks `b=0..7`. A dependency becomes negative only when a nominal
read is placed after the nominal work end. Exhaustive arithmetic over w=0..15
shows that frozen M1056 has negative delay for w=0..6, and nonnegative delay for
w=7..15. In particular, w=8 has minimum delay 0 and is legal.

If one incorrectly substitutes fifteen timed banks, the analogous arithmetic
has negative delay for w=0..13; w=14 already has delay 0. Thus even the
hypothetical statement "1..14 all produce a negative dependency" is off by
one. More importantly, that hypothetical geometry is not M1056.

Zero still needs an additive semantic repair because the unmodified M1056
generator emits eight read/write pairs even when no arithmetic work exists.
The physically consistent zero behavior is no psum event, no grant and no
`last_write` mutation.

## Candidate repairs and fairness

| Candidate | Physical legality | Frozen-ledger fairness | Risk of false acceleration | Decision |
|---|---|---|---|---|
| Schedule only an inferred subset of banks for short work | Legal only if the trace proves that the omitted output blocks do not exist | Breaks M1016's fixed eight-block task and its 16 common psum accesses unless all services and denominators are rebuilt as a new coordinate | High: silently deletes required psum work | `STOP` as a patch |
| Compress events or impose same-cycle read/write order | Fifteen width slices already act in parallel and need no event compression; one 1RW port cannot grant a logical read and write in one cycle | Changes the port model unless a different macro primitive is declared and applied to every design | High if used only to shorten the candidate | `STOP` |
| Pad short work to at least 15 cycles | Avoids the hypothetical failure but has no basis in the eight-bank schedule | Rewrites design-specific work and breaks exact M1016/M528 anchors; candidate-only padding is asymmetric, all-design padding is conservative but still a new denominator | No direct optimistic bias if universal, but it can distort ratios and is not an exact replay | `STOP` |
| Keep 8 logical banks; zero emits no events; positive work >=8 delegates bit-identically to M1056; 1..7 rejects | Matches the frozen work unit, port geometry and canonical value lattice | Preserves every common service, work anchor and design comparison | None: it adds no cycle shortcut | **Unique `GO`** |

## Unique recommendation

Use one additive domain adapter only:

1. `work_cycles == 0`: return an empty task result at `work_start`, with no
   psum event/grant and no `last_write` mutation.
2. `work_cycles >= 8`: delegate exactly to frozen M1056 without changing event
   generation, arbitration, timing or counters.
3. `1 <= work_cycles <= 7`: fail closed before scheduling.
4. At the provenance boundary, additionally assert the frozen M1016/M528
   lattice `work_cycles % 8 == 0`. Any 9..14 value would therefore identify a
   unit/provenance bug even though the generic M1056 arithmetic can schedule it.

Do not add a fifteen-bank event path, minimum-15 padding, bank-subset inference,
or same-cycle dual operation. If a later implementation requires a different
number of output blocks, it must be introduced as a new same-resource ledger
with new common-service counts and baselines, not as a repair to this replay.

## Claim boundary

This review authorizes no production replay and admits no cycle, speedup, PPA,
energy, full-network or paper result. It only supplies the semantic decision
for a separately authored and independently hammered additive repair.

`docs/359` remained at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
