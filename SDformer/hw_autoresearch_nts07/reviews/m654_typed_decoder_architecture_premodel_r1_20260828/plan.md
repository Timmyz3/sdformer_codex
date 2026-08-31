# M654 conditional mixed-type ConvTranspose decoder premodel

Date: 2026-08-28  
Status: **PLAN ONLY — M649 outcome unknown; no cycle, speedup, RTL, GPU or EDA result**

## 1. Fail-closed verdict

The failed M511 one-shot invalidates the old claim that every raw decoder
`ConvTranspose2d` input is binary.  It does **not** invalidate the first D0
record: that record is a sealed 576,000-byte exact bitpack containing
4,608,000 bits, 839,586 ones and 18.220182292% activity.  The producer then
failed at D1 with `RuntimeError: M511 raw ConvTranspose2d input is not exact
binary`; the canonical M511 payload is absent and the one-shot is consumed.

Source order supports, but does not measure, a narrower hypothesis.  For D1–D3,
`Spiking_STSwinNet.py` calls
`skip_ftn(predictions[-1], x, dim=2)`, while `skip_concat` returns
`torch.cat([x1, x2], dim=dim)`.  The previous two-channel flow prediction is
therefore expected at channel indices 0 and 1.  M649 is the independent numeric
audit of this hypothesis.  It is currently source-authored only; its canonical
result is absent.  This plan never assumes that M649 will pass.

Only the following measured M649 outcome may advance this plan:

- all 10x4 inputs are finite float32 records with the frozen identities;
- D0 is exact `{0,1}` in every channel;
- D1–D3 channels `[2,C)` are exact `{0,1}`;
- D1–D3 channels `[0,2)` are finite and contain measured non-binary values;
- no sample, module, channel position or identity is relaxed or selected after
  observing performance.

If any item fails, the mixed-type decoder is `NO_GO` and no thresholding,
coercion or channel permutation is allowed.

## 2. What is reusable and what is not

Reusable exact infrastructure:

1. M514 maps a real K3/S2/P1/OP1 source directly to its legal 4/6/9 taps using
   `dst=2*src-1+k`, without materializing inserted zeros.  Directed VCS passed
   43 taps, all four phases, stalls, same-edge replacement and a protocol
   attack.  The standalone mapper met 3 ns in pre-macro DC at 383.670001 um2.
2. M523 proves directed, same-tag/time cross-event packing of legal taps into
   K8 bundles, including event, time, tag and stream-last boundary flushes.
3. C2's typed signed-source contract supplies the conceptual K8 service and
   Acc24 completion boundary.  The decoder must instantiate a decoder-specific
   model and may not inherit FC2 cycle ratios.
4. C1/M528 supplies the exact subset-parent/residual algorithm, a physical 1RW
   parent-scratch discipline, dead-write elision and conservation equations.
   The decoder must rebuild every count and capacity entry from its own trace;
   no M528 1.741232x cycle number transfers.
5. Prosperity's official `find_product_sparsity` semantics may be used as an
   externally cited parity oracle.  The decoder claim is the typed split and
   finite-resource capture path, not invention of product sparsity.

Not reusable as performance evidence:

- M510's 4.4767–4.8139x dense/sparse range is an aggregate analytical bound
  built before the mixed type was known.
- M514/M523 are functional support, not decoder cycles.
- M216's K8/K1 ratio is FC2-only; K8 versus equal-service K1x8 is the required
  decoder comparison.
- M528's bottleneck-Conv cycles, 213,376-byte point and generated-macro energy
  are not decoder measurements.
- the M512 phase-balanced EPD scheduler remains killed.  All configurations in
  this plan receive the same polyphase mapping and deterministic bank support.

## 3. Conditional exact arithmetic decomposition

For a legal transposed-convolution contribution,

```text
y[t,co,dy,dx] += x[t,ci,sy,sx] * W[ci,co,ky,kx]
dy = 2*sy - 1 + ky,  dx = 2*sx - 1 + kx.
```

Under the exact M649 split, D1–D3 are decomposed without approximation:

```text
y = W[:, 0:2] * flow[0:2] + W[:, 2:C] * binary[2:C].
```

D0 contains only the second term with the full channel range.  Binary zeros are
skipped exactly; binary ones enter the product-capture path.  A nonzero flow
scalar enters a typed signed-source residual path and never enters the binary
subset matcher.  Both paths meet only at the same Acc24 destination state.

The source-ordered channel split is a static module descriptor, not an online
classifier.  It costs a two-bit source type plus channel-range comparison; it
must not scan values to choose a fast path.

The flow path is exact only if a successor capture proves one of these two
conditions before simulation:

1. every flow value is exactly representable by the frozen signed-source
   fixed-point width and scale; or
2. a common full-precision fallback unit and its traffic, latency, area and
   energy are present in **every** configuration.

If neither condition holds, Acc24 cannot silently replace float32 and the exact
typed path is `NO_GO`.  For a fixed-point path, export the frozen signed-INT8
decoder weights, prove the source/product widths, use a wider internal sum if
needed, and prove every canonical result fits Acc24 with zero overflow and zero
mismatch.  Reassociation is permitted only after integer no-overflow proof;
floating-point reassociation is not called exact.

## 4. Decoder product capture

For each fixed `(sample, module, timestep, kernel tap, output-96 block,
input-channel-16 partition, spatial row tile)`:

1. form masks only from the measured binary channel suffix;
2. run the clean-room strict-subset parent selection and check it against the
   official Prosperity function on a frozen stratified subset;
3. reconstruct every child mask as `parent XOR residual` and require zero mask
   mismatch;
4. compute one parent partial output vector and only residual product terms for
   a child;
5. read/write the parent partial through a charged 1RW scratch; do not use the
   M473 concurrent 1R1W ceiling;
6. add typed flow-channel products independently at the same destination;
7. commit a destination once all phase/tap, binary residual and flow terms have
   retired.

Parents may cross neither module, timestep, kernel tap, output block, weight
identity nor precision type.  Row-tile boundaries are physical boundaries.
Future-parent edges require an explicit bounded order directory and are
charged; an offline reordering oracle is forbidden.

M514's tap fanout and M523's K8 packer are common adapters.  A bundle may combine
only compatible tag/time/weight/output-state contexts.  All forced flushes and
bank conflicts are cycles, not free metadata.

## 5. Fair denominator ladder

Every row below uses the same frozen 10-sample population, K3/S2/P1/OP1
polyphase map, signed-INT8 decoder weights, output precision, Acc24 semantics,
3.0 ns clock, 240 KiB physical on-chip SRAM cap, 64 GB/s decimal DRAM cap
(192 bytes/3 ns cycle), cache policy, output tiling, queue depths, external
ports and completion definition.  No baseline materializes inserted zeros.

| ID | Exact execution | Service/resources | Purpose |
|---|---|---|---|
| B0 Dense96 Fixed-T10 | issue every real input source for all ten timesteps and every legal tap, including numeric zeros | eight-source service ceiling and eight weight banks; no sparsity metadata | strong dense, not a zero-insertion strawman |
| B1 PTB-like K1x8 | for each frozen eight-source group, issue all eight lanes when any lane is nonzero; skip only an empty group | eight scalar services with all replicated state/control charged | project-defined structured baseline; never label official PTB |
| B2 exact K1 | issue each exact nonzero binary or typed-flow source through one scalar service | one service; same memory capacity/ports | bandwidth-scaling diagnostic only |
| B3 exact K1x8 | issue exact nonzero sources through eight independent scalar services | replicated queue/control/state and eight weight banks charged | strongest equal-service baseline |
| C2 typed K8 | same exact nonzero source multiset as B3, one shared typed descriptor/state service, maximum eight accepted sources | shared K8 logic; same external eight-bank peak | isolates K8 organization; compare only with B3 |
| Ours mixed capture | C2 plus exact suffix-only subset-parent/residual capture and separately charged two-channel flow residual | K8 + 1RW parent scratch + directory + flow skid/type state | candidate decoder contribution |

The implementation manifest must freeze the eight-source grouping before the
run.  B0/B1/B3/C2/Ours have the same peak source and weight service.  B2 cannot
be the headline denominator.  K8/K1 is a bandwidth Pareto point, while
Ours/B3 is the fair acceleration comparison.

All rows must execute the analog flow work at the same precision.  B0 issues it
dense; B1 applies only its predefined group rule; B2/B3/C2/Ours may skip exact
zeros.  Only Ours may reuse a binary parent product.  Unsupported work executes
inside the same model; no row may omit it.

## 6. Common 240-KiB/Acc24 ledger

The simulator rejects a point unless the sum of macro-rounded capacities is at
most 245,760 bytes.  It reports logical and physical bytes separately and
charges at least:

- Acc24 resident destination state and validity bits;
- eight-bank weight tile/cache and tags;
- binary suffix bitmap ping-pong storage;
- typed flow payload/skid storage and type bits;
- M514 tap FIFO and M523 K8 bundle/completion queues;
- row/order tags, source descriptors and fixed control reserve;
- for Ours only, row-indexed parent partial scratch, liveness/refcount metadata,
  matcher masks and scheduler queues;
- for B1/B3, every replicated queue/control/state item;
- DMA staging that overlaps compute.

M528 shows why a new decoder ledger is mandatory.  Its admitted Conv candidate
uses 213,376 macro-rounded bytes with 19-bit psums.  Merely replacing its psum
item by a minimum-depth 8-bank, 96-lane Acc24 item would consume approximately
another 27,648 bytes and leave only 4,736 bytes.  That is a topology warning,
not a decoder capacity result: decoder weight tiles, flow state and tap queues
differ.  The new ledger must either fit them honestly, reduce a common tile
dimension/bank organization for **all** rows, or fail.  It may not call the
mapper's standard-cell area free SRAM.

The parent partial width is independently derived from decoder INT8 weights and
the 16-source partition.  It is not inherited as signed12.  Final accumulation
remains Acc24.  Parent scratch is 1RW unless a real 1R1W macro, area and energy
are supplied for every compared configuration that needs it.

## 7. Executable cycle model

Use one deterministic, address-timed discrete-event simulator.  For each
sample, execute modules D0 to D3 in frozen model order and timesteps 0 to 9.
Within a module execute output-96 blocks, phase banks, row tiles and output-16
slices using one predeclared order.  The simulator advances only when all
resource and dependency conditions are met:

1. source scan/decode and K8 grouping;
2. M514 tap production and M523 bundle acceptance;
3. weight SRAM hit or address-timed DRAM refill;
4. optional parent search, 1RW read/forward and residual issue;
5. typed flow issue;
6. Acc24 bank read/modify/write or forwarding;
7. completion and output commit.

One cycle may overlap independent stages, but each physical port accepts at
most its declared transactions.  Total latency is the simulated completion
time, not a sum of isolated components and not `max(compute,memory)` applied
after the fact.  Emit per-cycle resource occupancy or an equivalent event log
that can reconstruct:

- useful binary, flow and parent/residual issues;
- source-scan, matcher, tap, weight, parent, Acc and commit transactions;
- compute, weight, parent-port, Acc-bank, queue and completion stalls;
- SRAM/DRAM reads and writes by named object;
- final completion count and exact conservation equalities.

Required conservation for every configuration:

```text
accepted real sources = skipped_exact_zeros + issued_sources
legal taps = mapper outputs = bundled taps + boundary flush payloads
weight responses = retired direct/residual/flow products
parent reads + forwards = consumed parent edges
parent writes + exact elisions = generated reusable parents
Acc updates = direct + residual + parent + flow contributions
commits = complete output vectors; all queues and refcounts drain to zero
```

Ours must also reconstruct every binary suffix mask and match an independent
fixed-point ConvTranspose reference at every committed output.  A cycle result
with no numeric miter is rejected.

## 8. Structural bounds, not performance claims

M510's topology provides two useful ceilings, still labelled analytical:

- the omitted decoder accounts for 21.5720–22.8262% of the corrected analytical
  envelope;
- deleting the decoder entirely would cap old-scope overall improvement at
  1.27505–1.29578x.  No decoder mechanism can exceed this overall ceiling.

If M649 proves the proposed split, the two flow channels of D1–D3 correspond to
452,974,080 dense products per frame under B0: 40,669,440 in D1, 82,139,520 in
D2 and 330,165,120 in D3.  This is 0.5744865% of the 78,848,509,440 dense
decoder products.  Therefore 99.4255% of dense decoder arithmetic is
**binary-path eligible**, not 99.4255% skippable.  Skip and parent-reuse rates
remain unmeasured until exact payload capture.

For every measured baseline `b`, the only valid local ceiling is emitted from
the same run:

```text
S_decoder = C_b / C_ours
C_ours >= C_common_commit + C_flow + C_parent_port_and_residual
S_corrected_sensitivity = E / (E - D_b + D_ours)
```

where `E` is a frozen corrected envelope and `D_b,D_ours` are directly measured
decoder cycles in the same scheduler.  The last expression is a labelled
sensitivity until the decoder is integrated into the complete ordered system
simulation.  Product-count reduction, traffic reduction and cycle speedup are
three different columns.

## 9. One-day decision gate after M649

No RTL is written first.  The next one-day job is authorized only by a fresh
M649 result hammer with P0=0/P1=0 and an explicit typed-split GO.

1. Capture 40 records into typed payloads: a suffix bitpack plus exact first-two
   flow values for D1–D3, and a full bitpack for D0.  Seal global call order.
2. Export and seal the four frozen signed-INT8 decoder weight tensors.
3. Build the common B0/B1/B2/B3/C2/Ours simulator and capacity ledger; run
   clean-room/official parent parity on a frozen stratified subset.
4. Run all ten samples and report ratio-of-sums, arithmetic/geometric mean,
   minimum/maximum per-sample speedup, cycles, stalls and traffic.
5. Run independent numeric/conservation/capacity checks before interpretation.

Decision:

- `GO_RTL_MAIN` only if exact output mismatches and overflow are zero, physical
  capacity is `<=245760 B`, Ours is at least 1.20x versus B3 in total decoder
  cycles, every sample is at least 1.10x, and no hidden service/port differs.
- Mark it a strong decoder candidate only at at least 1.30x versus B3 with a
  minimum per-sample speedup of 1.15x and at least 20% measured DRAM-byte
  reduction.  These are gates, not predictions.
- `SUPPORT_ONLY` if cycle speedup is 1.10–1.20x or if cycles are weaker but DRAM
  bytes fall at least 30%; report the successful axis only.
- `NO_GO_RTL` if exactness/capacity/fairness fails, or if speedup is below 1.10x
  and DRAM reduction below 20%.

The gate compares Ours with B3, never with K1.  B0 and B1 remain useful waterfall
rows, while official Prosperity remains an external opportunity reference.

## 10. Paper-safe contribution if the gate passes

The legal claim is not "a new product-sparsity algorithm."  It is:

> We expose an exact heterogeneous decoder boundary in event optical flow: a
> two-channel signed flow residual precedes a high-dimensional binary feature
> suffix.  A typed polyphase path routes the residual through signed K8 service
> while applying finite-1RW parent/product capture only to the binary suffix,
> then atomically merges both into Acc24 under a 240-KiB memory budget.

The paper must cite Prosperity for product sparsity, Delta-style work for
residual/delta computation, and transposed-convolution polyphase decomposition.
M514/M523 are support hardware under C2; the typed decoder is not a fourth
unrelated novelty.  Table B may include it only after exact cycles and physical
cost exist.  Table A still requires the decoder-complete unified system run.

## 11. Frozen claim boundary

This premodel performed no GPU, CPU performance simulator, VCS, DC, Formality,
PTPX, DRAMsim3 or remote run.  It authorizes none.  It contains no measured
decoder cycle, speedup, traffic, energy, PPA or system result.  M649 remains
unmeasured at the time of sealing.  `docs/359` remains unchanged at SHA256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
