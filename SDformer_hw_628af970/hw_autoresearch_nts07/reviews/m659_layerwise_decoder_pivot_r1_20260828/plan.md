# M659｜H67 layer-static heterogeneous decoder pivot

Date: 2026-08-28  
Status: **CONDITIONAL PLAN ONLY — producer-sealed M649 is still under M658 independent review**

## 1. Fail-closed verdict

The producer-sealed M649 canonical result reports
`PASS_NUMERIC_AUDIT__NO_GO_EXACT_TYPED_SPLIT`.  Its member and outer seals
verify locally, but M658 has not yet admitted the result.  Every numeric fact
below is therefore **conditional on a fresh M658 verdict with P0=0 and P1=0**.
If M658 does not pass, M659 authorizes nothing.

Conditional on that review, M654's channel-static hypothesis is disproved:
D1 is not a two-channel analog prefix followed by a binary suffix.  M659 does
not relax M649's gate and does not threshold, coerce, round, permute or select
values.  It pivots to a **module-static** boundary:

- D0, D2 and D3 are binary-module candidates because every audited raw
  `ConvTranspose2d` input element is exactly 0 or 1 across the frozen S10;
- all 770 channels of D1 form one common full-precision dense fallback in
  B0/B1/K1x8/K8/Ours, even though 81.508846% of the audited D1 values are
  exactly zero;
- route selection is a compile-time module ID.  There is no online value
  classifier, threshold, zero elision or suffix detector for D1.

This is a plan, not a decoder cycle result, speedup, RTL or paper headline.

## 2. Conditional M649 facts

The following aggregation is recomputed directly from all 40 M649 records.
`one` means exact float equality to 1.0; `nonbinary` means finite and neither
exactly 0.0 nor exactly 1.0.

| Module | S10 input shape per sample | Elements, S10 | Exact zero | Exact one | Finite nonbinary | Static route |
|---|---:|---:|---:|---:|---:|---|
| D0 | `10x1x1536x15x20` | 46,080,000 | 37,783,828 | 8,296,172 | 0 | binary candidate |
| D1 | `10x1x770x30x40` | 92,400,000 | 75,314,174 | 0 | 17,085,826 | common FP32 dense fallback |
| D2 | `10x1x386x60x80` | 185,280,000 | 153,544,434 | 31,735,566 | 0 | binary candidate |
| D3 | `10x1x194x120x160` | 372,480,000 | 267,646,872 | 104,833,128 | 0 | binary candidate |

Thus D0/D2/D3 observed exact-one rates are 18.003845%, 17.128436% and
28.144633%.  D1 is a sparse-analog diagnosis: exact-zero/nonbinary rates are
81.508846%/18.491154%, with zero nonfinite values.  **The D1 zero rate is not
used for execution savings in this plan.**  Every D1 element, including zero,
pays the same fallback service in every configuration.

### Why D1 can differ across the entire 770-channel tensor

The frozen topology explains why source-level channel reasoning was the wrong
boundary:

1. the network first concatenates the previous two-channel prediction and
   feature tensors;
2. the frozen `MS_SpikingTransposeDecoderLayer.forward` then applies `sn(x)`
   **before** `deconv(x)`;
3. M649 hooks the input of `deconv.0`, so it observes the 770-channel output of
   D1's neuron, not the pre-neuron concat;
4. the installed official-ATLIF surrogate returns `out * thre`, i.e. semantic
   binary activity encoded as `{0, learned_threshold}`, not necessarily the
   literal float set `{0,1}`.

For the audited first/last diagnostic channels, every S10 finite maximum is
`0.9999954104423523`, consistent with a D1 threshold that is very close to but
not exactly one.  This explains the observed pattern, but it is not permission
to replace that value by one: M659 deliberately treats the whole D1 module as
full precision.  D0/D2/D3 are routed from their measured exact values, not from
an assumption that every ATLIF threshold must be one.

## 3. Layer-static architecture

The candidate has exactly two routes selected from a frozen four-entry module
table:

```text
D0 -> BINARY_EXACT
D1 -> FP32_DENSE_FALLBACK
D2 -> BINARY_EXACT
D3 -> BINARY_EXACT
```

### BINARY_EXACT: D0/D2/D3 only

1. consume exact little-bit-first bitpacks; zeros are omitted only in the
   configurations whose frozen definition permits exact zero skipping;
2. use the common K3/S2/P1/output-padding1 polyphase map
   `dst = 2*src - 1 + kernel`; no configuration materializes inserted zeros;
3. use the same M514-style legal 4/6/9-tap mapping and M523-style deterministic
   boundary flush in every configuration;
4. K1x8 and K8 consume the same nonzero-source/tap/weight multiset and expose
   the same eight-bank peak service;
5. Ours alone may apply strict-subset parent/residual product capture within a
   frozen `(module,timestep,tap,output-block,input-partition,row-tile,weight)`
   context, with 1RW parent scratch, refcounts, directory traffic and dead
   writes charged;
6. every child mask must reconstruct as `parent XOR residual` with zero mask
   mismatch.  Parent edges cross no module, time, tap, weight or precision
   boundary.

"Exact" at the mask/product-selection level is not yet an exact checkpoint
numeric bridge.  Before any cycle row is admitted, a successor must either:

- preserve the frozen FP32 weight and contributor order and obtain identical
  streamed output hashes, including all floating-point reassociation effects;
  or
- freeze a signed-integer decoder deployment identity, prove its
  checkpoint-to-deployment accuracy separately, prove all widths/overflow, and
  obtain zero mismatch relative to that integer reference.

Failure of both options is `NO_GO`; it is not converted to an unlabelled lossy
result.

### FP32_DENSE_FALLBACK: all of D1

D1 is one opaque, ordered, full-precision transposed-convolution service.  It
executes all 770 channels and every legal tap for T10.  The same FP32 input,
weight, accumulation order, latency model, traffic, area and energy are charged
to B0, B1, K1x8, K8 and Ours.  It has no value classifier and does not exploit
the measured 81.508846% zeros.  Downstream D2 is fed only after the common D1
fallback completion boundary.

The M512 phase-balanced EPD scheduler remains killed.  Polyphase mapping is
common infrastructure, not a new speedup claim.

## 4. Structural arithmetic and ceilings

For K3/S2/P1/output-padding1, exact valid spatial taps are
`(3H-1)(3W-1)`.  Dense products per frame are therefore:

| Module | Valid-tap dense products/frame | Fraction of four deconvs | Fraction of corrected envelope if decoder is 21.57--22.83% |
|---|---:|---:|---:|
| D0 | 15,311,831,040 | 19.419303% | 4.1887--4.4334% |
| D1 fallback | 15,657,734,400 | 19.857997% | 4.2834--4.5336% |
| D2 | 15,852,927,360 | 20.105551% | 4.3368--4.5901% |
| D3 | 32,026,016,640 | 40.617149% | 8.7611--9.2729% |
| Total | 78,848,509,440 | 100% | 21.57--22.83% |

D0+D2+D3 are 80.142003% **binary-path eligible dense arithmetic**.  This is
not an 80.142% skip rate, product reduction, cycle reduction or speedup.  The
common D1 floor is 19.857997% of dense decoder arithmetic.

Under the deliberately optimistic proportional-dense-work assumption, making
D0/D2/D3 free while retaining D1 gives only a 5.035755x decoder ceiling.  Using
M510's old corrected-envelope decoder-share range gives a 1.20899--1.22394x
overall sensitivity ceiling.  These are structural upper limits; memory,
fallback, completion and parent-port costs can only reduce them.

## 5. Frozen five-row denominator

All five rows use the same S10 population and call order, module-static route,
K3/S2 polyphase map, 28-nm/3.0-ns label, 96 product lanes, physical eight-bank
source-service peak, 240-KiB total on-chip SRAM cap, 64-GB/s decimal DRAM cap
(192 B/cycle), queue/bank/port tuple, weight identity, output commit and D1 FP32
fallback.  Added area, state, ports, traffic and energy are always charged.

| Row | D0/D2/D3 binary-module execution | D1 execution | Interpretation |
|---|---|---|---|
| B0 / Fixed | dense real-source T10 issue, including exact zeros; common polyphase, no inserted-zero strawman | common dense FP32 fallback | cumulative dense denominator |
| B1 / structured | frozen eight-source group: if any bit is one, issue the whole group; skip only all-zero groups | common dense FP32 fallback | project-defined structured baseline, never call official PTB |
| K1x8 | exact nonzero sources through eight replicated scalar services | common dense FP32 fallback | strongest equal-service exact baseline |
| K8 | the exact K1x8 source multiset through shared K8 state/control | common dense FP32 fallback | isolates shared-state organization |
| Ours | K8 plus exact strict-subset parent/residual capture and charged 1RW scratch | common dense FP32 fallback | candidate |

The D1 fallback service is a common parameterized physical coordinate, not a
free black box.  The CPU DSE must sweep the same FP32 service throughput and
buffering point for every row, then nominate at most one physically realizable
coordinate for later PPA.  A row-specific fallback width or overlap is a
fairness failure.  B0/B1/K1x8/K8/Ours must bind one common-resource manifest
hash at each swept coordinate.

### Legal double-denominator presentation

- `C_B0 / C_Ours` is the cumulative decoder result versus Fixed.  It includes
  exact activation-zero skipping, K8 organization and parent capture; it may be
  prominent only after direct simulation and only when K1x8 is on the same
  page.
- `C_K1x8 / C_Ours` is the primary fair mechanism denominator.  It holds the
  exact nonzero multiset and peak source/weight service fixed.
- `C_K1x8 / C_K8` isolates shared-state K8 organization; `C_K8 / C_Ours`
  isolates parent capture.  These direct reruns are an ablation and are never
  multiplied into a headline.
- a binary-only slice may explain utilization, but the headline decoder ratio
  includes the common D1 floor.  D1's 81.508846% zeros cannot be credited to
  any row under this plan.

Suggested paper table columns are: configuration, direct decoder cycle/S10,
speedup vs B0, speedup vs K1x8, direct incremental ratio, FP32-fallback cycles,
weight/psum/parent/DRAM bytes, macro-rounded SRAM, area and energy.  Until
direct cycles and physical costs exist, every speedup cell stays blank.

## 6. Minimum successor payload

M649 intentionally saved no raw activation.  The smallest new capture that can
drive the CPU gate is:

1. **30 binary input bitpacks:** S10 x {D0,D2,D3}, C-order and little-bit-first,
   with exact sample/module/call order.  Their aggregate sizes are 5,760,000 B,
   23,160,000 B and 46,560,000 B, total **75,480,000 B**.
2. **D1 boundary only, no 369.6-MB raw input payload:** for each S10 call, stream
   a canonical input-content hash and ConvTranspose-output-content hash, plus
   shape, dtype, stride, exact-zero/nonbinary/nonfinite counts and the frozen
   FP32 fallback identity.  D1 raw values must not be used to classify work.
3. **Binary-module output hashes:** stream canonical D0/D2/D3 ConvTranspose
   output hashes so the later numeric model can fail closed on any FP32 or
   integer-reference mismatch without retaining raw outputs.
4. **Four weight identities:** preserve M649's shapes/content hashes; separately
   seal the exact payload used by the numeric path.  FP32 and any integer
   deployment package are distinct identities and must never share a result
   row.
5. **Global manifest:** checkpoint/config/source hashes, 10 sample IDs, 40-call
   order, tensor layouts, bit order, per-member size/hash, module parameters,
   no-running-BN protocol, completion marker, member seal and outer seal.

The capture must not modify, revive or reuse the consumed M511 one-shot or its
failed staging.  A fresh static hammer and fresh result hammer are required.

## 7. One-day CPU decision gate

No RTL is written first.  The CPU job is permitted only after (a) M658 admits
M649 with P0=0/P1=0 and (b) the successor payload receives an independent
P0=0/P1=0 result review.

The deterministic simulator must execute all four modules in order and all ten
timesteps, charge source scan, mapping/bundling, weight SRAM/DRAM, FP32 D1,
optional parent 1RW, Acc/psum, output commit and queue tails, and emit exact
conservation plus streamed numeric hashes.  It reports ratio-of-summed cycles,
arithmetic/geometric means, min/max per sample, traffic and every stall class.

Decision, always relative to total decoder cycles including D1:

- `GO_DECODER_RTL`: zero numeric/conservation mismatch, no overflow, all rows
  bind the same resource coordinate, macro-rounded total SRAM <=245,760 B,
  Ours/K1x8 ratio-of-sums >=1.15x and minimum sample >=1.05x.
- `MAIN_LOCAL_TABLE`: all GO conditions plus Ours/K1x8 >=1.20x, minimum sample
  >=1.10x and measured DRAM bytes at least 20% below K1x8.
- `SUPPORT_ONLY`: exact/fair/capacity checks pass and either total-cycle gain is
  1.05--1.15x or a named traffic axis falls at least 30%; report only the axis
  that passed.
- `NO_GO_RTL`: any exactness/fairness/capacity failure, or total-cycle gain
  below 1.05x with DRAM reduction below 20%.

The B0 ratio may be larger and is reported, but it never substitutes for the
K1x8 gate.  CPU results authorize no system headline, energy or PPA; they only
decide whether a small decoder endpoint is worth Synopsys implementation.

## 8. Paper-safe role

If the gate passes, this is not a fourth unrelated novelty and not a claim that
polyphase or product sparsity was invented here.  It is a C2 endpoint:

> A frozen layer table exposes an exact binary/analog boundary across the H67
> optical-flow decoder.  Three exact-binary transposed convolutions reuse the
> signed-source K8/polyphase fabric and finite-1RW product capture, while the
> exceptional 770-channel analog layer remains a fully charged, checkpoint-
> faithful fallback shared by every baseline.

The paper cites transposed-convolution decomposition, Prosperity for product
sparsity and FireFly-T/ELSA-style multi-source service.  The contribution is
the measured H67 layer boundary, common fallback discipline and finite-resource
integration.  It may enter a local mechanism table only after the CPU and
physical gates pass; the system table still requires a decoder-complete unified
run.

## 9. Claim boundary

M659 ran no GPU, EDA, VCS, DC, Formality, PTPX, DRAMsim3 or performance
simulator.  It did not modify `docs/359`.  It contains no admitted decoder
cycle, speedup, traffic reduction, energy, PPA, system result or DATE headline.
`docs/359` remains at SHA256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
