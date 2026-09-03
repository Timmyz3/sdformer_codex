# M2065 / M2064 ep34 G>48 FC2 exact-continuation quick-gate result hammer

## Verdict

**PASS, 98/100; P0/P1/P2 = 0/0/2.** M2064 is a sound CPU/source
quick gate and may advance to **one separately reviewed VCS-source design
stage only**. This review does not authorize a VCS launch, EDA, GPU work,
paper admission, or any system-speedup claim.

The independent hammer did not import the M2064 analyzer. It reopened the
M1707 capture and all 1,925 members of the M2057 result, recovered the 3,840
M2057 calibration observations, decoded the eight target FC2 layers, and
independently rebuilt chunking, cache order, cycle charges, directed INT8
arithmetic, Acc24 bounds, and the 2,880-workload combined aggregate.

The main result is exactly reproducible:

| Scope | Ordinary model cycles | TSBG model cycles | Nominal ratio | Pessimistic ratio |
|---|---:|---:|---:|---:|
| New eight G>48 FC2 layers, 960 workloads | 74,746,384 | 39,992,560 | **1.869007235x** | **1.864194493x** |
| Existing 1,920 G<=48 FC workloads under the same full-FC fees | 230,690,920 | 96,958,930 | 2.379264293x | 2.375295770x |
| Combined 24 FC layers, 2,880 workloads | 305,437,304 | 136,951,490 | **2.230259079x** | 2.225990783x |

All these values remain a VCS-calibrated CPU/source model. They are not new RTL
cycles, full-FC wall time, a network result, FPS, energy, or system speedup.

## Seal and identity audit

- The M2064 contract and its two sidecar seals verify. The contract pins the
  exact analyzer SHA `58c0589178b23ab31826a0dd9e329bab977333829cefb8518553629a18af4161`.
- The M2064 result has exactly two inner members (`result.json` and
  `summary.json`), no extras or symlinks, and a valid outer seal.
- M1707's capture manifest and outer seal verify. All eight capture members
  were rehashed; the consumed `fc_frames.bin`, `layers.json`, and
  `sample_order.json` match the frozen M2051 decoder pins.
- The M2051 fixture metadata currently has SHA
  `3ac7048f0a97aeea0ac91627d303f4eea06b8a48bab816468825acfee180ccc5`,
  exactly the fixture identity stored in M2057.
- All 1,925 M2057 manifest members were rehashed. The result identity and
  1,920 unique PASS logs remain intact.
- `docs/359` stayed at
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`
  before and after the review.

## M2057 calibration

The independent hammer reparsed both ordinary and TSBG observations from all
1,920 M2057 logs using the frozen equation

`27 + (7/6)*issues + (21/2)*weight_bundle_beats`.

The 3,840 residuals (`source_model - VCS_execute_cycles`) reproduce exactly:

| Residual | -5 | -4 | -3 | -2 | -1 | 0 | +1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Count | 3 | 8 | 57 | 413 | 1,297 | 1,940 | 122 |

The absolute maximum is five cycles. M2064's pessimistic interval subtracts
five cycles from ordinary and adds five to TSBG for every output-tile/chunk,
which is conservative relative to this observed calibration envelope.

## Geometry and global weight addressing

The fixed workload cardinality is exactly `40 samples * 8 FC2 layers * 3
quartets = 960`. The cohort contains four sequences with ten samples each.
The layer geometry is six G96 layers and two G192 layers:

- every G96 workload uses global group bases `{0, 48}`;
- every G192 workload uses global group bases `{0, 48, 96, 144}`;
- every physical chunk contains at most 48 source groups.

The correct continuation accumulation indexes directed weights with the global
group slice. To test that this was not accidentally masked by the data, the
hammer deliberately remapped every chunk after the first to weight groups
`0..47`. This alias mutation changes 1,029,923 output values across 743 of 960
workloads and is rejected. Therefore the observed zero-mismatch result is
sensitive to preserving `global_group_base`; it is not an address-insensitive
fixture.

## Exactness and Acc24

- Independently checked integer outputs: **1,843,200**.
- Correct full-group versus chunk-continuation mismatches: **0**.
- Acc24 overflow observations: **0**.
- Maximum observed intermediate/final absolute accumulator: **5,652 / 5,652**.
- A simple worst-case bound for G192 is
  `192 groups * 16 sources * 128 = 393,216`, still far below signed Acc24.

The test uses deterministic directed INT8 weights. It does not claim captured
hardware weights or task accuracy.

## Fair cycle charging

Every output tile pays the same fixed charges on ordinary and TSBG axes:

- **384 cycles per chunk** for descriptor preload;
- **2 cycles per intermediate chunk** for continuation;
- **one final 27-cycle retire**, only after the last chunk.

The per-chunk 27-cycle term in the M2057-calibrated service equation is removed
before summing data-dependent service, then one final retire is added. Thus
there is neither a free TSBG continuation nor a duplicated per-chunk retire.
Independent comparison of all result-row fields finds zero row or fee
mismatches. All 960 workloads are non-regressive under the nominal model.

The combined 2,880 result was also reconstructed correctly: existing and new
ordinary cycles are added, existing and new TSBG cycles are added, and only
then is the ratio formed. No component ratios are averaged or multiplied.

## Severity findings

### P0: 0

No broken seal, input/checkpoint drift, arithmetic mismatch, overflow, missing
workload, global-group alias, unfair fixed charge, aggregate inflation, paper
promotion, or system-speedup claim was found.

### P1: 0

The result, model boundary, and VCS-source decision are internally consistent.
The following two provenance/observability weaknesses do not change the current
number and are classified P2.

### P2: 2

1. M2064 reads the M2051 fixture metadata but does not directly pin its SHA in
   its own contract or source. The current file exactly matches the identity
   already pinned by M2057, so this is not present-result drift. The next VCS
   source contract must directly pin
   `3ac7048f0a97aeea0ac91627d303f4eea06b8a48bab816468825acfee180ccc5`.
2. In M2064 row construction, the detailed per-chunk list is overwritten by
   the integer chunk count when `**cycles` is merged. The source executes the
   correct bases and the independent alias attack proves sensitivity, but the
   canonical row does not expose each `global_group_base`. The VCS fixture or
   source receipt must serialize `{0,48}` / `{0,48,96,144}` explicitly.

## Authorized next stage

One new VCS **source and contract** may be designed, subject to a fresh
different-author source review. It must:

1. pin M1707, M2051 metadata, M2057, M2018/M803, the generated fixture, and the
   new RTL/testbench by exact SHA;
2. expose global group base, first/intermediate/final chunk, retained Acc24
   context, and final-only terminal/commit explicitly;
3. charge identical preload, continuation, cache, output-tile, and retire work
   on both axes;
4. include G96 and G192 positive coverage plus an address-alias attack;
5. require zero oracle mismatch/overflow and exact commit cardinality; and
6. keep the CPU ratios out of the RTL result and paper until a new VCS result
   and independent result hammer exist.

This review does **not** authorize running VCS, any EDA tool, GPU work,
automatic retry, or editing the current ISCAS paper. It admits only the
transition from CPU quick gate to a separately reviewed VCS-source stage.

## Reviewer execution note

The first run of this review-only hammer exited 1 because the reviewer compared
the per-workload intermediate peak against the final peak instead of retaining
the maximum over chunks. That reviewer bug was identified and announced before
the rerun, fixed only in `independent_hammer.py`, and never touched M2064 source,
contract, input, or result. The corrected full replay exited 0 and produced the
values above.

No EDA, VCS/simv, GPU, license query, network experiment, paper edit, source
edit, contract edit, result edit, predecessor edit, or `docs/359` edit was
performed.

