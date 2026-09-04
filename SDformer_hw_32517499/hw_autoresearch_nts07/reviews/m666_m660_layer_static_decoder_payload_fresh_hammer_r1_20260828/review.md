# M666｜M660 layer-static decoder payload fresh independent hammer

## Verdict

**NO_GO_P1__DO_NOT_EXECUTE_OR_CONSUME_ONE_SHOT**

Score: **68/100**. Severity: **P0=0, P1=4, P2=2**.

This is a CPU-only/static, independent hammer of the exact frozen M660 author
candidate.  It did not import or execute the author test module, use CUDA,
consume the M660 one-shot, run a performance simulator, launch RTL/EDA, or
modify an author/predecessor artifact.  Under the frozen request policy, any P1
requires NO_GO and forbids publishing a candidate command.

Frozen targets rehashed exactly:

| object | SHA256 |
|---|---|
| producer | `2e1ea26b5293ba1063e7be0056cebd2b25e09903bb528c31427c032df8b73acc` |
| runner | `ae9902b42331f3e88e94b11d9c5a5f6f3bdfc3e2b473939a7569af38f2396281` |
| contract | `38200ef4db5795d8be70e6e776aabf09dad10818344b972add535900a95f2cb4` |
| author tests | `0dc63c88349dec0ecc77d2fb4aa51f0df82316d1c435a73f1d760ae50fb54cc0` |
| author-handoff outer-seal file | `341db83d1c084b3ea6e41b155d4a24039b858fafa9a23ca45e7a3319f105f414` |

The M658, M659, M662 and author-handoff double seals independently verify.
The M660 canonical output and attempt-consumed directory remain absent.
`docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## P1 findings

### P1-1｜The threshold path is incompatible with the frozen H67 topology

The producer derives the owner and sets `neuron = named[owner_name].sn`, then
requires that object itself to have `thresh`, `threshold_mode`, and
`output_mode`.  The frozen model boundary is different:

- `MS_SpikingTransposeDecoderLayer` owns `self.sn = Spiking_neuron(...)` and
  executes `self.sn(x)` before `self.deconv(x)`;
- `Spiking_neuron` is a wrapper whose actual neuron is
  `self.spiking_neuron`;
- the H9 installer replaces/configures `wrapper.spiking_neuron` with the ATLIF
  implementation that owns `thresh`, `threshold_mode`, and `output_mode`.

An independent exact CPU model/checkpoint load observed:

```
OWNER_SN_CLASS Spiking_neuron
WRAPPER_HAS_THRESH False
INNER_HAS_THRESH True
```

The independent faithful-wrapper attack therefore reproduces:

```
RuntimeError: M660 D1 is not the frozen scalar official-ATLIF binary neuron
```

The author unit test masks this defect by constructing a toy `Owner.sn` that is
the inner threshold-owning neuron directly.  On the real H67 object graph, the
one-shot would be consumed at the runner before the producer reaches and fails
this check.

Required repair: dereference and freeze the exact
`owner.sn.spiking_neuron.thresh` path, record that full parameter path, and add
a topology-faithful test plus a CPU exact-load preflight before one-shot
consumption.

### P1-2｜The “bit-exact” miter is numerical equality, not bit equality

`compare_tensors_streaming()` increments mismatches with
`left_chunk != right_chunk`.  IEEE-754 `+0.0` and `-0.0` compare equal
numerically but have different bytes.  The independent attack obtains:

| field | result |
|---|---|
| reported `bit_exact` | `true` |
| reported mismatch count | `0` |
| `+0.0` SHA256 | `df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119` |
| `-0.0` SHA256 | `6d58692645c9d1cfaf13541cbd258f86193ef63c2f1d38f6bbca9617372d7bd6` |
| hashes equal | `false` |

The contract requires all mismatch counts to be zero **and hashes to match**.
The final `d1_folded_miter_bit_exact` reduction checks only each row's
`bit_exact` Boolean and never requires original/reference hash equality.
Consequently unequal output bytes can be labeled
`D1_FOLDED_WEIGHT_MITER_BIT_EXACT` and can admit the folded weight for
deployment.

Required repair: compare the raw FP32 bit patterns (for example an exact
`uint32` view or canonical bytes), count bit-pattern mismatches, classify signed
zero separately, and require per-call output SHA equality in the final
deployment-admission reduction.

### P1-3｜Transient threshold drift is invisible and the “snapshot” aliases storage

`decoder_threshold_identity()` returns `neuron.thresh.detach()` without a
clone.  This aliases the live parameter.  The producer checks identity once
before S10 and once after S10, but never before/after every sample or hook.
An independent mutate-to-0.5/restore-to-original attack shows both problems:

- the returned supposed snapshot changes when the live parameter changes;
- the beginning and ending bytes are identical even though an intermediate
  value was different.

Such a run can create masks using more than one runtime threshold while
publishing one static threshold identity and one folded weight.

Required repair: clone the scalar to immutable CPU bytes, derive all mask and
folded-weight operations from that frozen scalar, and compare the live
parameter's exact bytes against the clone before and after every sample (or at
every D1 hook).

### P1-4｜Failure staging can retain candidates forbidden by the negative route

The producer serializes `d1.weight.folded_theta.f32le` and the output-scale
sidecar before the first sample.  Passing early D1 samples also write candidate
bitpacks.  Normal all-S10 fallback later deletes these files, but the general
exception handler writes only `FAILED.json`; it does not scrub the folded
weight, sidecar, or earlier `d1_candidate` files.

Therefore an arbitrary non-`{0,theta}` sample followed by another hook/model
failure can leave a failed staging tree containing the very D1 candidates the
request requires the negative route to remove.  It is not a canonical publish,
but it violates the requested fail-closed cleanup boundary and makes the
failure receipt's statement stronger than the files it seals.

Required repair: keep candidates in memory until the complete S10/global gate
passes, or make every pre-publication exception recursively scrub all D1
candidate bitpacks, folded weight, and sidecar before sealing failure evidence.

## P2 findings

### P2-1｜Bit-exact reproducibility controls are not frozen

The candidate records package/GPU provenance but does not set or receipt any of
the following: deterministic algorithms, cuDNN deterministic/benchmark modes,
CUDA matmul TF32, cuDNN TF32, or a maximum-ULP diagnostic.  This does not replace
the P1 byte comparator fix, but it is required to make a repaired S10
bit-exact result reproducible and diagnosable.

### P2-2｜The contract predicts closure before the runtime evidence exists

`m658_p2_closed_by_m660_runtime_receipt` is `true` in the static contract even
though no M660 runtime receipt exists.  The runtime receipt is designed to
close the provenance issue, but the contract should say “required to close” and
only a post-result independent hammer should record closure.

## Attacks that passed

- Every contract input and frozen target hash matched; independent nested seal
  population checks passed for M658/M659/M662 and the author handoff.
- Parent traversal and symlink components were rejected by the producer helper.
- `take_exact(..., 10)` made exactly ten `next()` calls and did not request item
  eleven.
- Invalid D1 values at positions 0, 7, 8 and 15 across two chunks caused the
  local candidate writer to remove both final and `.partial` files.
- A two-byte independent bitpack was `0x26 0x11`, little-bit-first, with
  independently recomputed popcount five.
- The 30/40 record and 40-hook lattice reductions are explicit and unique in
  the candidate; this remains a static property until a real result exists.
- Raw D1 activations are not serialized by the fallback summarizer.
- The original-weight/output-scale sidecar remains statically marked
  `admitted: false` and is not promoted by the current final reduction.
- Fresh staging, atomic rename, post-publication quarantine, environment
  allowlisting, and nested top-level seal binding are otherwise well designed.

## Admission boundary and next action

`GO=false`.  No GPU, cycle, speedup, RTL, EDA, energy, PPA, system-speedup, or
DATE-headline claim is created by M666.  M660 must not be executed and the
one-shot must remain unconsumed.

The only authorized next action is to repair all P1s, issue new producer/test/
contract/runner hashes and a new double-sealed author handoff, then request a
fresh independent static hammer.  Only a future review with P0=0 and P1=0 may
publish a unique execution command.

