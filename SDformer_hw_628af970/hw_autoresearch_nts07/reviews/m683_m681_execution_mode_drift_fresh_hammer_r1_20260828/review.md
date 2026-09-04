# M683 — M681 execution-mode drift fresh hammer

## Verdict

**GO_SCOPED_CAUSAL_DIAGNOSTIC — 98/100, P0=0 / P1=0 / P2=1.**

For the frozen sample-0 decoder-D0 workload and recorded RTX3090,
PyTorch 2.7.1/CUDA 12.8/cuDNN 9.7 stack, `cudnn_allow_tf32=false` is the unique
effective recorded control separating the M511-compatible bitpack from the
M660-r4 failure bitpack.

This GO admits only that scoped execution-mode diagnosis. It does not authorize
a direct GPU one-shot, payload, accuracy conclusion, cycles, speedup, energy,
PPA, RTL, EDA, or paper headline.

## Controlled comparison

`legacy` and `deterministic_tf32` both reproduce the frozen M511 payload
bit-for-bit:

- SHA: `ad2251f1fb8a470651044456e0b7182bd6db0e0a89fb63018efa3a9e6fcd6447`
- active/zero: 839,586 / 3,768,414

`m660` reproduces the failed M660-r4 payload bit-for-bit:

- SHA: `10981fb3970dd6918c0f89645ce7cf4b3cfb73816cd2fa9ab0e9ea3dc4895d5d`
- active/zero: 838,404 / 3,769,596

Independent decoding gives 264,066 XOR bits (5.7306%) between branches.

`deterministic_tf32` and `m660` have identical deterministic algorithms, cuDNN
deterministic mode, benchmark mode, CUDA-matmul TF32 setting and CUBLAS workspace
configuration. Their only effective backend-control difference is cuDNN TF32:

| control | deterministic_tf32 | m660 |
|---|---:|---:|
| deterministic algorithms | true | true |
| cuDNN deterministic | true | true |
| cuDNN benchmark | false | false |
| CUDA matmul TF32 | false | false |
| cuDNN TF32 | **true** | **false** |
| CUBLAS workspace | `:4096:8` | `:4096:8` |

The `m660` assignment of CUDA-matmul TF32 to false is an effective no-op because
it is already false in `deterministic_tf32`.

## Post-configuration source audit

The mode snapshot is taken before M511/profile loading, so I audited the full
frozen inference path rather than assuming it remains valid:

- `load_config` only parses YAML, resolves dataset paths, and freezes loader and
  input-size values.
- `build_model` constructs/loads the model, configures SNN step/backend and calls
  `eval`; it does not write deterministic, cuDNN, or TF32 flags.
- BN evaluation changes only running-stat fields.
- `runtime_backend` selects the SpikingJelly CuPy/Torch backend only.
- A Python-source search of the frozen inference graph finds no later writer of
  the audited flags.

The YAML `runtime.allow_tf32=true` and `runtime.cudnn_benchmark=true` values are
declarations used by training entrypoints; this profile path parses but never
applies them. Therefore they do not override M681's explicit mode controls.

## Causal boundary

The evidence is sufficient to diagnose the branch change at sample-0/D0. It
does not establish behavior for the other 39 decoder calls, overall accuracy,
or which numerical mode should be preferred. It shows why an M649 ledger made
under the TF32-enabled branch cannot serve as an equality gate for the strict
M660 branch.

## P2-1 — launch receipt completeness

The three sealed JSONs record the decisive controls, CUBLAS setting, input/model
identities, runtime versions/device name and output hash, but not complete argv,
environment, driver/GPU UUID, timestamp, or repetition count. Exact matches to
both independent historical payloads make the scoped diagnosis sound, but the
omission blocks any broader causal or deployment claim.

Future execution-mode DSEs should add a complete sealed runtime receipt and a
post-build/pre-forward control snapshot. No additional GPU run is authorized by
this review.
