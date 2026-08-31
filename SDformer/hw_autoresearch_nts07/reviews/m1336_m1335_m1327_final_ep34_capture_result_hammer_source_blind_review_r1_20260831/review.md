# M1336 — M1335 final ep34 result-hammer source blind review

## Verdict

`FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED`

M1335 correctly closes the five M1334 findings. Under the pinned Python 3.10.18
and NumPy 2.1.2 runtime, its 18 new tests and 13 inherited tests pass. The exact
M1334 failure authority, author double seal, `lexists`/`lstat` canonical gate,
record/seal/actual SHA equality, ordered 79-operator and 93-ATLIF identities,
attention dtype/geometry/statistics, and all older ep34/cohort/admission checks
were reproduced.

It is nevertheless unsafe to run on the eventual canonical capture. Nine
independently mutated, recursively resealed fixtures are incorrectly accepted,
forming six false-negative groups. Four are P0:

1. Raw FP32 size and dtype are unbound. Three raw bytes can represent a declared
   four-byte scalar, and a `.fp32.zlib` payload can be labelled `torch.float16`.
2. Raw values are not reconciled with input statistics. A +1.0 value passes with
   active/positive/negative/nonfinite all zero.
3. Positive/negative plane sizes are self-reported, not derived from
   `ceil(elements/8)`. Even the author positive fixture assigns one-byte planes
   to decoder tensors containing millions of elements.
4. Plane contents are not compared with raw signs and padding is not required to
   be zero. A positive value encoded as negative and nonzero tail bits pass.

Two P1 gaps remain: `zlib.decompress` accepts trailing bytes after the valid
stream, and attention archives accept invented members plus nonzero q/k padding
outside the declared `[2,W,H,N,D]` extent.

The minimum successor must derive byte/plane extents from input elements, decode
little-endian FP32 and recompute statistics/sign planes, enforce one complete
zlib stream with no trailing data, require exact NPZ keys and zero pack tails,
retain every current gate, add all nine attacks as regression tests, and undergo
another different-author recursive-seal review.

The real canonical path was checked only for lexical absence and was never read
or created. No remote, GPU, capture, replay, or EDA action ran. `docs/359`
remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
