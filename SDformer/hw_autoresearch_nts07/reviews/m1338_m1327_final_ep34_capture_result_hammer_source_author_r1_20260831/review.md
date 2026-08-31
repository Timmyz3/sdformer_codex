# M1338 — additive final ep34 result-hammer source author review

M1338 is a source-only additive successor to rejected M1335. It binds the exact
M1336 failure review and preserves the complete M1335/M1333 identity, runtime,
symlink, SHA, ordered operator/ATLIF, attention geometry, checkpoint, cohort,
ep34 and admission gates.

The retained payload boundary is now content-derived. Every retained input must
be exact `torch.float32`, declare `bytes == elements * 4`, decode as one complete
little-endian FP32 zlib stream with no trailing data, and reproduce the recorded
active/positive/negative/nonfinite counts. Each support plane must be exactly
`ceil(elements/8)` bytes, have zero tail padding and no positive/negative
overlap, and match the raw FP32 sign predicates element by element. Attention
NPZ members are exact rather than subset-matched, and unused Q/K packbits must
be zero.

The 12 new directed tests pass, including positive NaN/Inf semantics and all
nine M1336 accepted attacks now rejected. M1335 18/18 and M1333 13/13 remain
green. Source self-check passes under the pinned Python 3.10.18 and NumPy 2.1.2.

This is not production authorization. A fresh different-author blind mutation
hammer and recursive seal are still required before the eventual canonical
result may be read. The canonical path remained lexically absent and was never
read or created; no remote, GPU, capture, replay, VCS, or EDA action ran.
