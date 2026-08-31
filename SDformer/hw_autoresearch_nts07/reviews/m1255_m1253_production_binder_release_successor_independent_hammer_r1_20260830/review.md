# M1255 independent hammer of the M1253 one-shot binder successor

Verdict: **BLOCK**. M1253 is not authorized for production execution, even after all four strict-valid825 artifacts exist. The source closes M1251's sealed-child, candidate-pair, minimum-AEE, selected-projection, exact-claim-boundary-object, O_EXCL and no-retry controls, but four receipt-closure gaps remain.

The exact M1253 author suite passed 13/13, the exact M1248 predecessor regression passed 10/10, and the independent hammer passed 11/11. Independent tests 03-07 are adversarial acceptance probes: they deliberately demonstrate conditions that should fail but currently pass.

## Confirmed controls

- Exact M1241/M1234/M1228 bytes are descriptor-read, copied to three memfds, and sealed with WRITE/GROW/SHRINK/SEAL before the attempt. The child receives exactly those three descriptors through `pass_fds`.
- Eleven candidate inputs are snapshotted before attempt consumption.
- Candidate identity/epoch order is exact: legacy ep29, then resume ep30/32/34. Minimum finite nonnegative AEE and lowest-epoch tie break are recomputed; the selected row must be the exact winning projection.
- The claim-boundary object itself is exact. Attempt O_EXCL, consume-before-child, one child, no retry, and failure preservation all hold.

## Blocking findings

1. **Mode is captured but not rebound.** `FileSnapshot.mode` exists, yet `receipt_identity()` omits it and the sealed launcher adds only device/inode. A post-prepare chmod is accepted by `verify_receipt`. This directly contradicts the contract's seven-field identity claim.
2. **The final result root is open.** A resealed root-level `paper_metric=false` and `hardware_speedup=true` both pass because only the nested claim-boundary map is exact.
3. **Nested identity maps are open.** `_exact_identity` checks required values but not exact keys. A resealed extra claim in a candidate checkpoint identity passes.
4. **E0-E8 is self-consistent but not authority-bound.** Replacing both the result field and sidecar with an arbitrary one-row payload passes. The exact nine rows from sealed M1228 must be pinned.

Required successor: include mode in both frozen-child publication and pre-attempt comparison; close the final result, identity/profile/activity schemas; pin the exact E0-E8 list; retain the current sealed-memfd, exact selection and one-shot controls; then obtain a new different-author hammer. Production remains additionally gated on all four real strict-valid825 artifacts and fresh output/attempt/log namespaces.

No production binder, remote host, GPU, valid825, hardware replay, VCS or Synopsys job was run. No M1253/M1248 source was edited. `docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
