# M509 r2 FC2-only INT8 PTQ exporter independent static hammer

## Verdict

`STATIC_GO__ONE_SHOT_EXPORT_ONLY_UNDER_EXACT_RUNNER_RESOURCE_GATE`

Score: **94/100**. No P0 correctness blocker remains in the reviewed exporter/contract pair. This is authorization for one exact export only after the separate exact-SHA runner passes every resource and process gate. It is not an accuracy, hardware, PPA, cycle, energy, system-speedup, or DATE-headline admission.

Reviewed identities:

- exporter SHA256: `755bf1111d986de387714356092b3d25c6c3029f83fe738a294edbbaca0739ec`
- r2 contract SHA256: `133fad77621e7c3c3feacc6c2ce1dd1e740420f7ef81a706b7645d18426a8c8c`
- frozen docs/359 SHA256: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

The review was read-only with respect to the candidate. It did not load the 591,167,876-byte checkpoint and did not execute the exporter, GPU, VCS, DC, Formality, PrimeTime, or any DSE.

## P0 closure

1. **Old full-model pickle compatibility is repaired.** The exporter installs the same baseline/overlay namespaces and Shiftmax compatibility registration used by the established M32 full-model loader before `torch.load`. It then accepts an `nn.Module.state_dict()` source.
2. **The output really contains the modified state.** The loaded state is cloned into an independent CPU `OrderedDict`; each of the twelve target entries in that clone is replaced by the dequantized tensor; `output_payload` is then constructed as exactly `{"model_state_dict": state}` and saved. The executable full-model object is not re-pickled.
3. **M51 mapping is fail-closed.** The exporter requires 120 records, twelve module names, exact sample IDs 0..9, 120 unique module/sample pairs, null bias, matching shape/dtype/byte count, raw little-endian weight SHA, and trace input/output dimensions matching the loaded FC2 tensor.
4. **M160 is checked per module.** All twelve module-specific `fc2_int8_sumabs_max` values must match, followed by the aggregate signed-width census `15b x2, 16b x2, 17b x6, 18b x2`.
5. **Non-target state is isolated.** Source and output state key sets must agree and every non-target state entry is checked against its cloned source value before serialization.
6. **Contract and publication are bound.** The invoked contract must have the r2 schema/status, pin the running exporter, and resolve to the canonical output directory. Its observed SHA is embedded in the output manifest. Staged and final members plus the outer seal are both rehashed, and overwrite is refused.

## P1 findings and runner obligations

1. The exporter dynamically records the invoked contract SHA but cannot self-encode the independently approved contract hash. The exact runner must pin the reviewed r2 contract SHA above before invocation.
2. Full-model unpickling depends on repository class definitions. The exact runner should pin at least the profile loader, ATLIF source/installer, baseline spiking modules/Swin source, and BSA compatibility source. The currently inspected key sources match the already frozen M160 identities where available.
3. The serialized checkpoint is not reopened for a semantic post-save miter. Construction is direct and the final bytes are sealed, so this is not a static correctness blocker; the production-result hammer should reopen the passive checkpoint and independently verify its single-member container, all target tensors, all non-target tensors, NPZ codes/scales, and hashes.
4. Frozen inputs are hashed before use but not rehashed at the end. The exact runner should bind immutable input SHAs and the result hammer should rehash them after completion.
5. The canonical parent `system_handoff/outgoing/` was absent during this review. The exact runner may create that specific parent before invocation, but must still require the canonical final directory to be absent.

## One-shot conditions

- Pin the exact exporter, r2 contract, docs/359, frozen input, and pickle-compatibility source SHAs.
- Require canonical output only and consume one attempt immediately before Python launch.
- Pass the contract's three resource samples: no VCS/DC/Formality/PrimeTime/CPU-DSE conflict, MemAvailable/commit headroom/SwapFree each at least 8 GiB, no cgroup failure or OOM delta, and at least 2 GiB free disk.
- Do not admit the result until an independent receipt-blind post-export hammer verifies the final checkpoint and NPZ.
- Even after a successful export, keep valid825 accuracy, BN2 integer bridge, RTL, cycle, power, PPA, full-FFN, full-network, and headline claims false.

