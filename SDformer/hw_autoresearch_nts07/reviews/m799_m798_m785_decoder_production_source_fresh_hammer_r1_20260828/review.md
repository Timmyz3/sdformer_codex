# M802 orchestration / pinned M799｜M798 decoder production source fresh hammer

## Verdict

**PASS 100/100** — `PASS100_M798_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY`.

M798 closes the five M795 source defects: D1 remains charged but is excluded from the only headline ratio, canonical publication is atomic no-replace, future release validation binds the exact reviewed candidate/driver/runner, JSON duplicate keys fail closed, and the M699 outer-seal identity is a complete 64-hex SHA.

The orchestration milestone is M802, while the output directory intentionally uses the source-pinned `m799_m798...` identity. This preserves the executable future-release binding and is not an identity mismatch.

## Independent evidence

- Recomputed candidate, driver, runner and tests member SHA, sidecars and outer seals; all match their frozen source identities.
- `py_compile` passes for driver/tests, `bash -n` passes for the runner, synthetic self-test passes, and all **6/6** author unit tests pass.
- Independent duplicate-key attacks against `launch_now`, `release`, `candidate_binding`, and nested `canonical.result` are all rejected by the strict loader.
- Production aggregation retains D1 in `total_cycles`; `module_index == 1` is excluded from `headline_total_cycles`; `headline_ratio()` reads only that headline field. Three arbitrary D1 total-cycle perturbations leave the ratio invariant.
- Full publication API attack confirms `renameat2(RENAME_NOREPLACE)` rejects a post-precheck destination collision, leaves stage as a sibling, and creates no nested `result/stage`. The success path removes stage and leaves exactly the root four-member set: `result.json`, `detailed_rows.json`, `SHA256SUMS`, and `SHA256SUMS.seal.sha256`.
- Future release validation requires exact candidate SHA, candidate-identical source identity, exact reviewed candidate/driver/runner SHA, fresh-hammer review SHA plus double seal, and caller-supplied exact release SHA. A temporary synthetic exact release passes; independent wrong-candidate-SHA, wrong-reviewed-driver-SHA, and wrong-review-SHA mutations are all rejected.
- Missing-release runner attack exits `3` before one-shot consumption. Release, attempt, and production result remain absent.
- Frozen tuple remains `96 lanes / 245760 B / Acc24 / 3 ns / 192 B/cycle`, separate `40+120` populations, `T10`, and three configurations. The only legal named comparison remains typed signed K8 versus equal-service K1x8.
- M699 outer-seal file SHA is `eaf975a9a1a4829b2c0a2251e7ef297abd53b83b30e23630e5ce51db5c5de18c`.
- `docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Severity and boundary

- P0: 0
- P1: 0
- P2: 0

This is a source gate only. It authorizes drafting a separate exact-SHA true release, not production launch. No replay, cycles, speedup, decoder-complete claim, full-network completion, Table-A insertion, RTL/VCS/EDA/GPU/remote action, or `docs/359` modification is admitted.

The next legal step is: author the true release against this sealed review, obtain a fresh final launch hammer, then—and only then—consume the one-shot.
