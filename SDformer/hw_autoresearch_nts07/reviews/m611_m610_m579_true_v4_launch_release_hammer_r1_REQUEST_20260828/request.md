# M611 request｜M610 M579 production true-v4 contract + one-shot true release hammer

Perform a fresh independent **read-only true-launch hammer**. Do not call the runner with `--execute`; do not run the
formal 80-record CPU replay, GPU, EDA or remote; do not create result/attempt/consumed/quarantine/staging; do not modify
M610/M609/M605/M601 or `docs/359`.

## Frozen subjects

- production contract SHA `29a471dc489da4895e38b01700a4e101a5055bcbfd37323025a0762958011bb0`,
  outer-seal-file SHA `faf54c358b28967b33823ef53c95c03293e20790f8248e7ef268882c55299d79`
- true release SHA `b26bcb2ed9665e561ea84cad8038ff97f2406ac3b33be90538c88d4240c7c1f6`,
  outer-seal-file SHA `baa860bdcf6c9143348ff0f645a80b2ab893408f5ebec6ede5328645f32b5e52`
- M609 manifest SHA `fbdca56932deee9b966ec4e7846f999271e66cbd3484cf41969b04bbe4221b6d`,
  outer-seal-file SHA `37f90e75d09ad8988781fb2f979d5d522be2233e26636af79a1314b107602699`

## Mandatory review

1. Verify strict JSON, all member/outer seals and the full M601 -> M603 -> M605 -> M609 -> M610 SHA chain.
2. Verify contract schema is exact production v4 and authorization is one-shot: launch_now/run_cpu/max_attempts =
   true/true/1, workers=3, formal records=80, GPU/EDA/remote false. Verify release is `still_not_executed=true`.
3. It is allowed to run the analyzer directly with `--validate-contract-only`; require 15/15 inputs, 80/80 payloads,
   zero formal records and no result/attempt. **Never run runner `--execute`.**
4. Compare contract `.inputs` exactly with the M601/M605 frozen 15-input mapping. Recheck analyzer/runner/runtime,
   M43/M504/M505, chunk-major anchors, DMA/tail/commit/8 blocks and unique output coordinates.
5. Under lexists semantics verify result/attempt/consumed/quarantine/PID staging absent. Recheck same-parent trap,
   terminal rehash, double seal and `RENAME_NOREPLACE`; one consumed attempt must block a second run.
6. Recompute the three author resource samples against 48-GiB commit, 128-GiB MemAvailable, 32-GiB SwapFree,
   clean session/user cgroup and UID-local collision=0. Confirm honestly that the runner does not enforce these gates and
   root must repeat a fresh live check immediately before invocation.
7. Recheck complete accuracy disclosure: valid825 single seed +0.5730215096601543%, ten-frame 5/5, 64-frame PAFT
   regression 1.0189020311889285%, no Pareto. Recheck nine-row capacity 213,376 B and macro PPA/energy open.
8. Confirm no formal CPU/result/attempt was created by authoring and `docs/359` SHA is unchanged.

Give score/P0/P1/P2 and PASS/FAIL; PASS requires score>=95 and P0=P1=0. A PASS may recommend root perform a fresh live
resource/collision check, exact runner preflight, then exactly one frozen invocation. Raw result remains non-citable until a
fresh independent result hammer.

