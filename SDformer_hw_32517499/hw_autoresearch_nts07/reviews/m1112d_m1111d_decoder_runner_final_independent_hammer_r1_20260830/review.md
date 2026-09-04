# M1112D final independent hammer of the M1111D decoder runner

## Verdict

`STOP_M1112D_PUBLISH_GATE_ACCEPTS_FORBIDDEN_CLAIMS_AND_INCOMPLETE_FILESET`

Score: **72/100**. P0=1, P1=1, P2=0. Do not run M1111D production and do not create its canonical attempt. No external launch tuple or command is authorized by this review.

The exact runner SHA `52407204...`, contract SHA `82bba9ed...` and outer `7a94c5ce...`, author manifest `56989e64...` and outer `3e92ee45...`, M1105Dr2 source/contract/receipt chain, M1110D independent hammer, both mapper sources, Python 3.10.18 and `docs/359` all match. The three flat authorities each have six regular manifest members with exact file coverage.

The safe authority validation and synthetic six-kind schedule test pass. The runner `main()`, M1105Dr2 `build_canonical()`, `execute_production()`, and canonical payload were not called. Canonical attempt/result/lock/work/quarantine state is empty before and after this hammer.

## Attacks that close correctly

The exact source rejects extra argv, contract byte mutation, a source-file symlink, an authority-directory symlink, stale attempt/result/lock/work/quarantine namespaces, a broken result symlink, an unsealed work publication, and quarantine-name collision. Caller `PYTHONPATH`, checkpoint/M700-like variables and `LD_PRELOAD` are erased, leaving only the six fixed environment variables.

Static control flow has one call each to authority validation, environment sanitization, resource gate, lock, attempt consumption, production, publication, failure quarantine and lock release. Attempt consumption precedes canonical payload access; maximum attempts is one, automatic retry is false, and `main()` has no retry `while` or recursion.

The contract and honest generator pin:

- M700 input rejected;
- D1 exact scaled-binary θ word `1065353139` / little-endian `b3ff7f3f`, with no weight folding;
- final-checkpoint rebind required;
- `ratios_or_speedups = null` and all performance-admission fields false.

These generator facts do **not** close the publication P0 below.

## P0 — forbidden claims and incomplete output pass the publication validator

`publish_result()` verifies the atomic seal, then checks only:

1. the aggregate status token;
2. `system_speedup_admitted == false`;
3. `paper_ppa_ready == false`.

The independent dry attack created a temporary, atomically sealed work directory with only two payload members: the aggregate JSON and completion token. It deliberately omitted `m1111d_decoder_call_schedule.jsonl` and set:

- `diagnostic.ratios_or_speedups = {"forged_speedup": 999.0}`;
- `speedup_admitted = true`;
- `paper_citable_performance = true`;
- `final_checkpoint_rebind_required = false`;
- the two currently checked fields to false.

That payload passed every `publish_result()` validation and reached `rename_noreplace`. The hammer replaced `rename_noreplace` with a sentinel, so no temporary or canonical result was created. Reaching this boundary proves the gate would publish the forbidden/incomplete artifact.

This violates the contract's ratio ban, final-rebind boundary, exact three-file output and 120-call schedule requirements. It is a production-authorization P0 even though the current generator intends to write honest fields: the final no-replace gate is the admission boundary and must independently validate what it publishes.

## P1 — flat seal roots accept same-byte symlinks

The runner rejects a symlinked authority directory and manifest-listed symlink members, but `verify_flat()` does not require the root `SHA256SUMS` and `SHA256SUMS.seal.sha256` themselves to be regular files. A temporary copied M1110D directory whose two root seals were same-byte symlinks passed `validate_authorities()`.

This does not alter the currently frozen bytes, but it contradicts the regular-file authority contract and leaves a path-shape gap. The additive repair must apply `lstat`/regular/no-symlink checks to review, manifest and outer files and require exact manifest coverage.

## Minimal additive repair

Keep M1111D source, contract, author receipt and this STOP review frozen. Author a new M1111Dr2 namespace; do not edit or retry M1111D.

Before no-replace publication, M1111Dr2 must independently require:

1. exact payload files `{m1111d_decoder_result.json, m1111d_decoder_call_schedule.jsonl, RUN_COMPLETE.txt}` plus the atomic seal bundle, with no extras or symlinks;
2. exactly 120 strict-JSONL call rows, global ordinals 0–119, frozen sequence/sample/module order, and all required call fields;
3. recomputed JSONL file SHA and encoded-row stream digest equal to the aggregate receipt;
4. aggregate population `calls=120`, `timesteps_per_call=10`, positive transaction count, exact common-resource projection and exact checkpoint/source/contract identity;
5. `diagnostic.ratios_or_speedups is null`;
6. exact call-row and aggregate claim projections: diagnostic only; all speedup, system-speedup, paper-citable and PPA fields false; final checkpoint rebind true;
7. M700 false and D1 θ/no-fold identity at aggregate and D1-call rows;
8. regular no-symlink flat roots and exact manifest coverage.

A different-author hammer must mutate each field independently, omit/duplicate/reorder a call row, alter both digests, add an extra file, and repeat the same-byte seal-symlink attack. Only after all attacks reject may a new external tuple and unique command be issued.

## Claim boundary

This review admits only that the exact M1111D runner's authorities, safe static test, namespace and honest generator literals were inspected, and that its publication gate has a reproducible P0. It admits no production launch, transaction schedule, cycle, traffic, ratio, speedup, decoder completeness, Table-A row, energy, PPA or paper result.

`docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
