# R13 coverage recovery implementation

This new entry point separates coverage from update count. Every condition runs independent native, FF, CC and fixed block2 fine arms from the same original step135930 checkpoint and the same newly calibrated private BN buffers. Each arm uses 200 ordered training updates, identical FP32/AdamW/80-tensor/7,252,064-parameter settings inherited from R12. The `reference` order uses 40 identities five times; `diversity` uses 200 identities once, with the same sequence on each matched step.

The new manifest fixes recovery/BN/selector sequences separately from check sequences. The original backbone has nevertheless trained on all train sequences. This is not a backbone-unseen or official-test experiment. Region thresholds are newly fitted on the common diversity200 training identities for evaluation only, and do not enter the recovery loss. Thus both conditions share the same region diagnostic definition, while only their respective training orders update model parameters.

## Source reuse and execution

`inherited_pins.json` binds the immutable R12 runner, trainable wrapper, source contract, test and receipt. Mathematical forwards, original actual/dual references, parameter whitelist and gradient audit are imported directly. `execute_recovery.py` is the R12 execute loop with narrow changes: explicit 200-step order, duplicate-aware CPU input cache, new BN installation, condition metadata and R13 schema/status. All four final layouts are evaluated at each trained checkpoint. No old collector preparation, fixed fit20/check8 selection, old CMA payload, old learned head or R12 recovered weights are used.

No SSH or GPU action is performed by import or CPU tests. The runtime uses the existing shared GPU lock and a hard 30-minute alarm for each executed arm. New outputs are mandatory. GPU model allocation occurs only under the lock. The 200-identity CPU cache can hold approximately 3 GB of uncompressed input/GT/mask data; this is not GPU residency.

## Input contracts

- `--membership`: data protocol `manifest_v1.json`, schema `flow_round13_coverage_identity_v1`; full canonical rows, exact ordered identities, original roles and source hashes. Validation rejects sequence leakage, overlapping identity roles, bad canonical rows, unequal sequence schedules or wrong repetitions.
- `--runtime-manifest`: schema `flow_round13_runtime_v1`; `breadth_root`, `motion_root`, `base_args`, `source_pins`, `region_path`, `region_sha256`. `base_args` use the original base evaluator arguments and must omit all old `--bn-*`, `--svs-*`, `--regions*`, and `--run-inference` options. The new region JSON retains the existing region schema and definitions, but its `fit_target_ids` must exactly equal diversity200 order and its source evidence must match the accepted native GT/mask inventory.
- `--bn-buffers`: NPZ with exactly 72 flat fully qualified keys: 24 MLP BN modules × `running_mean`, `running_var`, `num_batches_tracked`. Mean/variance are FP32 with exact native shape, counters scalar int64 equal to176. Finite values and nonnegative variances are required. All values are validated before mutation.
- `--bn-receipt`: schema `flow_round13_common_bn_v1`, status `COMMON_BN_COMPLETE`; checkpoint and membership SHA; ordered `bn_target_ids`; `buffers_sha256`; `calibration_forwards=176`, `calibration_bn_count=78`, `retained_mlp_bn_count=24`, `BN_scope=MLP_R13_COMMON_CMA176`, `calibration_definition=all78_reset_CMA_train_nonBN_eval`, `old_CMA_loaded=false`, `GT_used=false`. Root's calibration resets all78 BN and uses training-side input-only cumulative calibration with other modules in eval; only private24 buffers are deployed.

The native model loader loads the original checkpoint running statistics directly. New24 MLP buffers then replace those statistics, and external54 running statistics are explicitly removed to use current input statistics. The old CMA176 payload is never loaded. Original model source and loss, every used array hash, timestamps, source manifests and CPU receipt are checked before GPU use.

## Full-forward budget (check8)

| Scope | native | each FF/CC/fixed block2 | Four-arm total |
|---|---:|---:|---:|
| Five-step smoke | 9 | 11 | 42 |
| 200-step recovery | 244 | 246 | 982 |

Each formal arm comprises 200 training forwards, 4/6 numerical audit forwards, and 40 evaluation forwards (8 initial own-layout plus 8×4 endpoint layouts). Across both conditions: 1,600 formal updates, 1,964 formal forwards, 40 smoke updates and84 smoke forwards. Common calibration adds176 forwards, totaling **2,224 full forwards**, not independent samples. Successful smoke weights are discarded. Every arm must pass its own condition's four five-step smoke receipts before formal recovery.

## Launch template

Create an input binding JSON with four absolute path fields `runtime_manifest`, `membership`, `bn_buffers`, `bn_receipt` and corresponding `*_sha256` fields. `launch_stage.py` requires that JSON's explicit SHA. Its default only runs source/data preflight. Add `--execute` to execute the explicitly requested stage. The launcher runs serially and aborts on first failure; it never provisions machines or changes CUDA visibility.

```bash
/path/to/python -B /path/to/implementation/launch_stage.py \
  --inputs /path/to/inputs.json --inputs-sha256 INPUT_JSON_SHA \
  --output-root /path/to/new_r13_runs --stage smoke
```

Repeat with `--execute` for smoke after reviewing preflight. Formal recovery uses `--stage recover` and reads the same output root's four completed smoke receipts per condition. For root-controlled staging, `--conditions reference` or `--conditions diversity` restricts the launch to one condition. No formal step can continue after its 30-minute cap; incomplete results are preserved with `INCOMPLETE_NO_SCIENTIFIC_CONCLUSION`.

## CPU validation

Run `test_coverage_cpu.py` using the CPU Torch environment. Six meaningful tests validate frozen membership and adversarial mutations; new BN installation and atomic rejection; all four toy native MLP layouts against original actual/dual references under the new BN state; one update's gradient and frozen-complement preservation; receipt rejection, schedule and exact forward accounting. A recursive bytecode global-dependency check also covers the extracted execute loop and its nested functions, including step logging. No full model, actual event/GT arrays, GPU, SSH or check quality is read by these tests. The passing receipt binds implementation bytes; it is not a full-model GPU smoke result.

Prelaunch correction: independent read review caught the extracted execute loop missing `import json`, which would fail at its first progress log. The import was added before any R13 recovery/smoke execution and the new dependency check covers this class of extraction error. No scientific result was affected.
