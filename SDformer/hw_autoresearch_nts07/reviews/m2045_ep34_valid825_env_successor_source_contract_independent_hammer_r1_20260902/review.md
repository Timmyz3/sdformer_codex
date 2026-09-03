# M2045 environment-successor independent review

Score: **93/100**. P0=0, P1=2, P2=3.

The exact reviewed wrapper changes the execution environment and output
namespace, while importing the frozen M2044 engine and passing it the same
reviewed bundle manifest. It does not change the candidate checkpoint, config,
evaluator, attention/QDQ semantics, population, or AEE gate.

The sealed M2044 failure contains only `eval.log` and `FAILURE.txt`. The pinned
trace stops while importing `Spiking_modules.py` with
`ModuleNotFoundError: No module named 'spikingjelly'`; model construction,
validation and AEE did not execute. This establishes the exact observed first
blocker, not a proof that no second environment dependency can appear after it.

## Admission

Exactly one preflight is authorized for source SHA
`890dfd6b...17e1` under `/opt/conda/envs/sdformerflow`. GPU execution is not
authorized by this review. The preflight receipt must be independently checked
before a separate one-run decision.

The preflight now performs more than package presence checks: it loads the
frozen M2044 engine, re-verifies all M2044 inputs, and performs the full deep
bundle verification against manifest `ef2b502f...bdd8c`.

## Remaining P1

1. The preflight imports `spikingjelly` and `torch` but does not perform a full
   no-GPU import smoke of the evaluator dependency graph. A second missing
   dependency may therefore still appear only when the successor run starts.
2. A successful output retains the M2044 engine schema and does not contain an
   M2045-wrapper SHA field. The independent result hammer must bind the M2045
   contract/source, output namespace, and `eval.log` interpreter path around
   the sealed M2044-schema result.

## Remaining P2

1. `verify_environment` constrains `sys.prefix` and the package locations, but
   does not explicitly require `sys.executable` itself to reside under that
   prefix. The preflight receipt exposes both values and must be checked.
2. Package versions and full dependency inventory are not frozen yet.
3. The source verifies the two failure evidence members individually and pins
   the exact manifest, but does not reject additional unsealed files in the old
   failure directory.

## Exact preflight command

```bash
/opt/conda/envs/sdformerflow/bin/python3 -I \
  hw_autoresearch_nts07/system_handoff/scripts/run_m2045_ep34_valid825_sdformerflow_env_successor.py \
  --expected-source-sha256 890dfd6bac5ddd2696af41ecfbc1a98cc1284d64ef6fbdbf993d485274dd17e1 \
  --preflight
```

Do not append `--run`; no GPU attempt is authorized by this review.
