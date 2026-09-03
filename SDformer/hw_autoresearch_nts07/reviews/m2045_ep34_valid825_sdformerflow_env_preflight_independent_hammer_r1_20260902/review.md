# M2045 sdformerflow preflight independent review

## Decision

Score: **94/100**. Severity: **P0=0, P1=2, P2=3**.

The fixed preflight receipt SHA is
`41da22f4b5745e5919f6177267bfefeb4d168815f43196c53755e89cad74079a`.
It is exactly the deterministic JSON emitted only after the M2045 wrapper has:

- checked the exact wrapper, contract, frozen M2044 engine, retained M2044
  failure, tensor audit, and result-namespace absence;
- imported `spikingjelly` and `torch` from `/opt/conda/envs/sdformerflow`;
- loaded the frozen M2044 contract and verified its inputs and reviewed bundle.

This closes the exact *observed* M2044 blocker (`spikingjelly` import). It does
not prove that no later evaluator dependency or runtime failure exists.

Exactly **one** GPU production run is authorized. Automatic retry is not
authorized. A failed attempt must remain sealed. A successful result is not a
paper accuracy result until a new independent result hammer binds the M2045
wrapper and contract around the M2044-schema result.

## Fixed execution identity

- Interpreter: `/opt/conda/envs/sdformerflow/bin/python`
- Wrapper SHA256: `890dfd6bac5ddd2696af41ecfbc1a98cc1284d64ef6fbdbf993d485274dd17e1`
- Contract SHA256: `4c3222055a7fa7b8b246ab43caf7b37a7eeb8554021f3556d9998942d302bdb0`
- Frozen M2044 engine SHA256:
  `edc5df9ce9debbb28863abf26426b7504c16552f7c47865b3a31a091b6cb9b20`

From the A800 frozen repository root, the only authorized command is:

```bash
/opt/conda/envs/sdformerflow/bin/python -I \
  hw_autoresearch_nts07/system_handoff/scripts/run_m2045_ep34_valid825_sdformerflow_env_successor.py \
  --expected-source-sha256 890dfd6bac5ddd2696af41ecfbc1a98cc1284d64ef6fbdbf993d485274dd17e1 \
  --run
```

The wrapper itself rechecks absence of the canonical, temporary, and failed
M2045 namespaces immediately before execution.

## Remaining findings

### P1

1. The preflight receipt is a minimal stdout capture and does not itself carry
   argv, host identity, wrapper SHA, or contract SHA. This independent sealed
   review binds its fixed SHA to those identities, but the production result
   must bind them again.
2. Preflight imports the repaired packages and deeply verifies the bundle, but
   it does not execute the evaluator dependency graph. A later dependency or
   runtime failure therefore remains possible; the no-retry failure seal is the
   correct containment.

### P2

1. Dependency versions and a complete environment lock are not included.
2. The receipt does not record GPU/driver/CUDA or remote host identity.
3. The successful output intentionally retains the M2044 result schema and
   producer identity; the M2045 lineage exists only in the wrapper namespace
   until the required independent result hammer is complete.

## Claim boundary

This authorization changes no algorithm, checkpoint, evaluator, valid825
population, AEE threshold, hardware cycle, speedup, energy, or PPA claim.
