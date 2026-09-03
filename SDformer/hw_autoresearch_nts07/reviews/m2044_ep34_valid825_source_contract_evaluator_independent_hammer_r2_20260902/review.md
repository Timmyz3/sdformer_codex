# M2044 source/contract/evaluator independent review R2

## Verdict

Score: **96/100**. P0=0, P1=2, P2=3.

The six P0 findings from the first review are closed in the exact reviewed
source. This review authorizes exactly one `--prepare-only` execution for source
SHA `edc5df9c...b9b20` and contract SHA `03f13063...57d2`. It does **not**
authorize the GPU `--run` phase. The resulting derived bundle must first receive
an independent review that pins its `SHA256SUMS` digest.

## Closed first-round P0 findings

1. The producer now checks the exact contract SHA before parsing it.
2. Existing prepare/run failure namespaces are rejected before a new phase can
   start. Expected exceptions after phase staging use the common sealed failure
   publication path.
3. `--run` requires an independently supplied bundle-manifest SHA. It also
   checks the bundle authority fields, exact four-member population, checkpoint
   and config hashes, exact config semantics, all 921 state keys, equality of
   all non-target tensors, and exact reconstruction of all eight target QDQ
   tensors.
4. The evaluator and the critical BSA/load-audit/ATLIF-installer/metric/dataset
   sources are SHA-pinned and rechecked after the long evaluator execution.
5. The validation-list SHA is fixed to `7f3dc280...25d0`; candidate and baseline
   evaluation protocol, metric contract, aggregation population, and per-sequence
   population must match.
6. The AEE admission gate is now one-sided: improvements are admitted and only
   degradation above 0.02 fails.

## Correct semantic points

- Four C1 weights retain native output axis 0.
- Four ConvTranspose weights are converted from canonical `[O,I,Ky,Kx]` back
  to native `[I,O,Ky,Kx]` with axis 1.
- Only the frozen eight target keys receive QDQ values. Before GPU execution,
  the source compares the full 921-key derived state with the original state and
  requires every non-target tensor to be equal.
- The derived config preserves H60, `alpha0=0.02`, and Motion-XOR alpha 0.125,
  while selecting Q7 score quantization and the existing RTL-order Q8-LUT/Q1.7
  Shiftmax path.
- The evaluator records runtime TF32/cuDNN flags and requires each transformed
  operator to execute exactly once per each of 825 samples.
- Claim boundaries remain subset-deployment accuracy only. They do not claim
  full-network INT8, full-network hardware-order equivalence, SystemVerilog
  equivalence, cycles, speedup, energy, or PPA. Even a passing AEE gate leaves
  `paper_accuracy_result=false` until a fresh independent result hammer passes.

## Remaining P1

1. The evaluator still discovers the H9 overlay by directory ordering. The
   intended source files are pinned, but the final GPU review should additionally
   record the imported modules' resolved `__file__` paths and SHAs on the A800.
2. `retain_failure` deliberately rejects an unexpected child directory instead
   of recursively sealing it. The frozen visualization configuration prevents
   expected child directories, but a future generic evaluator wrapper should
   recursively seal or explicitly quarantine them.

## Remaining P2

1. Record Python, PyTorch, CUDA, cuDNN, CuPy, NumPy and YAML versions in the
   final result.
2. Add explicit dtype/shape equality beside `torch.equal` for the 913 non-target
   state tensors, even though the current save/load path preserves them.
3. A future general contract can bind a sealed independent-review receipt rather
   than relying on the operator to supply the reviewed bundle manifest SHA.

## Exact prepare-only authorization

The authorized command must use the reviewed source SHA and must not include
`--run`:

```bash
/opt/conda/bin/python3 -I \
  hw_autoresearch_nts07/system_handoff/scripts/run_m2044_ep34_valid825_attention_eight_operator_qdq.py \
  --expected-source-sha256 edc5df9ce9debbb28863abf26426b7504c16552f7c47865b3a31a091b6cb9b20 \
  --prepare-only
```

Automatic retry is not authorized. A failed prepare must remain sealed as
`*_FAILED_DO_NOT_CITE`; a successful bundle is not yet an accuracy result.
