# M1544 ep34 S2/TSBG shared incremental capture handoff

This is a source-only, compact handoff. It does **not** authorize a GPU run.

Purpose:

- capture exact FC1/FC2/PATCH token and source-group order once;
- support lossless TSBG row-buffer comparison and S2 CCBS block fast-kill;
- add only compact S1 magnitude/debt histograms;
- never save complete FP16/FP32 activation or output tensors.

The fixed identity is Motion ep34 checkpoint
`4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48`
and the exact M1458 40-sample order. The validator also pins the M1458 inner
manifest and outer seal, plus M1540/M1541.

ep34 currently has **no formal INT8/fixed-point quantization authority**. Every
producer codebook must therefore carry `diagnostic_capture_only=true` and
`hardware_quant_authority=false`. TSBG exactness is limited to the captured
codeword/contributor stream; it is not model bit-exactness, an AEE result, an
Acc24 proof, or hardware admission. A formal INT8 result remains blocked on a
separately identified deterministic PTQ/QAT bridge and integer miter.

The intended producer emits one compressed JSON row per logical token. Zero
tokens remain as `groups=[]`, preserving scheduling and commit/tail evidence.
Within a non-zero token, only non-empty source groups are stored, with support,
sign, non-unit bitsets and non-zero fixed-point codes. The static weight
address, bank and strong-baseline row-buffer map appears once in `layers.json`.

Before any remote run:

1. Integrate the producer against the exact ep34 call paths without changing
   sample order.
2. Estimate the compressed result size before loading the checkpoint. Abort if
   it exceeds 12 GiB or leaves less than 16 GiB free.
3. Obtain a separate one-shot production release. This handoff is not one.
4. Run the bundled unit test and then validate the produced directory:

```bash
python3 -m unittest -v \
  hw_autoresearch_nts07.tests.test_validate_m1544_ep34_sparse_capture_handoff

python3 hw_autoresearch_nts07/system_handoff/scripts/\
validate_m1544_ep34_sparse_capture_handoff.py \
  --capture-dir hw_autoresearch_nts07/results/\
m1544_ep34_s2_tsbg_shared_incremental_capture_s40_r1_20260831
```

Passing validation proves capture structure and identity only. It does not
prove opportunity, cycles, bytes, energy, model bit-exactness, AEE, or RTL
admission.
