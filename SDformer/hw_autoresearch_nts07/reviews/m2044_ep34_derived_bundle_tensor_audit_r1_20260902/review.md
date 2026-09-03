# M2044 ep34 derived bundle final independent review

## Final verdict

- Score: **99/100**.
- Severity count: **P0/P1/P2 = 0/0/2**.
- Bundle tensor audit: **PASS**, CPU-only, CUDA unused.
- Decision: **AUTHORIZE_EXACTLY_ONE_GPU_VALID825_RUN**.

This review is an external independent admission event for the frozen M2044 contract. It does not modify the contract, producer, evaluator, source checkpoint, M2042 export, or derived bundle.

## Frozen authority

- M2044 producer source SHA256: edc5df9ce9debbb28863abf26426b7504c16552f7c47865b3a31a091b6cb9b20
- M2044 execution contract SHA256: 03f13063493d563cf0b26363498d18bde60c8bee5e785a4dfca95845555757d2
- Derived bundle manifest SHA256: ef2b502f7e17e2a28b11c4a627c8bc6f16ef78b5782b2636ace5a743544bdd8c
- Derived bundle outer-seal file SHA256: 32cf8a7f4a7c015bcf0086fd7676bc0b5360710981be7c425e14ae62475d06a2
- Bundle JSON SHA256: 01e7aadb454e82ce8fb04d25c4dc40f05bedd59cfd03d7e3835cdb2b967c3aee
- Source ep34 checkpoint SHA256: 4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48
- Derived checkpoint SHA256: daec6c188e7045ca3867c16cfcee5b25d2680eb4a7f1933541dfea17f0ac8371
- M2042 result SHA256: 455c9fe7036779b890d4b85911cc42dc47bcb62c9fb6f6a6ce9c28a2c833cf29
- Independent tensor-audit source SHA256: ee0555baf2a761912876cbda3a32353b26c1c0ba2e0f68cb5ee610eece7a393b
- Remote CPU-audit result SHA256: 0e8905bde3d54b53518b0795ea42656a16c7da305788d200e06e16261b415fe6

## Bundle governance and configuration review

The derived bundle has the expected four manifest members: RUN_COMPLETE, bundle JSON, derived checkpoint, and derived YAML configuration. Every member passes its inner SHA256 manifest, and the outer seal points to the exact reviewed manifest.

The bundle identity agrees with the execution contract and M2042:

- exactly four C1 Conv3x3 and four decoder ConvTranspose target weights;
- native output axes 0 for C1 and 1 for ConvTranspose;
- per-target canonical-code, scale-exponent, and source-weight identities match M2042;
- the source checkpoint, source configuration, derived checkpoint, and producer identities are exact-bound;
- the derived configuration preserves h60, alpha0=0.02, binary Motion-XOR alpha=0.125, BN no_running, and evaluation batch size 1;
- Q7 score, Q1.7 gate, hardware-order Shiftmax, TF32 disabled, cuDNN benchmark disabled, and the eight forward-audit targets are configured;
- no full-network INT8 or SystemVerilog-equivalent claim is made.

## Independent tensor audit

The exact audit source above was executed remotely in CPU-only mode. Its machine-readable result is sealed in this review and reports:

- source state keys: 921;
- derived state keys: 921;
- tensor keys checked: 921;
- non-target tensors checked: 913;
- non-target torch.equal passes: 913;
- target tensors checked: 8;
- target source-SHA passes: 8;
- target M2042 metadata passes: 8;
- independently reconstructed dyadic-INT8-QDQ torch.equal passes: 8;
- target recorded-SHA passes: 8;
- total mismatches: 0;
- non-target elements checked: 27,900,311;
- target elements checked: 28,373,760;
- CUDA used: false.

This closes the contract's derived-bundle review gate. It proves that only the eight intended tensors changed and that each changed tensor is exactly the M2042 dyadic INT8 reconstruction at checkpoint precision.

## Residual P2 observations

1. Bundle JSON does not directly embed the execution-contract SHA. The frozen producer independently pins that contract and reconstructs the expected configuration during run admission, so this is traceability hardening rather than a blocker.
2. The producer prints a generic PASS_M2044_VALID825_EXECUTION line after an executed accuracy-gate failure. The sealed result status, not terminal stdout, is authoritative; an accuracy-gate failure must remain a negative result and cannot be described as an accuracy pass.

## Exactly-one-run authorization

Immediately before execution on the A800 server, the operator must verify:

1. The producer source, execution contract, bundle manifest, and bundle outer seal equal the SHA256 values above.
2. The canonical result directory, its temporary directory, and its FAILED_DO_NOT_CITE sibling all do not exist.
3. No other M2044 evaluator process is running.
4. The interpreter is /opt/conda/bin/python3.

The only authorized command is:

    /opt/conda/bin/python3 -I hw_autoresearch_nts07/system_handoff/scripts/run_m2044_ep34_valid825_attention_eight_operator_qdq.py --expected-source-sha256 edc5df9ce9debbb28863abf26426b7504c16552f7c47865b3a31a091b6cb9b20 --expected-bundle-manifest-sha256 ef2b502f7e17e2a28b11c4a627c8bc6f16ef78b5782b2636ace5a743544bdd8c --run

This authorization permits one process and one attempt. Any runtime failure or accuracy-gate failure consumes the authorization. Automatic retry is forbidden. The result must be transferred back and independently hammered before it becomes paper-citable.

## Claim boundary

This review admits the derived checkpoint identity, unchanged 913-tensor population, eight exact dyadic-INT8-QDQ replacements, and the right to execute one paired valid825 experiment.

It does not by itself admit:

- valid825 accuracy or the AEE gate;
- attention or transformed-operator forward reachability;
- whole-network INT8;
- whole-network hardware-order or SystemVerilog equivalence;
- hardware cycles or speedup;
- energy, power, area, timing, or PPA;
- any paper result before independent result hammering.

## Disposition

**PASS_M2044_DERIVED_BUNDLE_FINAL_REVIEW_AND_AUTHORIZE_EXACTLY_ONE_GPU_VALID825_RUN.**
