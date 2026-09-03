# M2045 ep34 valid825 canonical result independent review

## Verdict

- Score: **96/100**.
- Severity count: **P0/P1/P2 = 0/1/2**.
- Result: **PASS_FOR_PAIRED_SUBSET_ACCURACY_ADMISSION**.
- Paper decision: main.tex may cite baseline AEE 1.199514 and candidate AEE 1.197367, delta -0.002147, only as the paired valid825 accuracy result of the existing attention hardware-order mode plus eight-operator dyadic-INT8 QDQ.

This review is read-only with respect to the canonical result and all predecessors. It does not admit full-network INT8, full-network RTL-order equivalence, cycles, speedup, energy, or PPA.

## Exhaustive seal and identity audit

The canonical M2045 result has exactly four sealed members plus its manifest and outer seal. There are no unsealed files, subdirectories, or symlinks.

- Canonical manifest SHA256: c25a4857b5cd40616aa94324b396ed9a96d457a1453307a29eb99918fadf59fa
- Canonical outer-seal file SHA256: a926d722381df3c1f2961adf81fb2bf5cbcf4963082227c554ea16b2711bea93
- result.json SHA256: bf73e27cba9c69461d5cfc0ff97fb30b4ceadb08ac726fb52443278a6629a831
- spike_profile.json SHA256: 3b9d5fe7adf2156ebf4f2d0df286a629e9f19b9df3a47b0d47a70c0e87d37e33
- eval.log SHA256: a4bb5d7a24c6a9ce68ad01d267c5ebe1ad0c542f29ab5b797450392a17818a95
- RUN_COMPLETE.txt SHA256: dbcfe480312ed17260e2310879c0aa65c535096d78f046c3110f286f5c26a1dc

The following execution chain is exact:

- M2045 wrapper SHA256: 890dfd6bac5ddd2696af41ecfbc1a98cc1284d64ef6fbdbf993d485274dd17e1
- M2045 contract SHA256: 4c3222055a7fa7b8b246ab43caf7b37a7eeb8554021f3556d9998942d302bdb0
- Frozen M2044 engine SHA256: edc5df9ce9debbb28863abf26426b7504c16552f7c47865b3a31a091b6cb9b20
- Frozen M2044 contract SHA256: 03f13063493d563cf0b26363498d18bde60c8bee5e785a4dfca95845555757d2
- Reviewed bundle manifest SHA256: ef2b502f7e17e2a28b11c4a627c8bc6f16ef78b5782b2636ace5a743544bdd8c
- Bundle checkpoint SHA256: daec6c188e7045ca3867c16cfcee5b25d2680eb4a7f1933541dfea17f0ac8371
- Bundle configuration SHA256: 977d8f654e7aa5d528ca77a3a374d5d6554cc51b7773c1e579c08a79bcc6646d
- Evaluator SHA256: 84daee48291d8ab2ee644f43458b909e96190c0dce7f5ff4d4179b61be30faac
- Paired baseline profile SHA256: 144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c
- Validation list SHA256: 7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0

The retained M2044 failure remains independently double-sealed and shows that the predecessor stopped on the missing spikingjelly import before model construction or accuracy evaluation. The M2045 eval log records the sdformerflow interpreter and evaluator exit code zero.

## Paired population and metric recomputation

Candidate and baseline use the same:

- 825 validation samples and aggregation frames;
- 18 DSEC sequences;
- 48,152,523 valid pixels;
- exact validation-list SHA;
- evaluation protocol, including batch size 1 and BN no_running;
- metric contract and frame-equal aggregation definition.

Every sequence has identical frame and valid-pixel counts between baseline and candidate. Independent recomputation from all 18 per-sequence rows closes:

- global frame-equal metrics to at most 4.45e-15;
- pixel-global metrics exactly at serialized precision;
- sequence-balanced metrics to at most 1.78e-15.

The separately serialized profile metrics differ from the aggregation-audit values by at most 3.12e-7, consistent with the evaluator's accumulation precision and immaterial to the gate.

The result metrics exactly reproduce the sealed profiles:

| Metric | Baseline | Candidate | Candidate - baseline |
|---|---:|---:|---:|
| AEE | 1.1995140134 | 1.1973673040 | -0.0021467094 |
| AAE | 5.4006410839 | 5.4128083761 | +0.0121672922 |
| AAE Benchmark | 5.1063634050 | 5.1216190149 | +0.0152556099 |
| DSEC Fl | 5.3133596618 | 5.3288341660 | +0.0154745042 |

The AEE delta is below the frozen maximum candidate-minus-baseline threshold of +0.02, so the accuracy-preservation gate passes. The paper should describe this as preservation under the subset deployment transform, not as a statistically established accuracy improvement.

Per-sequence AEE improves on 8 sequences and regresses on 10. The minimum delta is -0.062317 on zurich_city_09_a; the maximum is +0.021543 on zurich_city_06_a. This distribution is finite and plausible, but the aggregate improvement is substantially influenced by zurich_city_09_a.

## Load, backend, and forward audit

- Checkpoint load: missing=0, unexpected=0, overlay missing=0, overlay unexpected=0.
- Artifact checkpoint and configuration SHA values match the reviewed bundle.
- Installed modules: 105 ATLIFTernaryPSN and 12 ShiftmaxAttention.
- Backend: CUDA matmul TF32 disabled, cuDNN TF32 disabled, cuDNN benchmark disabled.
- All eight transformed operators are reached exactly 825 times.
- Recorded output-element totals independently match the eight fixed tensor geometries.
- The deployment contract identifies four C1 Conv3x3 and four decoder ConvTranspose dyadic-INT8-QDQ weights; all other operators remain in checkpoint precision.

## Residual findings

### P1

The canonical result intentionally retains the M2044 engine schema and producer SHA, so it does not embed the M2045 wrapper or M2045 contract SHA. This review binds the exact M2045 canonical namespace, wrapper, contract, sdformerflow interpreter evidence, frozen engine, and sealed output. Future artifacts should place the successor wrapper and contract identities directly in result.json.

### P2

1. Profile headline metrics and aggregation-audit frame means differ by up to 3.12e-7 because they traverse slightly different accumulation precision. Both are internally consistent and the difference is far below the +0.02 gate.
2. Eight of eighteen sequences improve in AEE and ten regress. The -0.002147 aggregate delta must not be presented as statistically significant or universally improving; it supports accuracy preservation for the exact paired population.

## Exact paper claim boundary

Permitted:

> On the identical local DSEC valid825 population, enabling the existing Q7/Q1.7 attention hardware-order path and replacing only four bottleneck Conv3x3 plus four decoder ConvTranspose weights with per-output dyadic-INT8 QDQ changes AEE from 1.199514 to 1.197367 (delta -0.002147), passing the predefined maximum-degradation gate of +0.02.

Required qualifiers:

- local DSEC validation split, not the official hidden test;
- paired subset deployment transform;
- attention hardware-order configuration plus exactly eight QDQ weights;
- all other weights and operators remain at checkpoint precision.

Forbidden:

- calling the candidate full-network INT8;
- claiming full-network SystemVerilog or hardware-order equivalence;
- interpreting this accuracy experiment as a cycle, speedup, energy, power, area, timing, or PPA result;
- multiplying or combining this AEE result with component speedups;
- claiming statistically significant accuracy improvement from the small negative aggregate delta.

## Final disposition

**PASS_M2045_PAIRED_VALID825_SUBSET_ACCURACY_FOR_PAPER_CITATION_WITH_EXACT_QUALIFIERS.**
