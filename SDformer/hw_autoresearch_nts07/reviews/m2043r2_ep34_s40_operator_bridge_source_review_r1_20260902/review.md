# M2043r2 ep34 S40 operator bridge independent source review

## Verdict

- Review score: **97/100**.
- Severity count: **P0/P1/P2 = 0/0/3**.
- Decision: **AUTHORIZE_EXACTLY_ONE_REMOTE_A800_PRODUCTION_ATTEMPT**.
- This review is the independent external source-review authorization required by the execution contract status SOURCE_REVIEW_REQUIRED__NO_PRODUCTION_AUTHORITY. It does not edit, replace, or reinterpret that frozen contract.

The authorization is bound to exactly these immutable inputs:

- Producer source: hw_autoresearch_nts07/system_simulator/scripts/run_m2043_ep34_s40_operator_int8_bridge.py
- Producer source SHA256: 3be570ab39a9c72223ba4c2ae0919b317683a69b19d0b417117825bd7c293557
- Execution contract: hw_autoresearch_nts07/contracts/m2043r2_ep34_s40_operator_bridge_execution_contract_r1_20260902.json
- Execution contract SHA256: 92fd28fcdbd4cf6f2e6d8d76a3fa28f9e46acabc775d8d1ff5927337bee324e3
- Protected docs/359 SHA256: dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
- Canonical output: hw_autoresearch_nts07/results/m2043r2_ep34_s40_eight_operator_int8_bridge_r1_20260902

The one-attempt authorization additionally requires, on the remote host immediately before execution:

1. Recheck both source and execution-contract SHA256 values above.
2. Confirm the canonical output and its deterministic temporary directory do not exist.
3. Use the reviewed A800 CUDA environment and pass the exact producer SHA through --expected-source-sha256.
4. Run one process once. Do not automatically retry after any failure.
5. Treat any failure as FAILED_DO_NOT_CITE pending a new independent review.

## Source findings

### Operator equations and layouts

The C1 path implements unflipped Conv3x3 cross-correlation with stride 1 and padding 1. Its sampled oracle uses iy=oy+ky-1 and ix=ox+kx-1, which matches the frozen operator contract.

The decoder path consumes canonical O,I,Ky,Kx codes. It inserts each input at even coordinates in a 2H by 2W zero tensor, flips both spatial kernel axes, then applies stride-1 unfold with padding 1. This gives:

- zero-insert coordinate: y'=2iy, x'=2ix;
- correlation condition: y'=oy+u-1 and x'=ox+v-1;
- after u=2-ky and v=2-kx: oy=2iy-1+ky and ox=2ix-1+kx.

Therefore it is exactly equivalent to the frozen ConvTranspose2d configuration stride=2, padding=1, output_padding=1 and no native kernel flip. Independent small-shape NumPy reconstruction also produced zero mismatches.

Unfold returns N,I*9,L and the code matrix is O,I*9. Torch matmul broadcasts the latter over N and returns N,O,L; reshaping to N,O,Hout,Wout is geometrically correct for both operator families.

### Integer-domain exactness and Acc24

The integer path does not use a hidden convolution algorithm. It uses explicit zero insertion where required, unfold, and TF32-disabled FP32 GEMM.

All support values are exactly 0 or 1 and all INT8 code values are exactly representable binary32 integers. The sealed M2042 export proves a global maximum absolute-code or decoder-polyphase sum of 200,219. Every reduction-tree partial sum is bounded by the sum of magnitudes of its contributing terms, so its magnitude is below 200,219, which is below 2^24. Products and all reordered partial sums are consequently exact in binary32 when TF32 is disabled and highest FP32 matmul precision is selected.

The full final accumulator population is checked for integrality and signed-Acc24 range. At least eight deterministic coordinates per call are independently recomputed using Python integer addition. A formal absolute-sum proof covers every direct-dot-product prefix and every support pattern; sampled observed prefixes remain explicitly labeled as sampled rather than a full observed prefix population.

### Population, quantization, and conservation

The sealed M1458 population is exactly 160 calls: four C1 operators times 10 calls and four decoder operators times 30 calls. Independent census found unique sample IDs and sample keys within every operator.

Per-output power-of-two scales, exact IEEE-754 alpha words, no-bias semantics, canonical code layout, and the decoder axis-1 sentinel inherited from the sealed M2042 export are consistent. Full-output FP32-versus-QDQ MAE, RMSE, maximum absolute error, and cosine are accumulated. The one-zero cosine case is no longer mislabeled as perfect similarity.

Reusing M1597 for C1 parent/add-sub conservation is legitimate. The exact-sealed M1597 review is bound to the same checkpoint SHA256 and M1458 ordered-record SHA256, covers the same four C1 operators and 51,840,000 rows, and independently closes parent-edge, dead-read/forward, and dead-write/elision equalities. Recomputing that ledger inside this operator-local numerical bridge is unnecessary.

### Publication

Production is restricted to the canonical output path. The source refuses an existing canonical output or stale temporary directory, writes into a private temporary directory, creates an inner manifest and outer seal, atomically publishes, and reads back the outer seal, manifest, and result identity before returning PASS.

## Residual P2 observations

These are defense-in-depth improvements and do not block the single authorized production attempt:

1. The M1597 loader relies on an exact-pinned sealed review plus its chain-pass boolean. A future successor could additionally assert the review's embedded checkpoint and ordered-record SHA fields directly before copying conservation values.
2. Post-publication readback validates the manifest, outer seal, and result, but does not separately hash-check RUN_COMPLETE or enforce an exact final-directory topology. A successor can add those checks.
3. The inherited ConvTranspose axis-1 sentinel is copied from an exact-pinned M2042 result but is not separately required to equal PASS in this source. A successor can make that inherited premise explicit.

## Claim boundary

If and only if the authorized run completes and its result passes an independent result hammer, the following may be admitted:

- operator-local full-population FP32-versus-QDQ metrics for the frozen 40 C1 and 120 decoder calls;
- full final integer-domain accumulator range and integrality;
- sampled independent integer-oracle equality;
- the formal direct-dot-product prefix Acc24 proof;
- the exact-sealed M1597 C1 parent/add-sub conservation result;
- the inherited exact-sealed M2042 ConvTranspose axis-1 sentinel.

The review does **not** authorize claims of:

- valid825 AEE or end-task accuracy preservation;
- downstream ATLIF equivalence;
- whole-network hardware-order equivalence;
- hardware cycles, component or system speedup;
- energy, power, area, timing, or PPA;
- a paper-ready result before independent result hammering.

## Final disposition

**PASS_SOURCE_REVIEW_AND_AUTHORIZE_ONE_REMOTE_A800_ATTEMPT.**

The execution contract remains byte-for-byte unchanged. This independently sealed review is the external authorization event requested by that contract.
