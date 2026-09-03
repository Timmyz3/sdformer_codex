# M2043r2 ep34 S40 eight-operator INT8 bridge independent result hammer

## Verdict

- Status: **PASS_M2043R2_EP34_S40_OPERATOR_BRIDGE_RESULT_HAMMER**.
- Review score: **99/100**.
- Severity count: **P0/P1/P2 = 0/0/1**.
- Scope: independent read-only result audit; no production replay, GPU, EDA, RTL, or remote execution was used.
- Mutation boundary: the producer source, execution contract, authority, production result, upstream evidence, and protected `docs/359` were not modified.

The sealed result is admissible only as an operator-local numerical and integer bridge for the frozen ep34 C1 and decoder population. It is not an accuracy, cycle, PPA, energy, or system-speedup result.

## Audited result and seal

Canonical result:

- Path: `hw_autoresearch_nts07/results/m2043r2_ep34_s40_eight_operator_int8_bridge_r1_20260902`
- `result.json` SHA256: `1378c5404cef42934e679591bc9ea1e9cd26da3e279a396dcc255afd8644f7de`
- `RUN_COMPLETE.txt` SHA256: `257ea08c1537a724368e2d4892881cc8f4bab23d8ce72094ab3f97895ddb23e8`
- `SHA256SUMS` SHA256: `192dd19e986b4b5bed078f19a13b6f23c0ff91cdc51a8aabee72c88fa349ab8d`
- Outer-seal file SHA256: `755067bb8fd56da677baf8fd5b0c6c16a249eddd99fac7009663ea229f56a4b0`

Both inner member checks pass. The outer seal contains the exact SHA256 of `SHA256SUMS`. Manifest membership is exact: the only sealed payload members are `result.json` and `RUN_COMPLETE.txt`. The PASS token is exact, and the result directory contains no symbolic links.

## Identity and upstream pins

Every recorded identity was independently hashed from disk and matched:

- Producer source SHA256: `3be570ab39a9c72223ba4c2ae0919b317683a69b19d0b417117825bd7c293557`
- M2042 quantization authority SHA256: `7f4d09b1d7d9bd3ffafb0e03b5d74100ec0082992518306cc8e81b6939c44cd0`
- M2043r2 execution-contract SHA256: `92fd28fcdbd4cf6f2e6d8d76a3fa28f9e46acabc775d8d1ff5927337bee324e3`
- Checkpoint SHA256: `4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48`
- Protected `docs/359` SHA256: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

M2042 export chain:

- Result SHA256: `455c9fe7036779b890d4b85911cc42dc47bcb62c9fb6f6a6ce9c28a2c833cf29`
- `SHA256SUMS` SHA256: `519b8621a0c16f67ed33c8c624adc6bbfbc1c4a27224b2812542da3d92fc3881`
- Outer-seal file SHA256: `da977b9effab3accaff229877bc4d9f0e930f82de1c0833be5c872e63aee142b`
- ConvTranspose native-axis-1 sentinel: `PASS`.

M1458 capture chain:

- Capture manifest SHA256: `3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d`
- Ordered JSONL SHA256: `5956085b196979848c3d283744396ea3b0a38a268fb21af0eaecb53e87fc6c9c`
- `SHA256SUMS` SHA256: `f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e`
- Outer-seal file SHA256: `7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed`

M1597 conservation chain:

- Independent review SHA256: `bfa3414ebb69d4a3022182ef7a4989d738c8370a855dff3ce5232c320623c33f`
- `SHA256SUMS` SHA256: `36dc79f7ca76bb98dfe1126aa05c7158dfc460d33215ee39d6fee4edd98e016c`
- Outer-seal file SHA256: `8f53a7fa74a2d0245448e822bc35b040df31b3e7d40d46d8ea739e6856d4df8b`
- Sealed M1590 result SHA256: `facfecaf3b25a4c79299517de31283ed3815af26a5dd87c91a6985f6fc68516f`

All three upstream seal trees pass `sha256sum -c`.

## Source-review authorization closure

The frozen execution contract retains its original status `SOURCE_REVIEW_REQUIRED__NO_PRODUCTION_AUTHORITY`. The independent source reviewer authorized exactly one A800 production attempt for the exact producer and contract hashes before the run; that review was subsequently persisted without modifying the frozen source, contract, or result:

- Receipt path: `hw_autoresearch_nts07/reviews/m2043r2_ep34_s40_operator_bridge_source_review_r1_20260902`
- Review SHA256: `aa52ee5c90a54a47f88092926e5a7a7f821ac5ec869fc6b6470c7e82883fad9e`
- `SHA256SUMS` SHA256: `7943f79c035d27008a32a2066d9dbc02e714b7aa3a20dd0b341039e872de3ef4`
- Outer-seal file SHA256: `60f73668b5528b0ac98f8aa29053610bf68f09a46cd4539e3152d082ec0e81f0`
- Source-review verdict: 97/100, P0/P1/P2 = 0/0/3, authorize the exact source SHA for exactly one remote A800 production attempt.

The receipt has exact two-member manifest membership, a valid outer seal, an exact PASS token, and no symbolic links.

## Population and uniqueness

The M1458 ordered population was rescanned independently. The selected identity set matches the result call set exactly, including global sample ID, sample key, module, per-module call ordinal, and raw FP32 payload SHA256.

- Operators: 8 unique modules.
- Calls: 160 total.
- C1: four Conv3x3 operators, 10 calls each, 40 calls total.
- Decoder: four ConvTranspose operators, 30 calls each, 120 calls total.
- Unique global sample IDs: 40.
- Unique sample keys: 40.
- Each selected sample contributes exactly four operator calls.
- Duplicate call identities: 0.
- Output elements: 3,271,680,000.

The result is grouped by operator whereas the capture is ordered by sample. Equality is therefore established as an exact identity set rather than by requiring the two files to have the same serialization order.

## Weighted aggregate rederivation

The eight layer rows were aggregated using ratio-of-sums rather than an unweighted mean. Calls, elements, absolute-error sum, squared-error sum, reference squared sum, candidate squared sum, and dot product all close against the published aggregate. The only floating summation difference was `7.45e-09` in an absolute-error total of approximately 29.8 million.

- Calls: 160.
- Elements: 3,271,680,000.
- Absolute-error sum: 29,793,307.99198673.
- Squared-error sum: 503,624.15513570374.
- Reference squared sum: 11,934,956,565.165379.
- Candidate squared sum: 11,936,372,122.727932.
- Dot product: 11,935,412,531.869087.
- Weighted MAE: 0.009106424831275287.
- Weighted RMSE: 0.01240703097148129.
- Maximum absolute error: 0.23920392990112305.
- Aggregate cosine similarity: 0.9999789043085248.

These are full-output, operator-local FP32-versus-QDQ metrics. They are not valid825 AEE and do not prove downstream-network accuracy preservation.

## Integer oracle and Acc24 proof

- Deterministic Python-integer probes: 1,280, exactly eight per call.
- Integer-oracle mismatches: 0.
- Full observed final accumulator range: `[-29,680, 27,619]`.
- Sampled observed hardware-order prefix range: `[-6,064, 10,998]`.
- Sampled prefixes are correctly labeled as sampled, not as a full observed prefix population.

The per-operator static and formal maximum absolute prefix bounds are:

1. C1-0: 200,219.
2. C1-1: 192,772.
3. C1-2: 199,711.
4. C1-3: 177,812.
5. Decoder-0: 87,136.
6. Decoder-1: 39,672.
7. Decoder-2: 19,955.
8. Decoder-3: 10,453.

For every layer, the formal bound equals the exported static absolute-code or polyphase bound. The global maximum is 200,219, strictly below `2^24` and far inside signed Acc24 `[-8,388,608, 8,388,607]`. Since operands are exact integer support values in `{0,1}` and exact signed-INT8 codes, every product is exactly representable in binary32. The absolute-sum bound covers every support pattern, issue order, and reduction-tree prefix; with TF32 disabled, all integer partial sums remain exact.

## C1 parent/add-sub conservation

The bridge correctly inherits rather than recomputes the sealed M1597 51.84-million-row ledger:

- Source rows: 51,840,000.
- Parent edges: 16,189,026.
- Dead reads: 14,506,449.
- Dead forwards: 1,682,577.
- Dead writes: 9,070,756.
- Dead elisions: 16,233,457.

The independently checked identities are:

- `14,506,449 + 1,682,577 = 16,189,026` parent edges.
- `9,070,756 + 16,233,457 = 25,304,213` active rows.

The checkpoint/capture/ordered-ledger chain and all M1597 conservation equalities remain PASS. M2043r2 truthfully records `recomputed_by_m2043r2=false`.

## Claim boundary

The exact admitted true claims are:

- C1 support-parent add/sub conservation inherited from sealed M1597.
- Formal all-population prefix bound.
- Full final accumulator range.
- Full integer final population.
- Full operator-output population.
- Operator-local FP32-versus-QDQ metrics.
- Required operator-bridge outputs complete under the M2043r2 contract.
- Sampled independent integer oracle.
- Sampled observed hardware-order prefix.

The following remain explicitly false and must not be inferred:

- valid825 AEE or end-task accuracy preservation;
- downstream ATLIF equivalence;
- whole-network hardware-order equivalence;
- full observed hardware-order prefix population;
- hardware cycles or hardware speedup;
- energy, power, area, timing, or PPA;
- system speedup;
- an unrestricted paper-ready system result.

## Residual P2 observation

The sole P2 is governance provenance: the independent source review happened before and authorized the production run, but its durable double-sealed receipt was persisted after the production result and is therefore not reverse-pinned inside the already frozen result. The exact producer and execution-contract SHA256 values in that receipt match the result and close the current chain. A successor should include the source-review receipt SHA256 directly in its execution authority and production result.

This does not alter the numerical result or expand its claim boundary.

## Citable sentence

> Across 160 frozen ep34 C1 and decoder calls comprising 3.272 billion outputs, the operator-local INT8 QDQ bridge achieves 0.009106 MAE, 0.012407 RMSE, 0.239204 maximum absolute error, and 0.9999789 cosine similarity; all 1,280 sampled integer-oracle probes match, while the full observed accumulator range and the 200,219 formal prefix bound fit signed Acc24.

## Final disposition

**PASS_M2043R2_EP34_S40_OPERATOR_BRIDGE_RESULT_HAMMER, 99/100, P0/P1/P2 = 0/0/1.**
