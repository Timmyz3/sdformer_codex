# M2112 independent result hammer: M2067 R10 FC2 continuation VCS

Verdict: **PASS (99/100; P0/P1/P2 = 0/0/1).** The R10 result admits a scoped component-VCS cycle statement and expands fixed-protocol FC2 layer-identity coverage from 4/12 to 12/12. It does not admit full-token/full-FC, real-weight, same-area, power, energy, or system-speedup claims.

The canonical result directory is regular, non-symlink, exhaustive, and double sealed. Its four sealed evidence members and outer seal verify. The consumed attempt directory is likewise exhaustive and double sealed; its PID 1372000 is dead, its nonce matches `result.json`, `automatic_retry=false`, and both attempt and result report zero inherited logs. No R10 failure or private work/stage/failure namespace remains.

The source and launch chain is exact. The M2101 contract and all 17 frozen inventory members verify; the M2102 source review is exhaustively double sealed; the M2103 one-shot release is double sealed; and the M2104 release hammer is exhaustively double sealed. The observed runner, parser, contract, source-review, release, predecessor-failure, and attempt hashes match the identities recorded in `result.json`. `docs/359_DATE终局冻结_20260813.md` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

The execution evidence contains one compiler header, one inline compile/link, one simulation runtime header, and one simulation report. The result reports exactly one VCS compile and one simv run; the single simv batch contains slots 0 through 959 once each. Compile error records and simulation error/fatal records are zero. The independently verified attempt budget is one compile, one simv, zero retries, and zero inherited logs.

Independent parsing reconstructs the complete workload Cartesian product: 40 samples, four sequences, eight previously unsupported FC2 layer identities, and first/middle/last fixed B4 token regions, for 960 unique workloads. Six layer identities use G96 continuation with two chunks at bases 0 and 48; two use G192 continuation with four chunks at bases 0, 48, 96, and 144. All first/intermediate/final chunk flags and all output-tile/chunk Cartesian records match the frozen fixture.

The transcript contains 960 reset markers, 960 individual PASS records, 960 final ledgers, 13,440 row-chunk records, and exactly one final batch PASS. Both axes complete 115,200 commits and 1,843,200 integer checks. Ordinary and TSBG address checks total 12,313,344 and 5,693,568. Oracle mismatch and Acc24 overflow totals are zero. The batch also executes 960 G96 and 960 G192 alias attacks; ordinary and TSBG each reject all 1,920 attacks.

The independently summed cycle result is:

| Axis | Cycles |
|---|---:|
| Ordinary | 80,129,099 |
| Continuation-aware TSBG | 45,381,069 |
| Weighted component ratio | 1.7656943912x |
| Time reduction | 43.3651% |

Of 960 workloads, 554 improve, 404 tie, and two regress by only one and seven cycles; the worst ratio is 0.9998735x. These tails do not alter the aggregate claim and must not be hidden if a per-workload distribution is shown.

The prior independently reviewed population covers four G<=48 FC2 identities; R10 covers the eight disjoint G96/G192 identities. Therefore **12/12 FC2 layer identities are now exercised under the fixed three-region component protocol**. This is layer-identity coverage, not exhaustive token coverage or a complete FC inference wall time.

Permitted paper language:

> Across eight previously unsupported G96/G192 FC2 layer identities (40 ep34 samples and three fixed token regions; 960 component workloads), continuation-aware TSBG reduced ordinary post-load RTL cycles from 80.129M to 45.381M (1.7657x), while 1.843M integer checks per axis and 115,200 commits per axis completed with zero mismatch or overflow. Combined with the prior four G<=48 identities, the fixed-protocol validation covers all 12 FC2 layer identities.

Mandatory adjacent boundary: activity/sign descriptors are real ep34, but weights are deterministic directed INT8 verification values. The result is a component VCS-cycle measurement over fixed token regions, not full-token/full-FC wall time, real-checkpoint-weight inference, same-area performance, power, energy, or full-network/system speedup. It must not be multiplied with C1, C3, Prosperity, Phi, or another component factor.

P2: use “12/12 layer identities under the fixed three-region protocol,” never “all FC2 inference.”

This review performed only read-only file, SHA-256, JSON, transcript, fixture, and process-presence checks. It ran no VCS, simv, DC, PT, license query, GPU workload, or remote job, and modified no result, source, or frozen document.
