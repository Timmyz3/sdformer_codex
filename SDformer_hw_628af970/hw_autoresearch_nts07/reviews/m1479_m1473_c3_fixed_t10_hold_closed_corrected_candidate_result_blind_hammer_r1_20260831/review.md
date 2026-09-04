# M1479 independent result hammer of the M1473 corrected C3 candidate

Verdict: **PASS (100/100; P0/P1/P2 = 0; 108/108 mutations rejected).**

I independently verified commit `62b55e5e3b58d000670b217c183daa24381e57f6`, the exact M1473 receipt, the M1454 DC, M1456 PrimeTime, and M1457 Formality run seals, and the exact M1471 failure seal and commit binding. I also reverified the M917 and failed M1288 comparison-source seals. No EDA, VCS, simulation, GPU, remote, power, or performance run was launched.

The M1473 correction is exact and unique. The sealed M1454 hold report contains 100 reported paths. Its first and minimum slack is `+0.034585 ns`; `+0.044297 ns` occurs on seven later paths and is not WNS. Comparing the 26 common raw numeric fields in M1470 and M1473 finds no other changed value. The three raw-run trees are unchanged between the M1470 and M1473 commits.

The unchanged evidence also reparses:

- M1454 DC: area `63756.125879 um^2`, zero macros, setup/hold WNS `+0.000300/+0.034585 ns`, 100 paths per check and zero violations.
- M1456 PrimeTime: setup/hold WNS `+0.000299/+0.030474 ns`, zero violations and zero unconstrained diagnostics. Coverage is `127700 = 64394 met + 0 violated + 63306 untested`; the untested reasons are exactly `63042 constant_disabled + 264 no_paths`.
- M1457 Formality: gate-to-gate M917 mapped versus M1454 mapped only, `verify_return=1`, 11,180 passing points, and zero failing, unmatched, unverified, or aborted points. This is not direct RTL-to-M1454 proof.
- Comparison: M917 area `62433.503388 um^2` to M1454 `63756.125879 um^2` is exactly `2.118449901458219%` overhead. The old `-0.022628 ns` hold value remains sourced from a sealed `FAILED_OR_INCOMPLETE_DO_NOT_CITE` M1288 quarantine and is diagnostic only; the improvement to independent PT `+0.030474 ns` is `+0.053102 ns`.

The M1471 failure is preserved, not erased: its exact commit, review, mechanical check, Markdown review, completion marker, manifest, outer seal, and failure status all match M1473's bindings. M1470 remains non-admitted.

M1473 is admitted only as corrected C3 component evidence under the frozen boundaries: TSMC 28-nm standard-cell, prelayout, logic-only, 3.000 ns, ideal clock, ZeroWireload, no SPEF, zero macros, and no physical interconnect. It does not support post-layout or macro-inclusive PPA, power, energy, throughput, speedup, system-speedup, paper-PPA-ready, headline, or direct RTL-Formality claims.
