# M283 independent hammer of M280 PAFT near-match DSE

Score: **91/100**. Severity: **P0=0, P1=3, P2=4**.

Verdict: **GO for the scoped frozen-trace opportunity and INT8 pre-scale
accumulator-error DSE; NOGO for lossy-threshold accuracy, hardware, system, or
headline promotion.**

The independent reviewer imports none of the M280, M251, or M43 producer
analyzers. It rehashes and decodes all 40 M248 raw packed payloads, reconstructs
51,840,000 partitions, applies the M77 train-only catalog, reads all four M256
weights in declared I-KY-KX-O order, and independently reimplements work,
wide/shared cycles, nearest-center selection, and occurrence-weighted INT8
accumulator deltas. Every author field and the tau-zero M251 miter match exactly.

The important boundary is architectural: tau 2/3/4 exceed 2x only on the
specialized WIDE144 PWP service. SHARED96 peaks at 1.7164x. Tau 2 is only
2.000245x and has 33,304 candidate-cycle headroom to the exact 2x boundary,
equivalent to 1.927 cycles per phase. No lossy threshold has modified-forward
or paired valid825 accuracy.

Tie-breaking is not cosmetic. There are 38,903,617 weighted vectors with
multiple nearest centers, and catalog-first would choose a different center for
28,730,267. The independent minimum-packed-uint16 implementation matches M280;
future RTL must prove the same rule.

Replay from the hardware root after moving or removing only a previous
`clean_replay` subdirectory:

```bash
bash results/m283_m280_paft_near_match_independent_hammer_r1_20260825/run_clean_replay_m283.sh
```

The launcher includes a corrupted-author-result negative test and requires it
to fail before performing the clean independent reconstruction. It does not use
any open-source RTL tool.
