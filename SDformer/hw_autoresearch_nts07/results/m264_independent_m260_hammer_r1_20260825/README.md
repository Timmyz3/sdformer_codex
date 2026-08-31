# M264 independent hammer review of M260

Verdict: **the M260 evidence is reproducible, but the current M235 Q16
candidate fails its paired first-ten downstream gate and must not advance to a
valid825 admission run.** This is a negative accuracy result, not a system,
speedup, power, PPA, headline, or standalone-contribution result.

## Independently verified

- The 11-entry remote absolute-path manifest relocates safely. A clean replay
  under a different temporary `SDformer` root regenerated the sealed analysis
  byte-for-byte (`461f4b9d...`).
- One altered reference CSV value failed with `remote payload SHA drift`; one
  altered evaluator source failed with `M260 frozen source identity drift`.
- The paired rows are exactly the first ten ordered entries of frozen DSEC
  `valid_split_seq.csv`, all unique and all from `zurich_city_09_a`.
- The evaluator selects the exact 12 FFN BN1 and 12 FFN BN2 underlying
  `norm_layer` modules. The forward hook recomputes moments over T,N,H,W with
  `unbiased=False` and returns Q16 `alpha*x+offset`, which replaces the selected
  BN output. PyTorch still owns moment finalization.
- Aggregate execution is consistent with 24 hooks/sample, 240 hooks, 220,800
  coefficient pairs, and 4,377,600,000 output values. All seven rail counts are
  zero. The receipt does not contain per-module call counters.

## Accuracy decision

| Metric | Reference | M235 Q16 | Relative delta | Candidate wins/losses |
|---|---:|---:|---:|---:|
| AEE | 0.9501519361 | 0.9665978201 | +1.730869% | 3 / 7 |
| DSEC Fl | 2.3105949034 | 2.4698466338 | +6.892239% | 4 / 6 |
| Spikes/frame | 101,920,461.1 | 101,932,344.7 | +0.011660% | 6 / 4 |

The tiny aggregate spike change does not make the approximation safe: spatial
and recurrent event flips propagate into materially worse flow. The current
Q16 configuration remains blocked. Because the ten frames are contiguous and
come from one sequence, this is a conservative screening decision rather than
a claim that the full 825-frame population is proven worse.

## Next bounded DSE

Keep the checkpoint, reference, first-ten order, current-batch moments, and all
non-FFN-BN operators frozen. Try at most four candidates: Q18 alpha/offset;
Q18 invstd/alpha/offset; the same with two Newton steps; and a selectively
widened or exact-bypass configuration based on per-module event flips. Add
per-module hook/error/event-flip receipts, pre-register the screening limits,
then use a disjoint ten-frame holdout before promoting one candidate to
valid825. If none passes, move to a separately checkpointed exact-recurrence
QAT line.

The full score, P0/P1/P2 findings, algorithm feedback, and DSE limits are in
`m264_independent_hammer_review_r1.json`. Independent calculations are in
`m264_independent_recompute_r1.json`; clean replay and SHA fault evidence are
in `m264_clean_replay_and_wrong_sha_receipt_r1.json`.

`docs/359_DATE终局冻结_20260813.md` was not modified and remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
