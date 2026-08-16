# DATE Algorithm Review Remediation (2026-07-17)

## Review Outcome

Independent algorithm-side review: **Weak Reject, confidence 4/5**. The main concern is experimental attribution and protocol rigor, not the current H67 accuracy result.

## Mandatory Actions

| priority | reviewer concern | corrective experiment or artifact | status |
|---:|---|---|---|
| P0 | H67 lacks an equal-budget H60/no-motion control | H81: same TTX epoch2 start, full30 schedule, checkpoints and valid825; only Motion-XOR weight changes `1/4 -> 0` | generated and queued after H66 |
| P0 | Paper AAE and local AAE are conflated | preserve legacy AAE-2D; add DSEC/Barron AE-3D and same-checkpoint NB0/H67/H81 audit | implemented; audit queued with H81 |
| P0 | valid825 is used for search and final claims | final candidate must be submitted to official DSEC test; if submission is unavailable, add a frozen MVSEC train-to-test protocol and label it external validation | protocol pending final candidate |
| P1 | single-seed evidence | retrain final top candidate and equal-budget control with at least three fixed seeds; report mean, standard deviation, and paired deltas | opt-in seed path and generator implemented; launch after top model freezes |
| P1 | mechanism attribution is incomplete | report H60/no-motion, H67 Motion-XOR, H70 event-density OR, and a correspondence-destroying XOR control under equal budget | H60/H67/H70 available; shuffle control pending H81 result |
| P1 | novelty can be read as a heuristic | add per-flow-magnitude and event-density error analysis; connect Motion-XOR to temporal event correspondence | metric audit first; stratified inference pending |
| P1 | energy excludes attention/control/memory | keep current number labeled `spike_energy_proxy`; add attention-inclusive operation and memory accounting before paper freeze | hardware/software accounting in progress |

## Decision Gates

1. H67 remains provisional until H81 completes. The algorithmic gain is `H67 - H81`, not `H67 - short-finetuned H60`.
2. Do not launch three-seed repeats for every candidate. Freeze the top candidate after H66/H73-H80, then repeat only the winner, H81, and NB0 if a reproducible NB0 training seed is required.
3. Official DSEC test is the primary external validation route. MVSEC train-to-test is the fallback/second dataset and must use a frozen public sequence split without MDR pretraining.
4. DATE tables must separate legacy local AAE-2D from benchmark AE-3D and separate spike proxy energy from complete estimated energy.

## Reproducibility Entry Point

`runtime.seed` and `runtime.deterministic` are optional and leave all old configs unchanged. After the top model freezes, generate the three repeat configs with `make_reviewer_seed_configs.py`; do not spend three-seed budget on every screened candidate.

## MVSEC Fallback

The frozen same-domain split and the `window=9` geometry constraint are documented in `neuron_autoresearch/MVSEC_TRAIN_TEST_PROTOCOL_20260717.md`. Data are ready, but launch is gated on the final DSEC candidate so the second-dataset budget is not spent on a provisional architecture.
