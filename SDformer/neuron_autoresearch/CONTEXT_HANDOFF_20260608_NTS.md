# Context Handoff 2026-06-08

1. Current active full run:
`neuron_experiments/H9_bipolar_self_attention/results/nts09a_hw_h60_freeze816_s1224_steps1224_auto_full_bs6_20260608_210900_setsid`

2. Current active config:
`neuron_experiments/H9_bipolar_self_attention/configs/nts09a_hw_h60_freeze816_s1224_steps1224_auto_full_20260608_210900.yml`

3. NTS08c status: **invalid interrupted full** (only ep0-3 weights, no state_dict; not formal evidence)

4. NTS09 priority short test: **completed** (`results/nts09_priority_20260608_200402`)
   - valid40 winner: `nts09a` (AEE 1.4658, AAE 13.57)
   - full30 promoted with `skip_state_save=false`
   - valid825 watcher running for ep19/24/29

4. NTS08 design:
H60, no carrier, no Kmag, no target-rate, `mismatch_penalty=0`, `single_active_penalty=0`, `bipolar_mu=0.05`, qk threshold capped with `max_threshold=1.15`, FFN official ATLIF kept but `threshold_eta=0` and `activity_eta=0`.

5. Why NTS08 exists:
NTS07b full30 was the current strongest hardware-friendly line, but qk ternary threshold kept rising late in training. Epoch24 was better on AEE, epoch29 was better on AAE/SOPs/firing. NTS08 tries to stop the late oversparsification without adding deployment operators.

6. Current best completed full run:
`nts07b_hw_h60_ffn_update0_act0_s1224_steps1224_auto_full_bs6_20260608_042113_setsid`

7. NTS07b valid825 highlights:
- epoch24: AEE `1.4793`, AAE `9.9090`, SOPs `3.5396G`, firing `0.08373`
- epoch29: AEE `1.4850`, AAE `9.7361`, SOPs `3.3576G`, firing `0.07942`

8. NTS08 short-test winner:
`nts08c_hw_h60_qk_cap115_s1224`

9. NTS08 valid40 short-test:
AEE `1.4511`, AAE `13.5908`, PE1 `0.5607`, PE2 `0.2168`, PE3 `0.0861`, SOPs `4.0573G`, firing `0.09597`

10. Current live training health:
latest observed region around step `1120` showed:
- step `1020`: `threshold_mean=1.02027`, `threshold_max=1.07529`, `ternary_activity_mean=0.06956`, `binary_activity_mean=0.05232`
- step `1060`: `threshold_mean=1.02050`, `threshold_max=1.07613`, `ternary_activity_mean=0.06916`, `binary_activity_mean=0.05192`
- step `1120`: `threshold_mean=1.02789`, `threshold_max=1.10290`, `ternary_activity_mean=0.06687`, `binary_activity_mean=0.05532`
No collapse signal so far; the qk cap line is still healthy enough to finish full30 before deciding whether to promote or stop it.

10b. Newer NTS08c evidence:
- `checkpoint_epoch3.pth` has been written.
- epoch3 validation loss: `1.3093095123767853`
- near epoch3 end:
  - step `1200`: `threshold_mean=1.02834`, `threshold_max=1.10453`, `ternary_activity_mean=0.06756`, `binary_activity_mean=0.05548`
  - step `1220`: `threshold_mean=1.02846`, `threshold_max=1.10494`, `ternary_activity_mean=0.06625`, `binary_activity_mean=0.05430`
- Interpretation: this is not a collapse trajectory; the cap line is still behaving like a plausible NTS07b successor rather than another over-sparsified dead end.

10c. Epoch4-in-progress evidence:
- The run is not stalled after epoch3; `train.log` shows the next epoch already advancing past the midpoint.
- Observed online points in the current epoch:
  - step `580`: `threshold_mean=1.03175`, `threshold_max=1.11676`, `ternary_activity_mean=0.06594`, `binary_activity_mean=0.05125`
  - step `640`: `threshold_mean=1.03208`, `threshold_max=1.11796`, `ternary_activity_mean=0.06554`, `binary_activity_mean=0.05435`
  - step `700`: `threshold_mean=1.03242`, `threshold_max=1.11916`, `ternary_activity_mean=0.06572`, `binary_activity_mean=0.05319`
- Interpretation: `NTS08c` remains active and healthy in epoch4 mid-run; sparsity has not collapsed, so this branch is still worth waiting for full-run evidence.

10d. More precise epoch4 continuation:
- Later online reads confirm the run is around `step ~708/1224` in the current epoch, i.e. already in epoch4 mid/late rather than merely just after epoch3.
- Across the observed online window, the trend is still stable:
  - `threshold_mean` roughly `1.025 -> 1.032`
  - `threshold_max` roughly `1.093 -> 1.119`
  - `ternary_activity_mean` roughly `6.5% -> 6.8%`
  - `binary_activity_mean` roughly `5.0% -> 5.6%`
- Interpretation: the cap line is not surviving only by hard-clipping; it is still training in a healthy regime and has not drifted into the late-run oversparse failure mode yet.

11. MD status:
All NTS04-NTS08 runs should be tracked in `neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md`. Added the previously omitted duplicate short-driver dirs for NTS04 and NTS05, and clarified that `nts04_223442` is a brief early summary while `nts05_014233` is effectively an empty starter dir with no formal summary artifact.

12. Next actions after NTS08 full finishes:
- run standardized profile/eval
- write full metrics into `EXPERIMENT_REDESIGN_PLAN.md`
- compare directly against NTS07b epoch24 and epoch29
- if NTS08 is not better, stop qk-threshold-cap line

13. Chosen mainline story for now:
Default DATE-facing direction is the most hardware-friendly one:
- keep deployment formula fixed to `h60` score-level fusion
- keep `no carrier`, `no Kmag`, `no target-rate`, `mismatch_penalty=0`, `single_active_penalty=0`
- keep FFN official ATLIF present but not allowed to collapse full training
- only change training-time qk threshold dynamics next (`cap`, `freeze`, or schedule)

14. Immediate next candidate priority if NTS08 loses:
1. `NTS09a` (`freeze_after_step=816`)
2. `NTS09d` (`cap115 + freeze816`)
3. `NTS09b` (`freeze_after_step=918`)
4. `NTS09c` (`eta0325 + freeze816`)

15. New operational helper added:
- `neuron_experiments/H9_bipolar_self_attention/entrypoints/wait_full_then_run_standard_valid825.py`
- Purpose: wait until a full run produces target checkpoints (default `19/24/29`), then call `run_h9_standard_valid825_eval.py`
- Verified with `py_compile` and `--help`
- Live watcher is currently running for `nts08c` and waiting for epochs `19/24/29`

16. New NTS09 priority launcher:
- `neuron_experiments/H9_bipolar_self_attention/entrypoints/run_nts09_priority_short.sh`
- Same four NTS09 candidates, but ordered as `09a -> 09d -> 09b -> 09c`
- Uses tighter promotion gates: `AEE<=1.75`, `AAE<=16.0`, `SOPs<=6.0G`
- Verified with `bash -n`

17. Current nts08c timing estimate:
- observed checkpoints:
  - epoch0 `2026-06-08T08:21:05Z`
  - epoch1 `2026-06-08T08:36:48Z`
  - epoch2 `2026-06-08T08:52:30Z`
  - epoch3 `2026-06-08T09:08:08Z`
- average pace: about `941s/epoch` (`15m41s`)
- rough ETA:
  - epoch9 `2026-06-08T10:42:14Z`
  - epoch19 `2026-06-08T13:19:04Z`
  - epoch24 `2026-06-08T14:37:29Z`
  - epoch29 `2026-06-08T15:55:54Z`

18. New nts08c interruption finding:
- The training process is no longer alive; only the `wait_full_then_run_standard_valid825.py` watcher remains.
- `train.log` proves the run advanced into the next epoch up to roughly `step708/1224`, so this was not an early-shape failure.
- But the run dir still contains only `checkpoint_epoch0..3.pth`; no `checkpoint_epoch4.pth` or later validation was written.
- No clear traceback / OOM line was found in `train.log`; interruption looks external rather than a deterministic model-code crash.

19. Why nts08c cannot be cleanly resumed:
- Baseline training does support `--resume`, but it requires the paired local training state file `checkpoint_epoch*_state_dict.pth`.
- This run has none of those files because the generated full config carried `runtime.skip_state_save: true`.
- Therefore `nts08c` cannot be treated as a clean resumable full30. A continuation from `checkpoint_epoch3.pth` would be a weight-only restart and must be documented as such.

20. Fix already applied for future full promotions:
- `neuron_experiments/H9_bipolar_self_attention/entrypoints/promote_best_rapid_screen.py` was updated so future generated full configs use `runtime.skip_state_save: false`.
- This preserves optimizer / scheduler / scaler state and keeps future interrupted full runs properly resumable.

21. NTS09 priority short screen has now been started:
- result dir: `neuron_experiments/H9_bipolar_self_attention/results/nts09_priority_20260608_172619`
- launcher: `neuron_experiments/H9_bipolar_self_attention/entrypoints/run_nts09_priority_short.sh`
- candidate order:
  1. `nts09a_hw_h60_freeze816_s1224`
  2. `nts09d_hw_h60_cap115_freeze816_s1224`
  3. `nts09b_hw_h60_freeze918_s1224`
  4. `nts09c_hw_h60_eta0325_freeze816_s1224`
- promotion gates:
  - `AEE <= 1.75`
  - `AAE <= 16.0`
  - `SOPs <= 6.0G`
