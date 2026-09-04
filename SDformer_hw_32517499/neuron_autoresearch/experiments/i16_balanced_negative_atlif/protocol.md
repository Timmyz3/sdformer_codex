# I16 Balanced Negative ATLIF Autopilot

Goal: repair the H9a ternary ATLIF path where negative spikes were almost
disabled by `negative_threshold_scale: 30.0`.

The experiment keeps the H9a replacement logic:

- Q/K attention neurons use ternary PSN + ATLIF.
- H9a selected FFN/downsample high-SOP modules use binary PSN + ATLIF.
- Shiftmax/BSA compatibility attention remains enabled.
- Training starts from the same baseline checkpoint used by H9a-style runs.
- All parameters are trainable.

Iteration plan:

1. Short sweep over Q/K negative trigger scale: 1, 2, 4, 8.
2. Promote the two best short runs into a second short run with small angular
   supervision (`lambda_ang=0.1`).
3. Select one full-run candidate by AEE, AAE, SOPs, firing rate, and negative
   spike participation.
4. Launch a 30-epoch H9a-style full run and profile epoch 29 automatically.

Selection is intentionally conservative. A candidate is penalized if the quick
profile has very poor AEE/AAE, if firing/SOPs become dense, or if negative
spikes remain near zero. This is an exploratory gate, not the final paper
metric.

Outputs:

- `generated_configs/`: configs used for each run.
- `results/*/train.log`: training logs.
- `results/*/profile/sops_summary.json`: quick/full profile metrics.
- `results/trajectory.csv`: compact comparison table.
- `results/autopilot_summary.json`: selected candidate and final status.

