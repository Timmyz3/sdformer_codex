# I17 Negative Rate Feedback

I16 showed that lowering `negative_threshold_scale` can create negative spikes
in short probes, but full training drives negative firing back to almost zero.
That means a fixed negative trigger scale is not enough.

I17 adds a closed-loop negative-rate controller to the experiment-local H9
ATLIF ternary node:

- The positive/negative output magnitude remains `+thresh` and `-thresh`.
- `thresh` still follows the ATLIF threshold-growth path.
- `negative_threshold_scale` becomes a mutable trigger-distance controller.
- If observed `neg_mean` falls below `negative_target_rate`, the trigger scale
  decreases, making negative spikes easier.
- If observed `neg_mean` is too high, the trigger scale increases, restoring
  sparsity.

The experiment keeps H9a's replacement logic and starts from the same PSN
baseline checkpoint. It runs:

1. Guard sweeps over negative target rate and controller strength. The stable
   starting point is H9a's `negative_threshold_scale=30`, not the dense I16
   low-scale regime.
2. Three-epoch medium checks for the best two guards.
3. One 30-epoch full run only after the medium check keeps negative spikes.
4. Valid40 profile for the final checkpoint.

Success criteria:

- `neg_mean` after medium/full should stay above 0.002 without driving guard
  AEE into the dense negative-spike failure mode.
- SOPs should stay close to H9a, ideally below the PSN baseline.
- AEE/AAE should not regress more than I16.
