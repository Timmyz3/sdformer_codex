# H64: Offline-Centered Symmetric ATLIF Protocol

Status: completed negative at valid1; no training promotion
Scope: DSEC only, all 105 ATLIF wrappers

## Hypothesis

The failed H63 runs show 38%-42% negative firing after switching existing weights to zero-centered symmetric thresholds. The hypothesis is that the trained PSN membrane distributions are offset, so symmetry must be defined around a fixed calibrated center rather than numerical zero:

```text
positive event: h >= c_t + theta
negative event: h <= c_t - theta
```

`c_t` is a per-module, per-timestep descriptor. The same scalar `theta` is used on both sides, so the neuron remains strictly symmetric. At inference the descriptors are constants; there is no target-rate feedback or online adaptation.

## Calibration

Run the original converged one-sided TTX checkpoint on one DSEC validation sample. For each ATLIF wrapper:

1. estimate `c_t = median(h_t)` from a bounded reservoir;
2. measure the original one-sided event budget `r = P(h >= theta_old)`;
3. set `theta_sym = quantile(|h-c_t|, 1-r)`;
4. save all 105 `center` and `thresh` tensors in a new state-dict checkpoint.

This is a deterministic checkpoint conversion, not a hyperparameter sweep.

## Controlled evaluation

1. H64-ref: centered symmetric ternary ATLIF + existing uniform all12 H60/TX. This isolates neuron feasibility; it may use K only as a diagnostic reference.
2. H64-STC: the same checkpoint + all12 raw signed-TX token/channel direct Shiftmax, with no gate×K.

Both first run valid1 with load audit `210/0/0`. H64-STC may train for 20 steps only if firing is below 20% or H64-ref proves that the centered neuron itself is stable. No G4/G8 is allowed before STC passes.

## Hardware mapping

The comparator bounds are precomputed as `lo_t=c_t-theta` and `hi_t=c_t+theta`. RTL therefore needs two signed threshold descriptors and two comparators; no runtime subtractor, estimator, or feedback controller is required. Event SRAM remains `(valid, sign)` for signed ternary.
