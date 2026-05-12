# E5b Official-style Ternary Spike

This experiment replaces the earlier simplified `exp_tsn` node with the
official Ternary-Spike activation and membrane update from:

`https://github.com/yfguo91/Ternary-Spike`

Official source commit:

`2aca58747f01d7960cb6f0284665bbb353d35aab`

The experiment keeps all baseline SDFormerFlow files read-only. Runtime patches
are contained in this experiment directory.

Official-style choices:

- `spike_activation(mem / V_th)` with ternary forward states `{-1, 0, 1}`.
- Official membrane update: `mem = mem * decay + input`, reset by
  `mem * (1 - abs(spike))`.
- `decay = 0.25`, `V_th = 1.0`, `temp = 3.0`, `fire_ratio = 1`.
- SGD with momentum and official `split_weights` decay/no-decay grouping.
- Cosine LR schedule.
