# M1111D decoder-only production runner author receipt

## Verdict

`PASS_M1111D_DECODER_RUNNER_AUTHOR_SOURCE__DIFFERENT_AUTHOR_FINAL_HAMMER_REQUIRED`

Score: **97/100**. P0=0, P1=0, P2=1. The P2 is the intentionally pending
different-author runner/launch hammer; this author receipt does not authorize
production.

## Frozen runner

The zero-argument runner derives every path from its own canonical location and
pins M1105Dr2 source/contract/author receipt, the M1110D GO authority, M672 and
M670r2 mapper sources, Python 3.10.18 and `docs/359`. It accepts no path,
configuration, output or authority from argv/environment. Runtime requires
isolated Python and replaces the caller environment with six fixed variables.

The ordered production boundary is identity/freshness/resource validation,
unique lock, one atomic attempt consumption, M1105Dr2 canonical preflight,
one 120-call streaming schedule, atomic seal/no-replace publication. A caught
post-attempt failure moves any partial work into an atomically sealed quarantine.
The attempt remains consumed and there is no retry.

## Production artifact boundary

If a future independent hammer authorizes the exact runner, its only output is:

- exactly 120 D0-D3 calls in the frozen three-sequence, ten-sample order;
- address-timed execution of input-descriptor read, weight read, psum read,
  compute, psum write and output commit under one 96-lane Acc24, 3 ns,
  245,760-byte resource;
- one compressed exact per-call transaction/schedule receipt with address,
  dependency and schedule digests plus kind-level first/last timestamps;
- diagnostic cycle and traffic totals only.

No performance ratio is calculated. `speedup_admitted`,
`system_speedup_admitted`, and `paper_ppa_ready` are false in call rows,
aggregate result and publish gate.

D1 consumes the exact scaled-binary bitpack with θ word `1065353139`
(`b3ff7f3f` little-endian), never coerces θ to one and never folds θ into the
weights. M700 is not loaded or accepted as an input. Any final checkpoint
change requires rebind of activity, D1 θ, weights, numeric miters, transaction
population, cycles, traffic, energy and system table.

## Author-stage evidence

The static self-test validates all authority pins and schedules one synthetic
dependency chain containing all six transaction kinds. It confirms the D1 θ
marker, no-fold policy and false performance claims. AST checks find exactly
one consume-attempt, production, publish and quarantine call in the required
order.

The runner `main()` and M1105Dr2 `build_canonical()` were not called. No M699
payload was opened, and no M1111D attempt, result, lock, work or quarantine was
created. The next action is a different-author final runner hammer which must
attack the exact runner/contract/receipt tuple before providing any launch
command.

`docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
