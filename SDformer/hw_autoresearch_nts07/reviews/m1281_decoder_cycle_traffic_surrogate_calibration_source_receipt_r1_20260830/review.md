# M1281｜decoder cycle/traffic surrogate calibration source receipt

## Outcome

**PASS source-only synthetic framework.** The additive script implements the
requested decoder surrogate and a post-seal calibration interface without
opening the growing M1111DR2 work prefix or any canonical decoder result.

The frozen equations are:

- cycles: `4 * group_count + layer_constant`;
- descriptor traffic: `16 B * group_count`;
- weight traffic: `16 B * active_source_terms`;
- psum read: `288 B * group_count`;
- compute count: `group_count`;
- psum write: `288 B * group_count`;
- output commit: `288 B per call`.

The fitter keeps slope 4 fixed and fits one arithmetic-mean residual constant
for each of D0/D1/D2/D3. It reports per-layer and global mean/max absolute and
relative error. A future analytical-cycle annex remains forbidden unless a
separately hammered 120/120 sealed result is supplied and the global/per-layer
maximum relative error is at most `0.001` (0.1%).

## Synthetic tests

Python 3.6.8 compile passed. Six fixture unit tests passed:

1. exact 120-call fixture fit;
2. fixture cannot enable a real analytical annex;
3. noisy fixture misses the 0.1% gate;
4. all six traffic equations conserve exactly;
5. unsealed input is rejected;
6. claim promotion and the full mutation suite are rejected.

The standalone fail-closed self-test rejected 15 attacks: 119 calls, ordinal
drift, layer-order drift, descriptor/weight/psum-read/compute/psum-write/commit
drift, missing seal, missing hammer, claim promotion, negative group count,
duplicate JSON and nonfinite JSON. A no-argument CLI invocation also failed
closed; `--self-test` is the only permitted CLI mode.

The exact fixture uses synthetic layer constants D0/D1/D2/D3 =
`17/23/31/41` and produces zero fitting error. These are test vectors, not H67
measurements. The fixture's `error_gate_pass=true` is kept separate from
`analytical_cycle_annex_allowed=false`.

## Future one-shot calibration interface

A future adapter may import `calibrate_payload` only after independently
verifying:

- exactly 120 sealed M1111DR2 calls and a passing result hammer;
- contiguous ordinals with 30 calls per layer in D0/D1/D2/D3 order;
- exact group/source-term and traffic conservation for every call;
- no final-checkpoint identity change, or a complete rebind if it changed.

The adapter then fits all four constants once, reports mean/max error, and may
set the analytical-cycle annex flag only if maximum relative error is ≤0.1%.
Failing the gate retains diagnostic traffic only and admits no cycle annex.

## Identity and boundary

- source SHA256:
  `098d7c0e96df18ed9eda2f43e26230b86ba5afbef3975c46d695ec8953e7a4ce`
- unit-test SHA256:
  `c812b11c05d4fc00b30b4d029686e0d245aaefafb27ca1135c11fca78c14f170`
- contract SHA256:
  `829a0766f1d79a8acfdade0fd42853f445699e533b9ab918c745e8bc460501f9`
- docs359 SHA256:
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

Every output is `calibration_only=true`, `system_speedup_admitted=false` and
`paper_ppa_ready=false`. No real decoder cycles, traffic, speedup, energy or
PPA are published. No EDA, GPU or remote task was run, and no live work-prefix
member was read.
