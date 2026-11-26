# R28 fixed Q4 recovery: CPU preparation only

This implements the single branch in [the pre-quality plan](../ENTRY_PRE_QUALITY_V1.md).
Nothing in this directory has launched a GPU experiment or read R27 development
quality. Root checks the actual R27 execution audit, stable native recovery,
Q8's two-reference quality qualification, and positive frozen cost allowance
`B = 0.9 C_Q8strong - C_Q4` before launch. Root records those source hashes and the
decision in `round28/LAUNCH_DECISION_V1.json`; this implementation adds no new
automatic readiness or approval system.

The independent new core differs from R27 only in its declared low mode Q4 and
four-bit arithmetic truncation: low planes0–3 execute on true tail rows only,
planes4–7 execute on all rows, with sign-plane coefficient−128. The dense integer
reference and input-gradient effective weights use the same signed drop4 rule.
Values such as−127 become−128; there is no reclamp or offset correction. Weight
STE, real integer forward, chunk/padding rules, all-T/channel/spatial support,
native BN/PSN, parameter objects, quantization scale and all477 training scope
are unchanged. The core retains Q8 solely as a mathematical/reference helper;
the smoke and formal command lines accept only `--arm Q4`.

The membership is R27's frozen256 train /64 development /512 ordered crops,
SHA `40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b`.
Both smoke and formal start from original W271482 with fresh AdamW. Smoke performs
one update and five full-model calls, including post-update full-resolution
STE/deploy/dense-integer bytewise checks. It never supplies formal weights or
optimizer. Formal uses512 updates, the same loss/teacher/LR/seed/eval-with-grad
recipe as R27,128 development predictions, and three final numerical diagnostic
calls:1155 total model forwards. The60-minute formal wall cap, shared single-GPU
lock, no development-GT loading, fixed final checkpoint, and source hashing are
retained. All earlier experiments remain immutable.

The output interfaces follow R27 except for new identities:

- smoke schema: `flow_r28_one_step_smoke_v1`;
- formal schema: `flow_r28_bounded_recovery_v1`;
- graph: `flow_r28_all12mlp_patch4conv_integer_Q4_complete9_tailQ8_all78current_v1`;
- arm: `Q4`; final status: `FIXED512_AND128_DEVELOPMENT_PREDICTIONS_COMPLETE`;
- quality artifacts: `evaluations.initial/endpoint.frames`,64 each, FP32 head3
  `[2,480,640]` pixel-unit NPZ plus all-four-head hashes and true execution records;
-512-step ledger, original/teacher/final state digests, actual optimizer state,
 24 no-gradient names, final parity checks and weights retain their R27 fields.

CPU preparation passed18 checks: six integer/STE/real original MLP+ResBlock
checks, four fixed-contract comparisons, three smoke contracts, five formal
orchestration checks. The latter include64 synthetic output artifacts and actual
hook cleanup; they are not full-backbone numerical tests. The real native-module
micro-tests use eight local module forwards and no real GT/GPU. Signed extremes,
low-four-plane tail-only work, actual weight/input gradients, pixel-unit loss,
event-only development access and the exact512 order are covered. R27's three
frozen source hashes and unchanged recipe/STE adapter ASTs were also checked.

On the existing A100 tree, the commands for root use:

```bash
/data/flow_a/envs/v2/bin/python -B /data/flow_a/code/research_20261019/round28/implementation/run_smoke.py \
  --r26-root /data/flow_a/code/research_20261017/window_precision_round26 \
  --members /data/flow_a/code/research_20261019/round27/independent_review/membership_v1.json \
  --runner-sha256 6a30a057b9cb3ca3d6ea157be57627526361243ca719752b04aacb15822ad2c5 \
  --arm Q4 --output FRESH_SMOKE_DIR --preflight-output FRESH_SMOKE_PREFLIGHT_JSON --wall-minutes 15

/data/flow_a/envs/v2/bin/python -B /data/flow_a/code/research_20261019/round28/implementation/run_recovery.py \
  --r26-root /data/flow_a/code/research_20261017/window_precision_round26 \
  --members /data/flow_a/code/research_20261019/round27/independent_review/membership_v1.json \
  --runner-sha256 a4c3d06c7267dd66ff6bdf30cec1dbf3094d81351f2ed9bcfb2ff59a709e4d60 \
  --arm Q4 --output FRESH_FORMAL_DIR --preflight-output FRESH_FORMAL_PREFLIGHT_JSON --wall-minutes 60
```

Without `--execute` these run CPU/source preflight only. Root adds it only after
the prescribed evidence checks and engineering smoke, with an external hard
timeout and fresh output path. No such command was executed here.

Q4 failure would apply only to this fixed scope and bounded recovery contract.
Even a successful Q4 recovery would establish an action baseline, not a flow
innovation. Future mixed-mode signed gains must be measured with the same
Q4-recovered weights; independently recovered Q8 weights cannot stand in for
those interventions. Restoring a tile to Q8 is not assumed to improve AEE
monotonically. No selector, mode search or automatic further run is included.
