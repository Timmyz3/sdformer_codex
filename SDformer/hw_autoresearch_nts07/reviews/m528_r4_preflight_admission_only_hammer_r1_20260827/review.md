# M528 r4 preflight admission-only hammer r1

## Verdict

PASS, 98/100, P0/P1/P2 = 0/0/2.

Root is authorized to execute exactly one non-production r4 preflight suite using preflight runner SHA `893a89c98ae3ea04fc1c316e71c3768fe5189cc4ce54527e352f0c6f3b3a0944` and admission SHA `e95f5711be9a26aa6b375e715f43744d2d8e97e544aea4f04c6a323c126617d7`. This review does not authorize CPU production, EDA, GPU, RTL, paper admission, or a performance claim.

## Integrity and identity

The admission passes the pinned strict duplicate-key JSON parser. Its canonical member sidecar records the admission JSON under its own basename, and the outer sidecar records the member sidecar under its own basename. Both checks pass. The admission binds the reviewed analyzer, preflight runner, production runner, execution contract, strict parser, PyTorch 3.10 Python, author handoff outer seal, static-review JSON, and static-review outer seal at the exact requested SHA-256 values.

The author handoff and static review each pass both seal layers. `docs/359_DATE终局冻结_20260813.md` remains SHA-256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Semantic authorization

The admission authorizes one preflight suite and zero production/EDA/GPU runs, with RTL false. Its expected behavior is one worker, three schema cases, worker initialization called, worker phase not called, no row replay, no production result, and no production attempt.

The sealed static review is unwithdrawn and has the exact PASS status and verdict with P0=P1=0. Its five live source identities, author seal, and preflight-only authorization match the admission. It explicitly denies production admission.

The preflight runner does not trust seal existence alone. Before any preflight action, it requires a caller-pinned runner SHA and admission path/SHA, strict-parses the admission, verifies its two seal layers, directly parses schema/status/authorization and expected token/workers/case count, then directly parses the exact static-review PASS/P0/P1/identity/authorization payload. It also re-establishes the r3 and r2 NO-GO boundaries. The actual preflight-only analyzer arguments and three cases are hardcoded in the byte-pinned runner.

## Non-blocking observations

1. The preflight runner cross-binds the admitted production-runner SHA through the exact sealed static review rather than hashing that production runner directly during this preflight.
2. It parses token/workers/schema count directly from the admission, while the remaining true/false expectations are enforced by pinned source behavior and the future receipt checks. Directly parsing all fields would improve uniformity.

Neither observation can broaden this admission or create a production path. An independent receipt hammer remains mandatory after the one permitted preflight.

## Review boundary

This was an admission-only review. It did not execute the analyzer, schema smoke, spawn self-test, preflight runner, production runner, CPU production, EDA, GPU, or RTL. At review time, both r4 preflight and r4 production canonical/attempt paths were absent.
