# M528 single-port same-ledger static admission hammer r1

## Verdict

**FAIL / NO-GO for the current admission.** This was a source-only review. I did not execute the runner, the CPU recompute, EDA, GPU, or RTL, and I did not modify the admission or `docs/359`.

The admission JSON itself has the requested schema and strict authorization values: one CPU run, zero EDA, zero GPU, and `rtl=false`. Its identities and both seal layers validate. However, the exact pinned runner has a deterministic terminal-path P0, so those otherwise-correct authorization fields must not be exercised.

## P0: successful analysis must still exit nonzero

The runner creates three files directly in `${m528_work}`:

- `resource_preflight.log` at lines 140–142;
- `production_stdout.log` at lines 157–162;
- `production_stderr.log` at lines 157–162.

Lines 172–174 **copy** those files into `${m528_work}/result`; they do not remove the originals. Line 189 moves only the `result` subdirectory to the canonical path. Line 190 then executes `rmdir "${m528_work}"`, although the three original log files are still in that directory.

Therefore `rmdir` must fail under `set -e`. The EXIT trap runs while `m528_complete=0`, marks and quarantines the leftover work directory, and preserves the nonzero exit. At that point the attempt sentinel has already been consumed and the canonical path already exists, so the exact runner cannot be legally rerun. Lines 191–193, including the PASS echo, are unreachable.

This is a release blocker, not a cosmetic cleanup issue. The safe correction is a new runner revision that moves the logs into `result` (or removes the originals before `rmdir`), followed by a new source hammer and a new admission pinned to the new SHA. This sealed admission must not be edited or reused.

## Identity and seal checks that passed

| Item | Live SHA-256 / result |
|---|---|
| Admission | `96832f878b6be79dbc342aeb1758ed7deaca09d618f283e72df13ed0bc08f8d7` |
| Admission double seal | PASS |
| Runner | `a31d891ab83a8c87fa98f31cabbc7a81174362ef9b4f469fe0a3220b80711531` |
| Analyzer | `c611f8c98253e44ccf93743d47476da0adc9835b013b247bc4e2d821953afb8a` |
| Execution contract | `910c804a9a9df13395ab4f6b2ef5988ea0dee56ab7e52a21f887fa8fe0d73a34` |
| Governing contract | `d0e3728f3a9991cf97c6af88181cd51996e457228b120f2b706a8986caf9ca51` |
| Author outer-seal file | `9c29e7950b1d6563e78004acac54a858fe8d8821e784500ff8f9cabbe2d4521a` |
| Prior static review | score 98, P0=0, P1=0; both seals PASS |
| `docs/359` | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` |

All admission identity fields match their live files. The prior review basis, author handoff, and governing review seals also validate. This new P0 is specifically in the runner terminal cleanup sequence that the prior static review marked as passing.

## Exact current caller identity — DO NOT EXECUTE

The following is the exact environment the current runner expects, and all three values satisfy its path/SHA/jq identity checks. It is recorded only to remove ambiguity; this review explicitly does **not** authorize executing it:

```bash
M528_EXPECTED_STATIC_ADMISSION_PATH=contracts/m528_single_port_same_ledger_static_admission_r1_20260827.json \
M528_EXPECTED_STATIC_ADMISSION_SHA256=96832f878b6be79dbc342aeb1758ed7deaca09d618f283e72df13ed0bc08f8d7 \
M528_EXPECTED_RUNNER_SHA256=a31d891ab83a8c87fa98f31cabbc7a81174362ef9b4f469fe0a3220b80711531
```

No command follows intentionally. A corrected runner will necessarily have a different runner SHA and therefore requires a new admission.

## Non-authorizing runtime snapshot

At `2026-08-27T17:44:45+08:00`, canonical and attempt paths were absent and the listed local EDA/simulation process counts were zero. `MemAvailable` and swap passed, OOM counters were clean, but commit headroom was only **56,253,960 KiB**, below the required **67,108,864 KiB**. Thus the resource gate was closed independently of the P0.

This snapshot grants no permission to bypass the threshold. Even if resources later pass, the current runner/admission remain rejected because of the deterministic terminal-path bug.
