# M522/M514 pre-tool failure independent hammer r1

Verdict: **the failed invocation did not consume the positive DC authorization. Exactly one retry of runner `58ae904ed019d27690544c474da1df03fd4b3eb69752d3f5e78fae21a1a6402f` is authorized.**

## What failed

The reported invocation passed the historical M514 VCS and independent-review `sha256sum -c` checks, then stopped at `actual == set(expected)` inside `m522_verify_sealed_dir`. That assertion is a pre-tool exact-topology check. The traceback does not identify which directory argument was being checked because the helper prints no root marker.

An independent current replay finds exact equality and valid double seals for all three possible roots: M514 VCS 94/94 members, M514 independent review 3/3, and M522 r3 static review 3/3. All are free of symlinks. No persistent integrity defect remains.

## Race assessment

The historical VCS and M514 review packages were already immutable. The M522 r3 review was assembled from 13:48:14.713 to 13:48:38.459, with its member manifest appearing at 13:48:28.568 and outer seal at 13:48:38.459. If the failed invocation overlapped this interval, a review-package publication race is the only mutable-directory explanation consistent with the current clean replay.

This race is plausible but not uniquely proven: the invocation left no timestamped pre-tool log, and the assertion itself lacks the verified root and expected/actual inventory. The report therefore does not claim more than the evidence supports.

## Why no DC authorization was consumed

Every call to the failing verifier precedes resource admission, staging creation, and the resolved `snps_shell -f` command. The latest possible failing verifier occurs at runner byte offset 7955; resource admission begins at 9344, staging at 9791, and DC at 12924. An assertion terminates the `set -e` runner before those points.

The workspace contains no M522 canonical output, staging directory, quarantine, `dc.log`, `dc.rc`, DC receipt, or active DC/`snps_shell` process. Process accounting is not available as historical proof, but the control-flow proof already makes positive DC invocation unreachable after the reported assertion.

## Retry decision

The exact runner, contract, Tcl, r3 static review, and frozen inputs have not changed. The r3 static review remains double-sealed and all prerequisite directories now pass the exact verifier. Therefore no r4 binding is required for this failure alone, and exactly one positive retry of the same runner SHA is authorized after this package is double-sealed.

If the same pre-tool assertion repeats, stop without another retry. The next step would then be an r4 runner that records the verified root and expected/actual inventories before failing, plus a new exact-SHA static binding.

The retry remains limited to standalone M514 logic-only 3 ns DC/STA for additive decoder-support area/timing. Any successful output still requires an independent receipt-blind DC hammer. No performance, energy, system-speedup, or paper-ready PPA claim is admitted here.

No runner or EDA tool was executed in this audit. Production files and `docs/359` were not modified; `docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
