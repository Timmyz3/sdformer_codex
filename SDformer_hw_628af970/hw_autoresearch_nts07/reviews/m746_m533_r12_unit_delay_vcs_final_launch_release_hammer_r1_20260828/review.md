# M746/M533 r12 UNIT_DELAY VCS final-release hammer

## Verdict

**PASS — 100/100, P0/P1/P2 = 0/0/0.**

The exact final release at SHA-256 `15822880094ef6bd6b7b3a2efb3d35511609486666b27d1c75a0cd8a4b9dc817` closes the M749 100/100 source/candidate audit, both runner-consumed M746 reviews, source contract, launch-now-false candidate, r11 consumed failure, M738/M741/M743/M744 causal-repair lineage, frozen RTL/TB/SVA/macro identities, foundry assets, VCS binary, and resource policy without mismatch. Every required member manifest and outer seal verifies.

## Authorization

Exactly one invocation of the immutable r12 runner is authorized now, subject to the runner repeating its collision, memory, commit-headroom, swap, cgroup failcnt, under-OOM and OOM-kill gates. The unique attempt is consumed only by atomic creation of `results/m746_m533_m528_dead_write_only_1rw_unit_delay_vcs_r12_20260828`.

The released command is:

```bash
env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_vcs_m746_m533_m528_dead_write_only_1rw_unit_delay_r12_exact_sha.sh
```

No other VCS identity, simv rerun, Icarus/Verilator run, DC/Formality/PT/PTPX run, CPU/GPU experiment, or network/remote job is authorized by this review.

## Functional/timing boundary

The runner compiles the checksum-identical foundry SRAM Verilog with exactly one `+define+UNIT_DELAY`; neither `+notimingcheck` nor `+no_notifier` is present. TB r7 is SHA `d194f912...` and the runner requires R7 PASS/COVERAGE, independent direct-forward and macro-response RAW recovery minima, six protocol attacks, clean scoreboard/SVA/error/fatal gates, and double-sealed terminal receipts.

This remains a functional-VCS-only attempt. Functional correctness is not established until it passes, and macro timing is never established by UNIT_DELAY. Macro-inclusive slow-DB DC/PT setup/hold remains separate.

## Fresh live gate and attempt state

Three fresh two-second-spaced samples passed: minimum MemAvailable `416103316 KiB`, SwapFree `56656636 KiB`, and commit headroom `91083276 KiB`, each above the immutable thresholds. Session/user cgroup-v1 failcnt, under-OOM, and OOM-kill were all zero. A fresh `/proc` scan found zero same-UID Synopsys/VCS/simv collisions.

The r12 result directory and declared attempt sentinel were absent at review time. The sealed r11 result remains `FAILED_DO_NOT_CITE`; M738 classifies its fatal as a TB oracle false positive without verifying C1 RTL. `docs/359_DATE终局冻结_20260813.md` remains at SHA-256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

This hammer ran no runner, VCS, simv, HDL compiler, EDA tool, CPU/GPU experiment, or remote job and modified no author source identity.
