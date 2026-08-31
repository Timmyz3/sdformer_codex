# M536 / M533-r3 functional-VCS launch-admission hammer request

Perform a fresh, read-only hammer of the sole launch-admission candidate at `contracts/m533_m528_dead_write_only_1rw_vcs_launch_admission_r1_20260827.json`.

No VCS, Icarus, Verilator, DC, Formality, PT, PTPX, CPU/GPU experiment, remote job, or result-directory creation is authorized. The reviewer must recompute every SHA, verify both seals, reject duplicate/non-standard JSON, and prove that the frozen runner consumes this exact path, schema, status, ten-key authorization object, and identity.

The already sealed M533 source-static hammer is the only prerequisite and must remain exactly PASS 100/100 with P0/P1/P2 equal to 0/0/0. The exact future result directory must remain absent.

At authoring time M519 is still running under UID `zhumd` as PIDs 4165439, 4165666 (`common_shell_exec -shell dc_shell`), and 4165667. Therefore `launch_now=false`. Any same-UID `dc_shell`, `dc_shell-t`, Synopsys `common_shell_exec`, `vcs`, or `simv` match must block launch. A fresh resource snapshot and fail-closed decision are also required immediately before result creation.

The frozen runner validates the admission but does not parse `launch_now`, the future hammer verdict, or the external collision/resource decision. This gap is deliberately disclosed. If an operator-only gate cannot be accepted as fail-closed, record a blocking P0/P1 and require a separately admitted wrapper/runner revision; do not waive the gap and do not run anything.

A PASS requires P0=0 and P1=0, a scored verdict, a member manifest, and an outer seal. Even a PASS does not allow launch until root confirms M519 and every collision are gone and the resource gate passes. Any future run needs a separate post-run receipt hammer and cannot establish trace recurrence, speedup, PPA, energy, full-network performance, or a paper headline.
