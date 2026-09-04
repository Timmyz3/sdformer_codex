# Fresh M746/M533 r12 final-release hammer request

Review the exact double-sealed `launch_now=true` foundry-`UNIT_DELAY` functional release and its complete M749 plus runner-consumed M746 lineage without running the runner, VCS, simv, any other HDL/EDA tool, CPU/GPU experiment, or remote job.

The command in `request.json` is only the object under review. This request authorizes zero executions. Publish it only if the fresh review scores 100/100 with P0=P1=P2=0, every exact hash and double seal passes, the r12 result and attempt identities remain absent, and the live resource/collision gate is green.

The reviewer must publish the runner-consumed attestation at `reviews/m746_m533_r12_unit_delay_vcs_final_launch_release_hammer_r1_20260828/review.json` and double-seal that package. A separate master review may be added, but it cannot substitute for the fixed-path attestation.
