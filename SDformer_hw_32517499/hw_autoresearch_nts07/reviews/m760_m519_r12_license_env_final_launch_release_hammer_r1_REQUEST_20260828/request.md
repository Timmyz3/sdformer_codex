# Fresh M519 R12 license-environment final-release hammer request

Review the exact double-sealed `launch_now=true` R12 release and its complete candidate/M759/M752/R11 lineage without running the runner, any EDA tool, or a license-server query.

The command in `request.json` is only the object under review. This request authorizes zero EDA attempts and zero license queries. Publish it only if the fresh review scores 100/100 with P0=P1=P2=0, all exact hashes and seals pass, the canonical R12 result and attempt identities remain absent, and the live shared-host resource/collision gate is green. The runner itself must perform the status-only Design-Compiler and DC-Ultra license gate before it may publish the unique attempt sentinel.

Required output: `reviews/m761_m519_r12_license_env_final_launch_release_hammer_r1_20260828`.
