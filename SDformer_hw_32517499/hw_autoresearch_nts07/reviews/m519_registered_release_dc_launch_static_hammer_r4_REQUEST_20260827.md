# M519 registered-release three-axis DC launch static hammer r4 — REQUEST

Date: 2026-08-27  
Execution boundary: read-only review. Do not run VCS, DC, Formality, PT, PTPX, or open-source EDA.

## Subject

- DC runner: `dc_handoff/scripts/run_dc_m519_fc2_registered_release_three_axis_exact_sha.sh`
- Exact runner SHA256: `7d4049dbf21ea6850776ca47b66634da996600fd98c7b6f09e6762aba033278a`
- DC Tcl SHA256: `591d791a8691d099a21e1e43c253ecd202c9c9091454a0b36b6b66785322929c`
- Frozen recovery contract r3 SHA256: `ed2c22ebb94bcdd8860340c81482170ed73768b6ad59322dcca80356d2552b36`
- VCS receipt SHA256: `7228d99fc3384fc2ee77e6fddbd1ca7e0df88870847c8a1c3525583df66627a8`
- VCS result outer-seal file SHA256: `fdc5002e8c34674d0f598161235ac4b4e534063ec26f66139b578410cf9f4ba7`
- Prior static hammer r3 outer-seal file SHA256: `37ef79c5ff984f61f347845595e8e04b9fe4739ab731097633f5b574a4cbd47e`
- Receipt-blind VCS hammer r2 outer-seal file SHA256: `b9a566af7e5c429e72d63453ba047c3d95f9b2efe9fce8b2c122afa78e4e090a`
- M496 loop-failure hammer outer-seal file SHA256: `c8e49b3aeb1406c103604d6fec23e48ff27682f58eaed0e9abdd5b2cae6b3b79`
- Frozen docs/359 SHA256: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

## Narrow repair under review

The runner's two references to the nonexistent
`reviews/m519_registered_release_vcs_hammer_r2_20260827/` were changed to the
sealed actual directory
`reviews/m519_registered_release_vcs_receipt_hammer_r2_20260827/`.
No RTL, SVA, testbench, filelist, SDC, Tcl, result, or frozen recovery contract
was changed.

## Required verdict

Independently verify all exact identities, double seals, topology and finite
JSON; reconstruct the M519 registered-release loop break; verify precompile
TIM-209/OPT-150 hard failure, one-attempt atomic publish, resource/process
collision gates, all three K1/K8/K1x8 points, five constraint classes, pin/top
identity, and fail-closed behavior. Confirm that the runner now consumes the
actual sealed VCS receipt review and that no stale nonexistent path remains.

If P0 is zero, create and double-seal a review, then prepare (but do not
execute) `contracts/m519_fc2_registered_release_dc_launch_admission_r2_20260827.json`
with status `AUTHORIZED_ONE_M519_DC_ATTEMPT`, binding the exact runner and all
final identities above. The admission must keep `paper_ppa_ready=false`,
`system_speedup=false`, and `headline=false`. The caller must still pin both
the runner SHA and the final admission SHA.

