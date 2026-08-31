# M236 independent-review handoff

Review these strict boundaries and artifacts:

1. Contract: `contracts/m236_dynamic_bn_lut16_newton2_coefficient_engine_contract_r1_20260825.json`.
2. Full vector generator and payload: `system_simulator/scripts/generate_m236_h67_lut16_newton2_full_vectors.py` and `results/m236_h67_lut16_newton2_full_vectors_r1_20260825/manifest.sha256`.
3. RTL/SVA/TB: `rtl_m236/`, `verif_m236/`, and `tb_m236/`.
4. Exact VCS receipt and seal: `results/m236_dynamic_bn_lut16_newton2_full220800_vcs_r1_exact_20260825/m236_vcs_receipt_r1.txt`, `RUN_COMPLETE.txt`, and `SHA256SUMS`.
5. DC runner/TCL/filelist: `dc_handoff/scripts/run_dc_m236_dynamic_bn_lut16_newton2_logic_only.sh`, its `.tcl`, and `dc_handoff/filelists/date_m236_dynamic_bn_lut16_newton2_coefficient_engine_rtl.f`.
6. DC receipt and evidence: `dc_handoff/runs/m236_dynamic_bn_lut16_newton2_logic_only_dc_3p000ns_r1_20260825/RUN_COMPLETE.txt` and `evidence_manifest.sha256`.
7. Matched baseline: `dc_handoff/runs/m235r2_synthesis_safe_logic_only_dc_3p000ns_r1_20260825/RUN_COMPLETE.txt` and `evidence_manifest.sha256`.
8. M234 review origin: `results/m234_independent_hammer_review_r1_20260825/SHA256SUMS`.

The reviewer should independently reload M233 NPZ, recompute all 220,800
16+2 integer outputs, run fresh VCS, inspect the single source multiply
operator and all eight FSM use states, verify the six tail extrema, and check
that the DC comparison does not become an area, energy, event or system claim.
