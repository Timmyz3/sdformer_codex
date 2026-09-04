# Direct activity capture on the mapped gate instance only.  Reset is released
# at 15 ns; the frozen workload marker rises at 21 ns and the first config
# drive begins at 22 ns.  This preserves the M425R4 measurement boundary.
power -gate_level all mda sv
power tb_m425_h67_balanced_selected_slice_direct_saif.dut.u_gate
run 21.5ns
power -enable
run
power -disable
power -report $::env(M438_GATE_SAIF_FILE) 1e-9 tb_m425_h67_balanced_selected_slice_direct_saif.dut.u_gate
quit
