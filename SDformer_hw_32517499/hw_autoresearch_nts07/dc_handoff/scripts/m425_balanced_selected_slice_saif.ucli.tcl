# Reset is deasserted at 15 ns.  The testbench raises its frozen measurement
# marker at 21 ns and does not drive the first configuration until 22 ns.
# Enabling at 21.5 ns therefore excludes reset while retaining all workload
# transactions and the final two quiescent clocks.
power -gate_level all mda sv
power tb_m425_h67_balanced_selected_slice_direct_saif.dut
run 21.5ns
power -enable
run
power -disable
power -report $::env(M425_SAIF_FILE) 1e-9 tb_m425_h67_balanced_selected_slice_direct_saif.dut
quit
