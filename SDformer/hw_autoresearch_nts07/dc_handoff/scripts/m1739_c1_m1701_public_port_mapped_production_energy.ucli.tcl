# Exact mapped-DUT-only activity window.  The testbench stops immediately
# before the first public prep acceptance and immediately after task_done.
power -gate_level all mda sv
power tb_m1739_c1_m1701_public_port_mapped_production_energy.dut
run
power -enable
run
power -disable
power -report $::env(M1739_SAIF_FILE) 1e-9 tb_m1739_c1_m1701_public_port_mapped_production_energy.dut
quit
