# M979 source-only UCLI controller. The first run stops exactly on accepted
# header; the second run stops one clock after accepted token_done. This makes
# the DUT-only SAIF duration equal to measured_cycles*3ns and excludes reset,
# pre-header idle, inter-case idle, testbench, and external memory models.
power -gate_level all mda sv
power tb_m979_c2_three_axis_mapped_gate_case_saif.dut
run
power -enable
run
power -disable
power -report $::env(M979_SAIF_FILE) 1e-9 tb_m979_c2_three_axis_mapped_gate_case_saif.dut
quit
