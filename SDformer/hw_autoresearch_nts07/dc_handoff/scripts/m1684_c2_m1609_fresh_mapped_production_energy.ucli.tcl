# Exact DUT-only direct gate-SAIF window.  M979 stops on accepted header and
# one edge after accepted token_done, excluding reset/testbench/memory activity.
power -gate_level all mda sv
power tb_m1684_c2_m1609_fresh_mapped_production_energy.core.dut
run
power -enable
run
power -disable
power -report $::env(M1684_SAIF_FILE) 1e-9 tb_m1684_c2_m1609_fresh_mapped_production_energy.core.dut
quit
