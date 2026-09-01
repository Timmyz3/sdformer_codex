power -gate_level all mda sv
power tb_m1790_c3_m1454_fixed_t10_mapped_energy.dut
run
power -enable
run
power -disable
power -report $::env(M1798_SAIF_FILE) 1e-9 tb_m1790_c3_m1454_fixed_t10_mapped_energy.dut
quit
