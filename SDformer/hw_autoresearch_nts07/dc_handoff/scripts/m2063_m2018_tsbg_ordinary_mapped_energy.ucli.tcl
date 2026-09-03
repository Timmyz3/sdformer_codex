power -gate_level all mda sv
power tb_m2063_m2018_tsbg_matched_mapped_energy.core.dut_base.g_mapped.mapped_implementation
run
power -enable
run
power -disable
power -report $::env(M2063_SAIF_FILE) 1e-9 tb_m2063_m2018_tsbg_matched_mapped_energy.core.dut_base.g_mapped.mapped_implementation
run
quit
