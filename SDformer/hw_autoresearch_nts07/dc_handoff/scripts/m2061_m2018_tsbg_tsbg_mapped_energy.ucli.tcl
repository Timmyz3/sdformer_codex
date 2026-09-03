power -gate_level all mda sv
power tb_m2061_m2018_tsbg_matched_mapped_energy.core.dut_tsbg.g_mapped.mapped_implementation
run
power -enable
run
power -disable
power -report $::env(M2061_SAIF_FILE) 1e-9 tb_m2061_m2018_tsbg_matched_mapped_energy.core.dut_tsbg.g_mapped.mapped_implementation
run
quit
