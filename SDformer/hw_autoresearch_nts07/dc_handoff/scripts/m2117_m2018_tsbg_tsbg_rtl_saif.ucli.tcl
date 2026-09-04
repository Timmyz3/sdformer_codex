power -gate_level all mda sv
power tb_m2117_m2018_tsbg_rtl_saifmap_power.core.dut_tsbg.implementation
run
power -enable
run
power -disable
power -report $::env(M2117_RTL_SAIF_FILE) 1e-9 tb_m2117_m2018_tsbg_rtl_saifmap_power.core.dut_tsbg.implementation
quit
