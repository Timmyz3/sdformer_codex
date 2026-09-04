power -gate_level all mda sv
power tb_m2113_m2018_tsbg_rtl_saifmap_power.core.dut_base.implementation
run
power -enable
run
power -disable
power -report $::env(M2113_RTL_SAIF_FILE) 1e-9 tb_m2113_m2018_tsbg_rtl_saifmap_power.core.dut_base.implementation
quit
