power -gate_level all mda sv
power tb_m2125_m2018_tsbg_rtl_saif_window_diagnostic.core.dut_tsbg.implementation
run
power -enable
run
power -disable
power -report $::env(M2125_RTL_SAIF_FILE) 1e-9 tb_m2125_m2018_tsbg_rtl_saif_window_diagnostic.core.dut_tsbg.implementation
quit
