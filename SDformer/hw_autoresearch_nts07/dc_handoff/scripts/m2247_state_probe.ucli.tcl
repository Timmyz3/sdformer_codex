power -gate_level all mda sv
power tb_m2217_m2018_tsbg_matched_native_saif_power.state_power_probe
power -enable
run
power -disable
power -report $::env(M2247_STATE_SAIF).prehistory.saif 1e-12 tb_m2217_m2018_tsbg_matched_native_saif_power.state_power_probe
power -reset
power -enable
run
power -disable
power -report $::env(M2247_STATE_SAIF) 1e-12 tb_m2217_m2018_tsbg_matched_native_saif_power.state_power_probe
quit
