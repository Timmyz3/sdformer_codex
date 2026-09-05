# Direct gate-net activity avoids stale RTL-to-gate maps after clock insertion.
power -gate_level all mda sv
power tb_m2217_m2018_tsbg_matched_native_saif_power.dut_axis
power -enable
run
power -disable
power -report $::env(M2256_OUTPUT)/prehistory.saif 1e-12 tb_m2217_m2018_tsbg_matched_native_saif_power.dut_axis
power -reset
power -enable
run
power -disable
power -report $::env(M2256_OUTPUT)/activity.saif 1e-12 tb_m2217_m2018_tsbg_matched_native_saif_power.dut_axis
quit
