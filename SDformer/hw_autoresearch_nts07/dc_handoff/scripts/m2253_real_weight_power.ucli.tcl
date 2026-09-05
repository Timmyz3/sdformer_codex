# Record unchanged post-load boundaries with real-weight sensitivity inputs.
power -gate_level all mda sv
power tb_m2217_m2018_tsbg_matched_native_saif_power.dut_axis
power tb_m2217_m2018_tsbg_matched_native_saif_power.state_power_probe
power -enable
run
power -disable
power -report $::env(M2253_OUTPUT)/prehistory.saif 1e-9 tb_m2217_m2018_tsbg_matched_native_saif_power.dut_axis
power -report $::env(M2253_OUTPUT)/state_prehistory.saif 1e-12 tb_m2217_m2018_tsbg_matched_native_saif_power.state_power_probe
power -reset
power -enable
run
power -disable
power -report $::env(M2253_OUTPUT)/activity.saif 1e-9 tb_m2217_m2018_tsbg_matched_native_saif_power.dut_axis
power -report $::env(M2253_OUTPUT)/state.saif 1e-12 tb_m2217_m2018_tsbg_matched_native_saif_power.state_power_probe
quit
