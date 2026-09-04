# M2217 single-DUT native RTL SAIF.  This is reused byte-identically by all
# six axis/window runs; output names are supplied by the one-shot runner.
power -gate_level all mda sv
power tb_m2217_m2018_tsbg_matched_native_saif_power.dut_axis
power -enable
puts "M2217_UCLI_PHASE order=1 action=power_enable_before_first_run scope=single_dut_axis"
run
puts "M2217_UCLI_PHASE order=2 action=first_run_returned window_begin_preceded_return=1"
power -disable
power -report $::env(M2217_PREHISTORY_SAIF_FILE) 1e-9 tb_m2217_m2018_tsbg_matched_native_saif_power.dut_axis
puts "M2217_UCLI_PHASE order=3 action=diagnostic_prehistory_reported never_annotate=1"
power -reset
puts "M2217_UCLI_PHASE order=4 action=power_reset_requested before_measurement=1"
power -enable
puts "M2217_UCLI_PHASE order=5 action=measurement_enable after_reset=1"
run
puts "M2217_UCLI_PHASE order=6 action=second_run_returned window_end_preceded_return=1"
power -disable
power -report $::env(M2217_MEASUREMENT_SAIF_FILE) 1e-9 tb_m2217_m2018_tsbg_matched_native_saif_power.dut_axis
puts "M2217_UCLI_PHASE order=7 action=measurement_reported scope=single_dut_axis"
quit
