# M2149 causal preflight: observe reset and preload before clearing history.
power tb_m2149_m2018_ordinary_single_axis_native_saif_preflight.dut_ordinary
power -enable
puts "M2149_UCLI_PHASE order=1 action=power_enable timing=before_first_run scope=single_ordinary_dut"
run
puts "M2149_UCLI_PHASE order=2 action=run_reset_and_preload observer_enabled=1"
puts "M2149_UCLI_PHASE order=3 action=first_stop_reached internal_census_preceded_stop=1"
power -reset
puts "M2149_UCLI_PHASE order=4 action=power_reset timing=after_first_stop_before_measurement_run"
run
puts "M2149_UCLI_PHASE order=5 action=second_stop_reached exact_window_complete=1"
power -disable
puts "M2149_UCLI_PHASE order=6 action=power_disable timing=before_report"
power -report $::env(M2149_RTL_SAIF_FILE) 1e-9 tb_m2149_m2018_ordinary_single_axis_native_saif_preflight.dut_ordinary
puts "M2149_UCLI_PHASE order=7 action=power_report scope=single_ordinary_dut"
quit
