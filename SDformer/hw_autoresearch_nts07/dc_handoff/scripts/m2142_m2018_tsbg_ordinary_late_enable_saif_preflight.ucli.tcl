# M2142 ordinary-only causal preflight.  The monitor is enabled before the
# first run, so reset and the 383-cycle preload establish observer state.  The
# first run stops only after the TB has printed its five-family knownness
# census.  Reset then clears activity history without modifying DUT state.
power -gate_level all mda sv
power tb_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight.core.dut_base.implementation
puts "M2142_UCLI_PHASE order=1 action=power_enable timing=before_first_run scope=ordinary_implementation"
power -enable
puts "M2142_UCLI_PHASE order=2 action=run_reset_and_preload observer_enabled=1"
run
puts "M2142_UCLI_PHASE order=3 action=first_stop_reached internal_census_preceded_stop=1"
power -reset
puts "M2142_UCLI_PHASE order=4 action=power_reset timing=after_first_stop_before_measurement_run"
run
puts "M2142_UCLI_PHASE order=5 action=second_stop_reached exact_window_complete=1"
power -disable
puts "M2142_UCLI_PHASE order=6 action=power_disable timing=before_report"
power -report $::env(M2142_RTL_SAIF_FILE) 1e-9 tb_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight.core.dut_base.implementation
puts "M2142_UCLI_PHASE order=7 action=power_report scope=ordinary_implementation"
quit
