# M2160 observes reset/preload, reports that prehistory, then clears it.
power -gate_level all mda sv
power tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.dut_ordinary
power -enable
puts "M2160_UCLI_PHASE order=1 action=power_enable timing=before_first_run scope=single_ordinary_dut"
run
puts "M2160_UCLI_PHASE order=2 action=first_run_returned census_and_begin_preceded_return=1"
power -disable
puts "M2160_UCLI_PHASE order=3 action=prehistory_power_disable timing=before_diagnostic_report"
power -report $::env(M2160_PREHISTORY_SAIF_FILE) 1e-9 tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.dut_ordinary
puts "M2160_UCLI_PHASE order=4 action=prehistory_power_report scope=single_ordinary_dut diagnostic_only=1"
power -reset
puts "M2160_UCLI_PHASE order=5 action=power_reset_requested timing=after_prehistory_report_before_measurement_enable"
power -enable
puts "M2160_UCLI_PHASE order=6 action=measurement_power_enable timing=after_reset_before_second_run"
run
puts "M2160_UCLI_PHASE order=7 action=second_run_returned end_and_pass_preceded_return=1"
power -disable
puts "M2160_UCLI_PHASE order=8 action=measurement_power_disable timing=before_measurement_report"
power -report $::env(M2160_MEASUREMENT_SAIF_FILE) 1e-9 tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.dut_ordinary
puts "M2160_UCLI_PHASE order=9 action=measurement_power_report scope=single_ordinary_dut admitted_candidate=1"
quit
