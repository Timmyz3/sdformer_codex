# Frozen UCLI protocol preflight.  It exercises the same power commands used
# by M979 and writes only the tiny DUT hierarchy to the caller-selected SAIF.
power -gate_level all mda sv
power tb_m1044_vcs_ucli_power_saif_preflight.dut
run
power -enable
run
power -disable
power -report $::env(M1044_PREFLIGHT_SAIF) 1e-9 tb_m1044_vcs_ucli_power_saif_preflight.dut
quit
