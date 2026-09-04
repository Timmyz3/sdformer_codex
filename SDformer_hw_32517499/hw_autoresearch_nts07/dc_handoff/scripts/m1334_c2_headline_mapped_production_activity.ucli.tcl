# M1334 captures only the mapped DUT cone.  The checker parses active Tcl
# commands, so commented copies cannot satisfy or redirect this scope.
power -gate_level all mda sv
power tb_m1334_c2_headline_mapped_production_activity.core.dut
run
power -enable
run
power -disable
power -report $::env(M1334_SAIF_FILE) 1e-9 tb_m1334_c2_headline_mapped_production_activity.core.dut
quit
