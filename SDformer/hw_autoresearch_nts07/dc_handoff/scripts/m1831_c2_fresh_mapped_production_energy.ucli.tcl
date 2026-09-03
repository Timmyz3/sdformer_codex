# DUT-only gate-level SAIF window.  The simulation-only compatibility shell is
# outside the scope; the implementation child is the exact derived mapped top.
power -gate_level all mda sv
power tb_m1831_c2_fresh_mapped_production_energy.core.dut.implementation
run
power -enable
run
power -disable
power -report $::env(M1831_SAIF_FILE) 1e-9 tb_m1831_c2_fresh_mapped_production_energy.core.dut.implementation
quit
