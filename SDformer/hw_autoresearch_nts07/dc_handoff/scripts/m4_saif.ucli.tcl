# Direct VCS SAIF collection avoids an impractically large RTL VCD on the
# 12,288-bit response interface.  Reset is released at 22.5 ns and the first
# descriptor is driven at 24 ns, so enabling at 23 ns excludes reset without
# dropping workload activity.
# The DUT is SystemVerilog (`logic` plus unpacked arrays), so `sv` is
# required in addition to `mda`.  Without it VCS can silently emit a tiny
# SAIF containing only a testbench task variable while omitting DUT ports.
power -gate_level all mda sv
power tb_qfit_dual_line_descriptor_resident_real.dut
run 23ns
power -enable
run
power -disable
power -report $::env(M4_SAIF_FILE) 1e-9 tb_qfit_dual_line_descriptor_resident_real.dut
quit
