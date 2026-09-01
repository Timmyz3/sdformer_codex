# Exact mapped-DUT-only activity window.  Two public warmup tasks complete
# before this script reaches its first stop.  Activity starts immediately
# before measured epoch5945 prep row0 and stops immediately after its task_done.
power -gate_level all mda sv
power tb_m1772_c1_m1701_two_bank_public_warmup_energy.dut
run
power -enable
run
power -disable
power -report $::env(M1772_SAIF_FILE) 1e-9 tb_m1772_c1_m1701_two_bank_public_warmup_energy.dut
quit
