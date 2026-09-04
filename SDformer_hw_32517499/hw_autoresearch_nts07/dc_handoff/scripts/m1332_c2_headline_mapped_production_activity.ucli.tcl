# M1332 source-only production activity window.  The frozen M979 core stops on
# accepted header and one clock after accepted token_done.  Capture only the
# mapped DUT, excluding wrapper/assertions/reset/test memory.
power -gate_level all mda sv
power tb_m1332_c2_headline_mapped_production_activity.core.dut
run
power -enable
run
power -disable
power -report $::env(M1332_SAIF_FILE) 1e-9 tb_m1332_c2_headline_mapped_production_activity.core.dut
quit
