// The compatibility shell above maps the M201 testbench DUT name to M202.
// Keeping the original scoreboard source byte-identical makes the ablation
// directly comparable while the M202-specific bound SVA proves the new path.
`include "tb_m201/tb_m201_fc2_raw4_to_descriptor4_stable_compactor.sv"
