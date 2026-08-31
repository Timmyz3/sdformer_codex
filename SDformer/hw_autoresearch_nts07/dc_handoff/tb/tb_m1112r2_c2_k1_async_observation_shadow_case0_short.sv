// Additive M1112r2 testbench identity.  It preserves the frozen 22-signal
// atomic bitmap and 128-cycle no-short-circuit behavior byte-for-byte after
// preprocessing, while binding the fresh M1112r2 mapped top.
`define tb_m1112_c2_k1_async_observation_shadow_case0_short tb_m1112r2_c2_k1_async_observation_shadow_case0_short
`define m1112_c2_k1_async_observation_shadow_wrapper m1112r2_c2_k1_async_observation_shadow_wrapper
`include "dc_handoff/tb/tb_m1112_c2_k1_async_observation_shadow_case0_short.sv"
`undef m1112_c2_k1_async_observation_shadow_wrapper
`undef tb_m1112_c2_k1_async_observation_shadow_case0_short
