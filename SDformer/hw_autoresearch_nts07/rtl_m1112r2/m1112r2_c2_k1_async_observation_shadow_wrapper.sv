// Additive M1112r2 identity.  The M1112 observation-only functional source is
// textually reused under a fresh module name so that the two M1113 P0 repairs
// remain confined to trust/provenance checking rather than changing RTL.
`define m1112_c2_k1_async_observation_shadow_wrapper m1112r2_c2_k1_async_observation_shadow_wrapper
`include "rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv"
`undef m1112_c2_k1_async_observation_shadow_wrapper
