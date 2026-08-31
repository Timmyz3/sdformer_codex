`timescale 1ns/1ps

// Frozen, design-independent VCS license-checkout preflight.
// Compilation of this file must create a fresh simv before the M1033
// production attempt namespace can be consumed.  It is never used for PPA.
module tb_m1030_vcs_license_checkout_preflight;
  logic marker;

  initial begin
    marker = 1'b0;
    #1 marker = 1'b1;
    #1;
    if (marker !== 1'b1) $fatal(1, "M1030 preflight marker mismatch");
    $display("PASS_M1030_VCS_LICENSE_CHECKOUT_PREFLIGHT_SOURCE");
    $finish;
  end
endmodule
