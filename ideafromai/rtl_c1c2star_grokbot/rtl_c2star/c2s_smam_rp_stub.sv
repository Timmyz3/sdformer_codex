// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c2s_smam_rp_stub — thin wrapper over real c2s_smam_rp (keeps top-compatible ports)

`timescale 1ns/1ps
`default_nettype none

module c2s_smam_rp_stub #(
  parameter int AMP_W = 8
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    gp_valid,   // from HBG-RP
  input  logic                    g,          // binary gate
  input  logic signed [AMP_W-1:0] p,          // real payload
  output logic                    mac_en,
  output logic                    mask_add_en,
  output logic signed [AMP_W-1:0] payload_q,
  output logic                    out_valid
);

  // mask_add_result unused at stub port; real dual-rail lives in c2s_smam_rp
  logic signed [15:0] mask_add_result_nc;

  c2s_smam_rp #(
    .AMP_W(AMP_W),
    .ACC_W(16)
  ) u_smam_rp (
    .clk            (clk),
    .rst_n          (rst_n),
    .valid          (gp_valid),
    .spike_gate     (g),
    .payload        (p),
    .mac_en         (mac_en),
    .mask_add_en    (mask_add_en),
    .mac_payload    (payload_q),
    .mask_add_result(mask_add_result_nc),
    .out_valid      (out_valid)
  );

endmodule

`default_nettype wire
