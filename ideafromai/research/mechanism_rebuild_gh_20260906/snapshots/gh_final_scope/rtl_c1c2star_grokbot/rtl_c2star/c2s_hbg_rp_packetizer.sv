// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh) / for Codex fill-in
// Date: 2026-09-05
// Module: c2s_hbg_rp_packetizer — HBG-RP hybrid binary-gate + real payload
// INNOVATION: binary gate g clocks PE; payload p is NON-ABSORBABLE ATLIF amplitude
// (not Bishop-style AAC binary-only; not weight-absorbed spike).
// Claim sentence: "Hybrid binary-gate + real ATLIF payload packetizer."

`timescale 1ns/1ps
`default_nettype none

module c2s_hbg_rp_packetizer #(
  parameter int AMP_W = 8,
  parameter logic signed [AMP_W-1:0] EPS = 8'sd1
) (
  input  logic                      clk,
  input  logic                      rst_n,
  input  logic                      amp_valid,
  input  logic signed [AMP_W-1:0]   amp,
  output logic                      g,           // gate
  output logic signed [AMP_W-1:0]   p,           // payload
  output logic                      gp_valid,
  output logic                      pe_clk_en    // == g when valid path active; 0 if !amp_valid
);

  // Sign-extend so abs of most-negative (-128 for int8) is representable in AMP_W+1
  wire signed [AMP_W:0] amp_sx  = $signed({amp[AMP_W-1], amp});
  wire        [AMP_W:0] amp_abs = amp_sx[AMP_W] ? -$unsigned(amp_sx) : $unsigned(amp_sx);

  // Strict > EPS (EPS=1 → |amp|==1 does not gate)
  wire g_comb = amp_valid && (amp_abs > $unsigned({{1'b0}, EPS}));

  assign g         = g_comb;
  assign p         = g_comb ? amp : {AMP_W{1'b0}};
  assign pe_clk_en = g_comb;  // == g; forced 0 when !amp_valid

  // gp_valid: registered handshake (amp_valid delayed 1 cycle)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      gp_valid <= 1'b0;
    else
      gp_valid <= amp_valid;
  end

endmodule

`default_nettype wire
