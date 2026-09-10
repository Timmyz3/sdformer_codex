// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c1s_ogec_gate — Occlusion-Gated Exact Capture (OGEC)
// Matched tiles → exact product path; unmatched → propagate/fill path.
// INNOVATION: occlusion/match gate splits ExactMatch vs Propagate — not zero-skip.
// Ablation: ABLATE_FORCE_EXACT=1 OR drive match_ok=all-1s → all ExactMatch.

`timescale 1ns/1ps
`default_nettype none

module c1s_ogec_gate #(
  parameter int N_TILE = 64,
  parameter bit ABLATE_FORCE_EXACT = 1'b0  // thin ablation cfg
) (
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic                 valid_i,
  // 1 = matched / exact path; 0 = unmatched / propagate
  input  logic [N_TILE-1:0]    match_ok,
  output logic [N_TILE-1:0]    exact_en,
  output logic [N_TILE-1:0]    prop_en,
  output logic                 ogec_valid
);

  logic [N_TILE-1:0] exact_comb;
  logic [N_TILE-1:0] prop_comb;

  wire [N_TILE-1:0] match_eff = ABLATE_FORCE_EXACT ? {N_TILE{1'b1}} : match_ok;
  assign exact_comb = match_eff;
  // Propagate only on active valid path for unmatched tiles
  assign prop_comb  = valid_i ? ~match_eff : '0;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      exact_en   <= '0;
      prop_en    <= '0;
      ogec_valid <= 1'b0;
    end else begin
      ogec_valid <= valid_i;
      if (valid_i) begin
        exact_en <= exact_comb;
        prop_en  <= ~match_eff;  // = ~exact when valid path active
      end
    end
  end

endmodule

`default_nettype wire
