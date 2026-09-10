// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c1s_ecp_qkv_predictor — ECP-QKV eager correlation prediction before QKV proj
// Idea packs 09/07 (FACT-style): cheap corr_score gates projection enable per tile.

`timescale 1ns/1ps
`default_nettype none

module c1s_ecp_qkv_predictor #(
  parameter int N_TILE   = 64,
  parameter int SCORE_W  = 8,
  parameter logic [SCORE_W-1:0] TH_S = 8'd2
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  // Cheap correlation / coarse attention-mass proxy (pre-projection)
  input  logic        [SCORE_W-1:0]    corr_score [N_TILE],
  // Optional OP-STW wake: force-project when wake & use_wake
  input  logic        [N_TILE-1:0]     wake_bitmap,
  input  logic                         use_wake,
  // proj_en=1 → run QKV projection MAC for that tile; skip=1 → skip
  output logic        [N_TILE-1:0]     proj_en_bitmap,
  output logic        [N_TILE-1:0]     skip_bitmap,
  output logic                         proj_valid
);

  logic [N_TILE-1:0] proj_comb;

  genvar gi;
  generate
    for (gi = 0; gi < N_TILE; gi++) begin : g_tile
      wire score_hit = (corr_score[gi] > TH_S);  // strict >
      wire wake_hit  = use_wake && wake_bitmap[gi];
      assign proj_comb[gi] = score_hit || wake_hit;
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      proj_en_bitmap <= '0;
      skip_bitmap    <= '0;
      proj_valid     <= 1'b0;
    end else begin
      proj_valid <= valid_i;
      if (valid_i) begin
        proj_en_bitmap <= proj_comb;
        skip_bitmap    <= ~proj_comb;
      end
    end
  end

endmodule

`default_nettype wire
