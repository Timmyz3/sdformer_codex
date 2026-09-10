// GROKBOT NEW FILE -- iscas_ssh
// Synth-only flattened ECP-QKV (Yosys cannot parse unpacked array ports).

`timescale 1ns/1ps
`default_nettype none

module c1s_ecp_qkv_predictor_synth #(
  parameter int N_TILE   = 64,
  parameter int SCORE_W  = 8,
  parameter logic [SCORE_W-1:0] TH_S = 8'd2
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic        [N_TILE*SCORE_W-1:0] corr_score,
  input  logic        [N_TILE-1:0]     wake_bitmap,
  input  logic                         use_wake,
  output logic        [N_TILE-1:0]     proj_en_bitmap,
  output logic        [N_TILE-1:0]     skip_bitmap,
  output logic                         proj_valid
);

  logic [N_TILE-1:0] proj_comb;

  genvar gi;
  generate
    for (gi = 0; gi < N_TILE; gi++) begin : g_tile
      wire [SCORE_W-1:0] sc = corr_score[gi*SCORE_W +: SCORE_W];
      wire score_hit = (sc > TH_S);
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
