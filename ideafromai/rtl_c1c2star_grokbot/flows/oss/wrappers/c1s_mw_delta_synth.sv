// GROKBOT NEW FILE -- iscas_ssh
// Synth-only flattened MW-ΔBuf (Yosys cannot parse unpacked array ports).

`timescale 1ns/1ps
`default_nettype none

module c1s_mw_delta_buf_synth #(
  parameter int N_TILE  = 64,
  parameter int SAMP_W  = 8,
  parameter logic [SAMP_W-1:0] TH_D = 8'd0
) (
  input  logic                              clk,
  input  logic                              rst_n,
  input  logic                              valid_i,
  input  logic signed [N_TILE*SAMP_W-1:0]   ref_samp,
  input  logic signed [N_TILE*SAMP_W-1:0]   cur_samp,
  output logic signed [N_TILE*(SAMP_W+1)-1:0] delta,
  output logic        [N_TILE-1:0]          delta_nz,
  output logic                              delta_valid
);

  logic signed [SAMP_W:0] delta_comb [N_TILE];
  logic        [N_TILE-1:0] nz_comb;

  genvar gi;
  generate
    for (gi = 0; gi < N_TILE; gi++) begin : g_tile
      wire signed [SAMP_W-1:0] rs = ref_samp[gi*SAMP_W +: SAMP_W];
      wire signed [SAMP_W-1:0] cs = cur_samp[gi*SAMP_W +: SAMP_W];
      wire signed [SAMP_W:0]   cur_sx = $signed({cs[SAMP_W-1], cs});
      wire signed [SAMP_W:0]   ref_sx = $signed({rs[SAMP_W-1], rs});
      wire signed [SAMP_W:0]   d_sx   = cur_sx - ref_sx;
      wire [SAMP_W:0] abs_d =
          d_sx[SAMP_W] ? -$unsigned(d_sx) : $unsigned(d_sx);
      assign delta_comb[gi] = d_sx;
      assign nz_comb[gi]    = (abs_d > {{1'b0}, TH_D});
    end
  endgenerate

  integer ti;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      delta_nz    <= '0;
      delta_valid <= 1'b0;
      delta       <= '0;
    end else begin
      delta_valid <= valid_i;
      if (valid_i) begin
        delta_nz <= nz_comb;
        for (ti = 0; ti < N_TILE; ti++)
          delta[ti*(SAMP_W+1) +: (SAMP_W+1)] <= delta_comb[ti];
      end
    end
  end

endmodule

`default_nettype wire
