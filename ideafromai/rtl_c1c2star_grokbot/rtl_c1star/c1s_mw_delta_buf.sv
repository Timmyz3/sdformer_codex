// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c1s_mw_delta_buf — MW-ΔBuf motion-warped residual / delta buffer (C1*)
// Idea packs 08/09: per-tile delta = cur - ref; sparse mask when |delta| > TH_D.
// Note: delta is a packed bus (tile-major) — iverilog cannot reliably drive
// multi-bit unpacked array output ports; TB / top unpack as needed.

`timescale 1ns/1ps
`default_nettype none

module c1s_mw_delta_buf #(
  parameter int N_TILE  = 64,
  parameter int SAMP_W  = 8,
  parameter logic [SAMP_W-1:0] TH_D = 8'd0
) (
  input  logic                              clk,
  input  logic                              rst_n,
  input  logic                              valid_i,
  input  logic signed [SAMP_W-1:0]          ref_samp [N_TILE],
  input  logic signed [SAMP_W-1:0]          cur_samp [N_TILE],
  // Packed residuals: delta[i] lives in bits [i*(SAMP_W+1) +: (SAMP_W+1)]
  output logic signed [N_TILE*(SAMP_W+1)-1:0] delta,
  output logic        [N_TILE-1:0]          delta_nz,
  output logic                              delta_valid
);

  localparam int DW = SAMP_W + 1;

  logic [N_TILE-1:0]             nz_comb;
  logic signed [N_TILE*DW-1:0]   delta_comb;

  genvar gi;
  generate
    for (gi = 0; gi < N_TILE; gi++) begin : g_tile
      wire signed [SAMP_W:0] cur_sx = $signed({cur_samp[gi][SAMP_W-1], cur_samp[gi]});
      wire signed [SAMP_W:0] ref_sx = $signed({ref_samp[gi][SAMP_W-1], ref_samp[gi]});
      wire signed [SAMP_W:0] d_sx   = cur_sx - ref_sx;
      wire [SAMP_W:0] abs_d =
          d_sx[SAMP_W] ? -$unsigned(d_sx) : $unsigned(d_sx);
      assign nz_comb[gi] = (abs_d > {{1'b0}, TH_D});
      assign delta_comb[gi*DW +: DW] = d_sx;
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      delta       <= '0;
      delta_nz    <= '0;
      delta_valid <= 1'b0;
    end else begin
      delta_valid <= valid_i;
      if (valid_i) begin
        delta    <= delta_comb;
        delta_nz <= nz_comb;
      end
    end
  end

endmodule

`default_nettype wire
